import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB import MMCIFParser
import random
import yaml
from sklearn.model_selection import KFold
from tqdm import tqdm
import wandb
from scheduler import CosineAnnealingWarmupRestarts
from scipy.stats import spearmanr, pearsonr

# Paths to your data files
FASTA_FILE = '/home/hkim/YMC2/transformer/rcsb_pdb_6K9K.fasta'
DOMAINS_FILE = '/home/hkim/YMC2/transformer/domains.yaml'
CIF_FILE = '/home/hkim/YMC2/transformer/alphafold/fold_atm_3056aa_model_0.cif'
CSV_FILES = {
    'missense': '/home/hkim/YMC2/transformer/data/train_strict_240918.csv',
    'synonymous': '/home/hkim/YMC2/transformer/data/train_strict_syn_240918.csv',
    'nonsense': '/home/hkim/YMC2/transformer/data/train_non_240918.csv'
}

# Load protein sequence
record = SeqIO.read(FASTA_FILE, 'fasta')
amino_acid_sequence_str = str(record.seq).upper()
sequence_length = len(amino_acid_sequence_str)

# Map amino acids to indices
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
num_amino_acids = len(amino_acids) + 1  # +1 for unknown amino acids

# Convert sequence to indices
amino_acid_sequence = [
    aa_to_index.get(aa, num_amino_acids - 1)
    for aa in amino_acid_sequence_str
]
amino_acid_sequence = torch.tensor(amino_acid_sequence, dtype=torch.long)

# Load domain annotations
with open(DOMAINS_FILE, 'r') as f:
    domains_data = yaml.safe_load(f)

domain_names = list(domains_data.keys())
num_domains = len(domain_names) + 1  # +1 for non-domain
domain_to_index = {domain: idx + 1 for idx, domain in enumerate(domain_names)}

# Initialize domain indices (0 for non-domain regions)
domains = torch.zeros(sequence_length, dtype=torch.long)
for domain_name, positions in domains_data.items():
    startpos = positions['startpos']
    endpos = positions['endpos']
    domain_idx = domain_to_index[domain_name]
    start_idx = startpos - 1  # Convert to 0-based index
    end_idx = endpos  # endpos is inclusive
    domains[start_idx:end_idx] = domain_idx

# Parse CIF file
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure('ATM', CIF_FILE)

# Extract coordinates
coordinates_list = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            if 'CA' in residue:
                ca_atom = residue['CA']
                coord = ca_atom.get_coord()
                coordinates_list.append(coord)
            else:
                coordinates_list.append(np.array([0.0, 0.0, 0.0]))

# Convert coordinates to tensor
coordinates = np.array(coordinates_list) / 35  # Scale (Std: 32.95, Mean: -0.615)
coordinates = torch.from_numpy(coordinates).float()

# Ensure coordinates match sequence length
if len(coordinates) != sequence_length:
    raise ValueError("Coordinates length does not match sequence length.")

def random_rotation(coords):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    R = np.dot(Rz, np.dot(Ry, Rx))
    coords_rotated = np.dot(coords, R.T)
    return torch.from_numpy(coords_rotated).float()

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, csv_files, amino_acid_sequence, domains, coordinates, augment=False):
        self.missense_data = pd.read_csv(csv_files['missense'])
        self.synonymous_data = pd.read_csv(csv_files['synonymous'])
        self.nonsense_data = pd.read_csv(csv_files['nonsense'])

        self.amino_acid_sequence = amino_acid_sequence
        self.domains = domains
        self.coordinates = coordinates
        self.augment = augment

        # 각 데이터의 길이
        self.lengths = {
            'missense': len(self.missense_data),
            'synonymous': len(self.synonymous_data),
            'nonsense': len(self.nonsense_data)
        }

    def __len__(self):
        return sum(self.lengths.values())

    def __getitem__(self, idx):
        if idx < self.lengths['missense']:
            row = self.missense_data.iloc[idx]
            mutated_amino_acid_type = 'missense'
        elif idx < self.lengths['missense'] + self.lengths['synonymous']:
            row = self.synonymous_data.iloc[idx - self.lengths['missense']]
            mutated_amino_acid_type = 'synonymous'
        else:
            row = self.nonsense_data.iloc[idx - self.lengths['missense'] - self.lengths['synonymous']]
            mutated_amino_acid_type = 'nonsense'

        aa_position = int(row['aa_position']) - 1  # Convert to 0-based index
        mutated_amino_acid = row['mutated_amino_acid']
        pathogenicity_score = row['asinh_score']

        features = row[['SIFT_score', 'FATHMM_score', 'MutationTaster_score', 'LRT_score', 'DANN_score', 
                        'Polyphen2_HVAR_score', 'PROVEAN_score', 'REVEL_score', 'CADD_phred', 
                        'GERP++_RS', 'ESM1b_score', 'EVE_score', 'AlphaMissense_score', 
                        'phyloP100way_vertebrate', 'boostDM_score', 'splice_score']]

        # aa_sequence copy
        aa_sequence = self.amino_acid_sequence.clone()

        if mutated_amino_acid_type == 'synonymous':
            aa_sequence[aa_position] = 21  # [SYN] token
        elif mutated_amino_acid_type == 'nonsense':
            aa_sequence[aa_position] = 22  # [NON] token
        else:
            mutated_amino_acid_index = aa_to_index.get(mutated_amino_acid, num_amino_acids - 1)
            aa_sequence[aa_position] = mutated_amino_acid_index

        coords = random_rotation(self.coordinates.numpy()) if self.augment else self.coordinates

        return {
            'aa_sequence': aa_sequence,
            'domains': self.domains,
            'coordinates': coords,
            'aa_position': aa_position,
            'features': torch.tensor(features, dtype=torch.float),
            'pathogenicity_score': torch.tensor(pathogenicity_score, dtype=torch.float)
        }

# DataLoader function with dynamic sampling
def get_data_loader(dataset, indices, batch_size=16):
    # missense 90%, synonymous 5%, nonsense 5%
    weights = [
        0.96 if idx < dataset.lengths['missense'] else 0.02 * 16276/5204
        if idx < dataset.lengths['missense'] + dataset.lengths['synonymous'] else 0.02 * 16276/1193
        for idx in indices
    ]

    sampler = WeightedRandomSampler(weights, len(indices))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=8)

def get_data_loader(dataset, indices, batch_size=16):
    missense_end = dataset.lengths['missense']
    synonymous_end = missense_end + dataset.lengths['synonymous']
    nonsense_end = synonymous_end + dataset.lengths['nonsense']

    missense_weight = 0.9  # 90% missense
    synonymous_weight = 0.05 * 16276/5204  # 5% synonymous
    nonsense_weight = 0.05 * 16276/1193  # 5% nonsense

    weights = [0] * len(dataset)

    for idx in indices:
        if idx < missense_end:
            weights[idx] = missense_weight
        elif idx < synonymous_end:
            weights[idx] = synonymous_weight
        else:
            weights[idx] = nonsense_weight

    # WeightedRandomSampler를 사용하여 인덱스에 있는 데이터만 가중치에 따라 샘플링
    sampler = WeightedRandomSampler(weights, len(indices))

    # DataLoader 반환
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=8)

# Model definition
class TransformerRegressor(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, sequence_length, num_amino_acids, num_domains):
        super(TransformerRegressor, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        # Embedding layers
        self.aa_embedding = nn.Embedding(num_embeddings=num_amino_acids + 2, embedding_dim=embedding_dim)
        self.domain_embedding = nn.Embedding(num_embeddings=num_domains, embedding_dim=embedding_dim)

        # Positional MLP
        self.positional_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, domains, coordinates, aa_position, features):
        """
        x: Tensor of shape (batch_size, sequence_length)
        domains: Tensor of shape (batch_size, sequence_length)
        coordinates: Tensor of shape (batch_size, sequence_length, 3)
        features: Tensor of shape (batch_size, 16)
        aa_position: Tensor of shape (batch_size,)
        """
        # Embeddings
        aa_embed = self.aa_embedding(x)
        domain_embed = self.domain_embedding(domains)
        positional_embed = self.positional_mlp(coordinates)

        sum_embed = aa_embed + domain_embed + positional_embed

        transformer_output = self.transformer_encoder(sum_embed)

        mutated_embed = transformer_output[torch.arange(x.size(0)), aa_position]

        output = self.fc(torch.cat([mutated_embed, features], dim=1))

        return output.squeeze()


def mse_loss(output, target):
    return F.mse_loss(output, target)


def compute_spearmanr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


def compute_pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


# Initialize wandb
wandb.init(project="DeepATM_combined", config={
    "num_heads": 8, # 8
    "num_layers": 2, # 2
    "embedding_dim": 64,
    "learning_rate": 1e-3, # 5e-4
    "weight_decay": 1e-2, # 1e-2
    "batch_size": 20, # 16
    "num_epochs": 150, # 150
    "augment": False,
    "scaling": "asinh"
})

MODEL_DIR = '/home/hkim/YMC2/transformer/models/combined/H{}_L{}_E{}_L{}_W{}_B{}_E{}{}/'.format(
    wandb.config.num_heads,
    wandb.config.num_layers,
    wandb.config.embedding_dim,
    wandb.config.learning_rate,
    wandb.config.weight_decay,
    wandb.config.batch_size,
    wandb.config.num_epochs,
    "_augment" if wandb.config.augment else "",
)
os.makedirs(MODEL_DIR, exist_ok=True)

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 20  # Stop training if no improvement for 10 epochs
BEST_VAL_LOSS = float('inf')  # Initialize with infinity
EARLY_STOPPING_COUNT = 0

def train_model(dataset, num_amino_acids, num_domains, embedding_dim, n_splits=5, batch_size=16, num_epochs=10, learning_rate=1e-4, weight_decay=1e-2, num_heads=8, num_layers=4):
    global BEST_VAL_LOSS, EARLY_STOPPING_COUNT
    # Device config
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f'Now {num_gpus} GPUs available.')
    device_ids = [1, 2]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        
        print(f"Fold {fold + 1}/{n_splits}")

        # DataLoaders
        train_loader = get_data_loader(dataset, train_indices, batch_size)
        val_loader = get_data_loader(dataset, val_indices, batch_size)

        # Model init
        model = TransformerRegressor(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            sequence_length=sequence_length,
            num_amino_acids=num_amino_acids,
            num_domains=num_domains
        )

        model = model.to(device)

        if device_ids:
            model = nn.DataParallel(model, device_ids=device_ids)
            print(f"Using {len(device_ids)} GPUs.")

        use_amp = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=learning_rate/100)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=10, cycle_mult=2, max_lr=learning_rate, min_lr=learning_rate/100, gamma=0.8)

        BEST_VAL_LOSS = float('inf')  # Track the best val loss for this fold

        pbar = tqdm(range(num_epochs))
        # Training loop
        for epoch in pbar:

            model.train()
            train_losses = []
            for i, batch in enumerate(train_loader):
                embeddings = batch['aa_sequence'].to(device)
                domains_batch = batch['domains'].to(device)
                coordinates = batch['coordinates'].to(device)
                aa_position = batch['aa_position'].to(device)
                am_score = batch['features'].to(device)
                targets = batch['pathogenicity_score'].to(device)

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = model(embeddings, domains_batch, coordinates, aa_position, am_score)
                    loss = mse_loss(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(epoch + i / len(train_loader))

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            valid_count = 0

            y_true = np.zeros((len(val_indices)), dtype=np.float32)
            y_pred = np.zeros((len(val_indices)), dtype=np.float32)

            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch['aa_sequence'].to(device)
                    domains_batch = batch['domains'].to(device)
                    coordinates = batch['coordinates'].to(device)
                    aa_position = batch['aa_position'].to(device)
                    am_score = batch['features'].to(device)
                    targets = batch['pathogenicity_score'].to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        outputs = model(embeddings, domains_batch, coordinates, aa_position, am_score)
                        loss = mse_loss(outputs, targets)

                    val_losses.append(loss.item())
                    batch_size_actual = targets.size(0)
                    y_true[valid_count:valid_count + batch_size_actual] = targets.cpu().numpy()
                    y_pred[valid_count:valid_count + batch_size_actual] = outputs.cpu().numpy()

                    valid_count += batch_size_actual

            val_loss = np.mean(val_losses)
            spearman_corr = compute_spearmanr(y_true, y_pred)
            pearson_corr = compute_pearsonr(y_true, y_pred)

            print(f"E {epoch + 1}/{num_epochs} | "
                  f"TL: {np.mean(train_losses):.4f} | "
                  f"VL: {val_loss:.4f} | "
                  f"SR: {spearman_corr:.4f} | "
                  f"PR: {pearson_corr:.4f}")

            # Log to wandb
            wandb.log({
                'train_loss': np.mean(train_losses),
                'val_loss': val_loss,
                'spearman_corr': spearman_corr,
                'pearson_corr': pearson_corr,
                'epoch': epoch + 1
            })
            
            if val_loss < BEST_VAL_LOSS:
                print(f"Valid loss improved from {BEST_VAL_LOSS:.4f} to {val_loss:.4f}, saving model checkpoint...")
                BEST_VAL_LOSS = val_loss
                EARLY_STOPPING_COUNT = 0
                # Save checkpoint
                torch.save(model.module.state_dict(), f"{MODEL_DIR}best_model_fold{fold+1}.pth")
            else:
                EARLY_STOPPING_COUNT += 1

            if EARLY_STOPPING_COUNT >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # End of fold, save final model for this fold
        torch.save(model.module.state_dict(), f"{MODEL_DIR}final_model_fold{fold+1}.pth")
        print(f"Final model for fold {fold + 1} saved.")

        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'spearman_corr': spearman_corr,
            'pearson_corr': pearson_corr
        })

    return fold_results


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create dataset
    dataset = ProteinDataset(
        csv_files=CSV_FILES,
        amino_acid_sequence=amino_acid_sequence,
        domains=domains,
        coordinates=coordinates,
        augment=wandb.config.augment  # Set to True to apply data augmentation
    )

    # Train model with 5-fold cross-validation
    results = train_model(
        dataset,
        num_amino_acids=num_amino_acids,
        num_domains=num_domains,
        embedding_dim=wandb.config.embedding_dim,
        n_splits=5,
        batch_size=wandb.config.batch_size,
        num_epochs=wandb.config.num_epochs,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        num_heads=wandb.config.num_heads,
        num_layers=wandb.config.num_layers,
    )

    # Print results
    for res in results:
        print(f"Fold {res['fold']}: "
              f"Val Loss: {res['val_loss']:.4f}, "
              f"Spearman: {res['spearman_corr']:.4f}, "
              f"Pearson: {res['pearson_corr']:.4f}")

    # End the wandb run
    wandb.finish()