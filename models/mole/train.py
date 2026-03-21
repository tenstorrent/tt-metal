"""
Training script for MoLE on Tenstorrent hardware

Implements end-to-end training with experts and router joint learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import os
from typing import Optional, Tuple
import time

from mole_ttnn import (
    MoLE, create_mole_dlinear, create_mole_rlinear,
    TimestampEmbedding, mole_loss
)


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        timestamps: Optional[np.ndarray] = None,
        stride: int = 1
    ):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.timestamps = timestamps
        self.stride = stride
        
        # Calculate valid indices
        self.valid_indices = list(range(0, len(data) - seq_len - pred_len + 1, stride))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Input sequence
        x = self.data[start_idx:start_idx + self.seq_len]
        
        # Target sequence
        y = self.data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
        
        # Timestamp embedding (first timestamp of input)
        if self.timestamps is not None:
            ts = self.timestamps[start_idx]
            if hasattr(ts, 'month'):  # pandas Timestamp
                ts_embed = TimestampEmbedding.embed_datetime(ts)
            else:
                ts_embed = torch.tensor(ts, dtype=torch.float32)
        else:
            ts_embed = torch.zeros(4)  # Default timestamp dim
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            ts_embed
        )


def load_dataset(
    dataset_name: str,
    data_path: str,
    seq_len: int,
    pred_len: int
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """
    Load standard time series datasets
    
    Supports: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
    """
    data_file = Path(data_path) / f"{dataset_name}.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Remove date column if present
    if 'date' in df.columns:
        timestamps = pd.to_datetime(df['date'])
        df = df.drop('date', axis=1)
    else:
        timestamps = None
    
    data = df.values
    
    # Split data
    n = len(data)
    if 'ETT' in dataset_name:
        train_size = int(n * 0.6)
        val_size = int(n * 0.2)
    else:
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    train_timestamps = timestamps[:train_size] if timestamps is not None else None
    val_timestamps = timestamps[train_size:train_size + val_size] if timestamps is not None else None
    test_timestamps = timestamps[train_size + val_size:] if timestamps is not None else None
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, train_timestamps)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len, val_timestamps)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len, test_timestamps)
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(
    model: MoLE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    diversity_weight: float = 0.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    count = 0
    
    for batch_x, batch_y, batch_ts in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_ts = batch_ts.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, expert_outputs, weights = model.get_expert_outputs(batch_x, batch_ts)
        
        # Compute loss
        loss = mole_loss(output, batch_y, weights, diversity_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else 0


def evaluate(
    model: MoLE,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on dataset"""
    model.eval()
    total_mse = 0
    total_mae = 0
    count = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ts in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_ts = batch_ts.to(device)
            
            output = model(batch_x, batch_ts)
            
            mse = torch.nn.functional.mse_loss(output, batch_y).item()
            mae = torch.nn.functional.l1_loss(output, batch_y).item()
            
            total_mse += mse
            total_mae += mae
            count += 1
    
    return total_mse / count, total_mae / count


def train(
    model: MoLE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.005,
    diversity_weight: float = 0.0,
    patience: int = 10,
    save_path: Optional[str] = None
):
    """Full training loop"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of experts: {model.num_experts}")
    print(f"Expert type: {model.expert_type}")
    print("-" * 60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, diversity_weight)
        
        # Validate
        val_mse, val_mae = evaluate(model, val_loader, device)
        
        # Test
        test_mse, test_mae = evaluate(model, test_loader, device)
        
        # Scheduler step
        scheduler.step(val_mse)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}")
        print(f"  Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
        
        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  鉁?Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print()
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train MoLE on time series data')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--seq_len', type=int, default=336,
                        help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of experts')
    parser.add_argument('--expert_type', type=str, default='dlinear',
                        choices=['dlinear', 'rlinear'],
                        help='Expert model type')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--diversity_weight', type=float, default=0.0,
                        help='Diversity regularization weight')
    parser.add_argument('--router_dropout', type=float, default=0.0,
                        help='Router dropout rate')
    parser.add_argument('--expert_dropout', type=float, default=0.0,
                        help='Expert dropout rate')
    parser.add_argument('--individual', action='store_true',
                        help='Use channel-independent linear layers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save best model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.dataset, args.data_path, args.seq_len, args.pred_len
    )
    
    enc_in = train_dataset.data.shape[1]
    print(f"Number of channels: {enc_in}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.expert_type == 'dlinear':
        model = create_mole_dlinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            num_experts=args.num_experts,
            individual=args.individual,
            router_dropout=args.router_dropout,
            expert_dropout=args.expert_dropout
        )
    else:
        model = create_mole_rlinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            num_experts=args.num_experts,
            individual=args.individual,
            router_dropout=args.router_dropout,
            expert_dropout=args.expert_dropout
        )
    
    model = model.to(device)
    
    # Save path
    if args.save_path is None:
        args.save_path = f"checkpoints/mole_{args.expert_type}_{args.dataset}_pred{args.pred_len}_seed{args.seed}.pt"
        os.makedirs('checkpoints', exist_ok=True)
    
    # Train
    best_val_loss = train(
        model, train_loader, val_loader, test_loader,
        device, args.epochs, args.lr, args.diversity_weight,
        save_path=args.save_path
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    model.load_state_dict(torch.load(args.save_path))
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Save results
    results = {
        'dataset': args.dataset,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'num_experts': args.num_experts,
        'expert_type': args.expert_type,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'best_val_loss': best_val_loss,
        'seed': args.seed
    }
    
    results_path = args.save_path.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
