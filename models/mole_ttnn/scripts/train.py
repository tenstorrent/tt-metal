"""
Training script for MoLE
"""
import os
import argparse
import torch
import torch.nn as nn
from models.mole import MoLE, MoLEConfig
from utils.data_loader import load_data
from utils.trainer import Trainer
from utils.metrics import MetricsTracker, metric


def parse_args():
    parser = argparse.ArgumentParser(description='Train MoLE model')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                       help='Prediction length')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='MoLE',
                       help='Model name')
    parser.add_argument('--num_experts', type=int, default=4,
                       help='Number of experts')
    parser.add_argument('--expert_type', type=str, default='dlinear',
                       choices=['dlinear', 'rlinear', 'rmlp'],
                       help='Type of expert model')
    parser.add_argument('--individual', action='store_true',
                       help='Use individual linear layers per feature')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Use top-k expert selection')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use_aux_loss', action='store_true',
                       help='Use auxiliary load balancing loss')
    parser.add_argument('--aux_loss_coef', type=float, default=0.01,
                       help='Coefficient for auxiliary loss')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        root_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        scale=True
    )
    
    # Get number of features from data
    sample_x, _ = next(iter(train_loader))
    enc_in = sample_x.shape[-1]
    
    print(f"Data loaded: seq_len={args.seq_len}, pred_len={args.pred_len}, enc_in={enc_in}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print(f"Creating MoLE model with {args.num_experts} experts...")
    config = MoLEConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=enc_in,
        num_experts=args.num_experts,
        expert_type=args.expert_type,
        individual=args.individual,
        top_k=args.top_k
    )
    model = config.create_model()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        use_aux_loss=args.use_aux_loss,
        aux_loss_coef=args.aux_loss_coef
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            
            outputs = model(batch_x)
            metrics_tracker.update(outputs, batch_y)
    
    test_metrics = metrics_tracker.get_metrics()
    print("\nTest Metrics:")
    metrics_tracker.print_metrics()
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{args.pred_len}.pth')
    trainer.save_model(model_path)
    
    # Save config
    import json
    config_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{args.pred_len}_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{args.pred_len}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        }, f, indent=2)
    
    print(f"\nModel saved to {model_path}")
    print(f"Config saved to {config_path}")
    print(f"Metrics saved to {metrics_path}")
    
    # Expert usage
    if hasattr(model, 'get_expert_usage'):
        usage = model.get_expert_usage()
        print(f"\nExpert Usage: {usage}")


if __name__ == '__main__':
    main()
