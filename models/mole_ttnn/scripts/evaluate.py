"""
Evaluation script for MoLE
"""
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mole import MoLE, MoLEConfig, MoLETTNN
from utils.data_loader import get_dataloader
from utils.metrics import MetricsTracker, metric, compute_expert_specialization


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MoLE model')
    
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
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--num_experts', type=int, default=4,
                       help='Number of experts')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Use top-k expert selection')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    return parser.parse_args()


def load_model(args, enc_in):
    """Load model from checkpoint"""
    # Create model
    config = MoLEConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=enc_in,
        num_experts=args.num_experts,
        top_k=args.top_k
    )
    model = config.create_model()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device, return_predictions=False):
    """Evaluate model on dataloader"""
    metrics_tracker = MetricsTracker()
    all_predictions = []
    all_targets = []
    all_router_weights = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass with weights
            outputs, weights, _ = model(batch_x, return_weights=True)
            
            # Update metrics
            metrics_tracker.update(outputs, batch_y)
            
            if return_predictions:
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                all_router_weights.append(weights.cpu().numpy())
    
    metrics = metrics_tracker.get_metrics()
    
    if return_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        router_weights = np.concatenate(all_router_weights, axis=0)
        return metrics, predictions, targets, router_weights
    
    return metrics


def visualize_results(predictions, targets, router_weights, output_dir, num_samples=5):
    """Visualize predictions and expert usage"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sample predictions
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        # Plot first feature
        ax.plot(targets[i, :, 0], label='Ground Truth', alpha=0.8)
        ax.plot(predictions[i, :, 0], label='Prediction', alpha=0.8)
        ax.set_title(f'Sample {i+1} - First Feature')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150)
    plt.close()
    
    # 2. Expert usage heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(router_weights[:100].T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Expert')
    ax.set_title('Router Weights (First 100 samples)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expert_usage.png'), dpi=150)
    plt.close()
    
    # 3. Average expert usage
    avg_usage = router_weights.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(avg_usage)), avg_usage)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Average Weight')
    ax.set_title('Average Expert Usage')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_expert_usage.png'), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            # Update args with config values
            for key, value in config.items():
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, value)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        root_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        flag='test',
        scale=True
    )
    
    # Get number of features
    sample_x, _ = next(iter(test_loader))
    enc_in = sample_x.shape[-1]
    
    print(f"Data loaded: seq_len={args.seq_len}, pred_len={args.pred_len}, enc_in={enc_in}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args, enc_in)
    print("Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions, targets, router_weights = evaluate_model(
        model, test_loader, args.device, return_predictions=True
    )
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.6f}")
    print(f"  MSPE: {metrics['mspe']:.6f}")
    
    # Expert specialization analysis
    specialization = compute_expert_specialization(router_weights, args.num_experts)
    print("\nExpert Specialization:")
    print(f"  Average usage: {specialization['avg_usage']}")
    print(f"  Entropy: {specialization['entropy']:.4f}")
    print(f"  Normalized entropy: {specialization['normalized_entropy']:.4f}")
    print(f"  Gini coefficient: {specialization['gini']:.4f}")
    print(f"  Most used expert: {specialization['max_expert']}")
    print(f"  Least used expert: {specialization['min_expert']}")
    
    # Save results
    results = {
        'metrics': metrics,
        'expert_specialization': specialization,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'results_{args.dataset}_{args.pred_len}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Visualize
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_results(predictions, targets, router_weights, args.output_dir)


if __name__ == '__main__':
    main()
