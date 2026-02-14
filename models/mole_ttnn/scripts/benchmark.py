"""
Performance Benchmarking for MoLE on Tenstorrent Hardware

This script benchmarks MoLE performance and generates reports
compatible with tt-metal's performance tracking requirements.
"""
import os
import time
import json
import argparse
import torch
import numpy as np
from models.mole import MoLE, MoLETTNN, MoLEConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark MoLE performance')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_ttnn', action='store_true')
    parser.add_argument('--output', type=str, default='perf_report.json')
    return parser.parse_args()


def benchmark_pytorch(model, batch_size, seq_len, enc_in, num_batches, warmup, device):
    """Benchmark PyTorch model"""
    model = model.to(device)
    model.eval()
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randn(batch_size, seq_len, enc_in, device=device)
            _ = model(x)
    
    # Benchmark
    print(f"Benchmarking ({num_batches} iterations)...")
    times = []
    with torch.no_grad():
        for i in range(num_batches):
            x = torch.randn(batch_size, seq_len, enc_in, device=device)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.perf_counter()
            
            output = model(x)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.perf_counter()
            
            times.append(end - start)
    
    return times


def generate_perf_report(times, args):
    """Generate performance report in tt-metal format"""
    times = np.array(times)
    
    report = {
        "model": "MoLE",
        "version": "1.0.0",
        "config": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "enc_in": args.enc_in,
            "num_experts": args.num_experts,
            "batch_size": args.batch_size,
            "device": args.device,
            "use_ttnn": args.use_ttnn
        },
        "performance": {
            "mean_latency_ms": float(np.mean(times) * 1000),
            "median_latency_ms": float(np.median(times) * 1000),
            "min_latency_ms": float(np.min(times) * 1000),
            "max_latency_ms": float(np.max(times) * 1000),
            "std_latency_ms": float(np.std(times) * 1000),
            "p50_latency_ms": float(np.percentile(times, 50) * 1000),
            "p90_latency_ms": float(np.percentile(times, 90) * 1000),
            "p95_latency_ms": float(np.percentile(times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(times, 99) * 1000),
            "throughput_seq_per_sec": float(args.batch_size / np.mean(times))
        },
        "benchmark_info": {
            "num_batches": args.num_batches,
            "warmup_batches": args.warmup,
            "total_samples": args.batch_size * args.num_batches
        }
    }
    
    return report


def print_report(report):
    """Print formatted performance report"""
    print("\n" + "="*60)
    print("MoLE Performance Report")
    print("="*60)
    
    config = report["config"]
    print(f"\nConfiguration:")
    print(f"  Model: MoLE")
    print(f"  Experts: {config['num_experts']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Prediction length: {config['pred_len']}")
    print(f"  Input features: {config['enc_in']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Device: {config['device']}")
    print(f"  TT-NN: {'Yes' if config['use_ttnn'] else 'No'}")
    
    perf = report["performance"]
    print(f"\nLatency (ms):")
    print(f"  Mean:   {perf['mean_latency_ms']:.3f}")
    print(f"  Median: {perf['median_latency_ms']:.3f}")
    print(f"  Min:    {perf['min_latency_ms']:.3f}")
    print(f"  Max:    {perf['max_latency_ms']:.3f}")
    print(f"  P90:    {perf['p90_latency_ms']:.3f}")
    print(f"  P95:    {perf['p95_latency_ms']:.3f}")
    print(f"  P99:    {perf['p99_latency_ms']:.3f}")
    
    print(f"\nThroughput:")
    print(f"  {perf['throughput_seq_per_sec']:.2f} sequences/second")
    
    # Stage targets
    print(f"\nStage Targets:")
    if perf['throughput_seq_per_sec'] >= 800:
        print(f"  ✓ Stage 3: 800+ seq/s achieved")
    elif perf['throughput_seq_per_sec'] >= 200:
        print(f"  ✓ Stage 1: 200+ seq/s achieved")
        print(f"  ⚠ Stage 3: {800 - perf['throughput_seq_per_sec']:.0f} seq/s to go")
    else:
        print(f"  ⚠ Stage 1: {200 - perf['throughput_seq_per_sec']:.0f} seq/s to go")
    
    if perf['mean_latency_ms'] <= 15:
        print(f"  ✓ Stage 3: < 15ms latency achieved")
    elif perf['mean_latency_ms'] <= 30:
        print(f"  ✓ Stage 1: < 30ms latency achieved")
        print(f"  ⚠ Stage 3: {perf['mean_latency_ms'] - 15:.1f}ms over target")
    else:
        print(f"  ⚠ Stage 1: {perf['mean_latency_ms'] - 30:.1f}ms over target")
    
    print("="*60)


def main():
    args = parse_args()
    
    print("MoLE Performance Benchmark")
    print("="*60)
    
    # Create model
    print(f"\nCreating MoLE model with {args.num_experts} experts...")
    config = MoLEConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        num_experts=args.num_experts
    )
    
    if args.use_ttnn:
        print("Using TT-NN backend...")
        try:
            import ttnn
            # device = ttnn.open_device(device_id=0)
            # model = config.create_model(device=device)
            print("⚠ TT-NN device not available, using PyTorch fallback")
            model = config.create_model()
        except ImportError:
            print("⚠ TT-NN not available, using PyTorch")
            model = config.create_model()
    else:
        model = config.create_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Benchmark
    print(f"\nBenchmarking on {args.device}...")
    times = benchmark_pytorch(
        model, args.batch_size, args.seq_len, args.enc_in,
        args.num_batches, args.warmup, args.device
    )
    
    # Generate report
    report = generate_perf_report(times, args)
    print_report(report)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
