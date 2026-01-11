#!/usr/bin/env python3
"""
Dataset benchmarking script for PatchTSMixer TTNN implementation.

This script:
1. Loads ETTh1, ETTh2, or Weather datasets
2. Runs inference with PyTorch reference and TTNN implementation
3. Computes MSE, MAE, RMSE, and correlation metrics
4. Validates TTNN output is within 5% of PyTorch reference
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ttnn
from models.demos.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from models.demos.patchtsmixer.tt.model_processing import (
    preprocess_embedding_proj,
    preprocess_forecast_head,
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_linear,
    preprocess_positional_encoding,
)
from models.demos.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerModelForForecasting


def download_ett_dataset(dataset_name="ETTh2", data_dir="./data"):
    """Download ETT dataset from GitHub"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    dataset_path = data_dir / f"{dataset_name}.csv"

    if not dataset_path.exists():
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset_name}.csv"
        print(f"Downloading {dataset_name} from {url}...")
        df = pd.read_csv(url)
        df.to_csv(dataset_path, index=False)
        print(f"Saved to {dataset_path}")

    return pd.read_csv(dataset_path)


def prepare_ett_data(dataset_name="ETTh2", context_length=512, prediction_length=96, data_dir="./data"):
    """Prepare ETT dataset for evaluation"""

    df = download_ett_dataset(dataset_name, data_dir)

    # ETT datasets have 'date' as timestamp and 7 features
    timestamp_column = "date"
    feature_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # Standard ETT splits (in hours for ETTh datasets)
    if dataset_name in ["ETTh1", "ETTh2"]:
        train_size = 12 * 30 * 24  # 12 months
        val_size = 4 * 30 * 24  # 4 months
        # test is remaining
    else:  # ETTm1, ETTm2
        train_size = 12 * 30 * 24 * 4  # 12 months in 15-min intervals
        val_size = 4 * 30 * 24 * 4  # 4 months

    # Extract test data
    test_start = train_size + val_size
    test_data = df.iloc[test_start:].reset_index(drop=True)

    # Normalize using train statistics (for proper evaluation)
    train_data = df.iloc[:train_size]
    feature_mean = train_data[feature_columns].mean()
    feature_std = train_data[feature_columns].std()

    test_data[feature_columns] = (test_data[feature_columns] - feature_mean) / feature_std

    return test_data, feature_columns, feature_mean, feature_std


def create_forecast_samples(data, feature_columns, context_length, prediction_length, num_samples=100):
    """Create forecast samples from time series data"""
    samples = []
    targets = []

    values = data[feature_columns].values
    max_start = len(values) - context_length - prediction_length

    # Sample uniformly across the test set
    start_indices = np.linspace(0, max_start, num_samples, dtype=int)

    for start_idx in start_indices:
        # Past values (context)
        past = values[start_idx : start_idx + context_length]
        # Future values (target)
        future = values[start_idx + context_length : start_idx + context_length + prediction_length]

        samples.append(past)
        targets.append(future)

    return np.array(samples, dtype=np.float32), np.array(targets, dtype=np.float32)


def compute_metrics(predictions, targets):
    """Compute MSE, MAE, RMSE, and correlation"""
    # Flatten for metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # MSE, MAE, RMSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(mse)

    # Correlation
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
    }


def load_or_create_model(config, checkpoint_path=None):
    """Load a trained model or create a new one"""
    model = PatchTSMixerModelForForecasting(**config).eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️  Using randomly initialized model (no checkpoint provided)")
        print("   For meaningful benchmarks, train a model first or use pretrained weights")

    return model


def preprocess_parameters_for_ttnn(torch_model, device, num_layers, use_gated_attn):
    """Convert PyTorch model parameters to TTNN format"""
    base = "model"
    sd = torch_model.state_dict()
    state_dict = {}

    # Embedding
    state_dict[f"{base}.patch_embed.proj.weight"] = sd["patch_embed.proj.weight"]
    state_dict[f"{base}.patch_embed.proj.bias"] = sd["patch_embed.proj.bias"]

    # Positional encoding
    state_dict[f"{base}.pos_enc.pe"] = sd["pos_enc.pe"]

    # Mixer layers
    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        for mixer in ["patch_mixer", "feature_mixer"]:
            state_dict[f"{prefix}.{mixer}.norm.norm.weight"] = sd[f"mixer_block.layers.{i}.{mixer}.norm.norm.weight"]
            state_dict[f"{prefix}.{mixer}.norm.norm.bias"] = sd[f"mixer_block.layers.{i}.{mixer}.norm.norm.bias"]
            state_dict[f"{prefix}.{mixer}.mlp.fc1.weight"] = sd[f"mixer_block.layers.{i}.{mixer}.mlp.fc1.weight"]
            state_dict[f"{prefix}.{mixer}.mlp.fc1.bias"] = sd[f"mixer_block.layers.{i}.{mixer}.mlp.fc1.bias"]
            state_dict[f"{prefix}.{mixer}.mlp.fc2.weight"] = sd[f"mixer_block.layers.{i}.{mixer}.mlp.fc2.weight"]
            state_dict[f"{prefix}.{mixer}.mlp.fc2.bias"] = sd[f"mixer_block.layers.{i}.{mixer}.mlp.fc2.bias"]

            if use_gated_attn:
                state_dict[f"{prefix}.{mixer}.gate.attn_layer.weight"] = sd[
                    f"mixer_block.layers.{i}.{mixer}.gate.attn_layer.weight"
                ]
                state_dict[f"{prefix}.{mixer}.gate.attn_layer.bias"] = sd[
                    f"mixer_block.layers.{i}.{mixer}.gate.attn_layer.bias"
                ]

    # Head
    state_dict[f"{base}.head.proj.weight"] = sd["head.proj.weight"]
    state_dict[f"{base}.head.proj.bias"] = sd["head.proj.bias"]

    # Convert to TTNN
    parameters = {}

    w_tt, b_tt = preprocess_embedding_proj(state_dict, f"{base}.patch_embed", device=device)
    parameters[f"{base}.patch_embed.proj.weight"] = w_tt
    parameters[f"{base}.patch_embed.proj.bias"] = b_tt

    tt_pe = preprocess_positional_encoding(state_dict, f"{base}.pos_enc", device=device)
    parameters[f"{base}.pos_enc.pe"] = tt_pe

    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        for mixer_name in ["patch_mixer", "feature_mixer"]:
            mixer_path = f"{prefix}.{mixer_name}"

            gamma, beta = preprocess_layernorm(state_dict, f"{mixer_path}.norm", device=device)
            parameters[f"{mixer_path}.norm.norm.weight"] = gamma
            parameters[f"{mixer_path}.norm.norm.bias"] = beta

            w1, b1 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc1", device=device)
            w2, b2 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc2", device=device)
            parameters[f"{mixer_path}.mlp.fc1.weight"] = w1
            parameters[f"{mixer_path}.mlp.fc1.bias"] = b1
            parameters[f"{mixer_path}.mlp.fc2.weight"] = w2
            parameters[f"{mixer_path}.mlp.fc2.bias"] = b2

            if use_gated_attn:
                gw, gb = preprocess_gated_attention(state_dict, f"{mixer_path}.gate", device=device)
                parameters[f"{mixer_path}.gate.attn_layer.weight"] = gw
                parameters[f"{mixer_path}.gate.attn_layer.bias"] = gb

    hw_tt, hb_tt = preprocess_forecast_head(state_dict, f"{base}.head", device=device)
    parameters[f"{base}.head.proj.weight"] = hw_tt
    parameters[f"{base}.head.proj.bias"] = hb_tt

    return parameters


def run_benchmark(
    dataset_name="ETTh2",
    checkpoint_path=None,
    device_id=0,
    context_length=512,
    prediction_length=96,
    patch_length=16,
    patch_stride=8,
    d_model=64,
    num_layers=4,
    mode="common_channel",
    expansion=2,
    use_gated_attn=False,
    num_samples=100,
    batch_size=32,
    data_dir="./data",
):
    """Run full benchmark on dataset"""

    print("=" * 80)
    print(f"PatchTSMixer Dataset Benchmark: {dataset_name}")
    print("=" * 80)

    # Prepare data
    print(f"\n1. Loading {dataset_name} dataset...")
    test_data, feature_columns, mean, std = prepare_ett_data(dataset_name, context_length, prediction_length, data_dir)
    num_channels = len(feature_columns)
    print(f"   Channels: {num_channels}")
    print(f"   Test samples: {len(test_data)}")

    # Create samples
    print(f"\n2. Creating {num_samples} forecast samples...")
    samples, targets = create_forecast_samples(
        test_data, feature_columns, context_length, prediction_length, num_samples
    )
    print(f"   Samples shape: {samples.shape}")
    print(f"   Targets shape: {targets.shape}")

    # Model config
    config = {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "patch_length": patch_length,
        "patch_stride": patch_stride,
        "num_channels": num_channels,
        "d_model": d_model,
        "num_layers": num_layers,
        "mode": mode,
        "expansion": expansion,
        "dropout": 0.0,
        "use_gated_attn": use_gated_attn,
        "head_dropout": 0.0,
        "eps": 1e-5,
    }

    # Load PyTorch model
    print(f"\n3. Loading PyTorch model...")
    torch_model = load_or_create_model(config, checkpoint_path)

    # PyTorch inference
    print(f"\n4. Running PyTorch inference...")
    torch_predictions = []
    torch_time = 0

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = torch.from_numpy(samples[i : i + batch_size])

            start = time.time()
            pred = torch_model(batch)
            torch_time += time.time() - start

            torch_predictions.append(pred.numpy())

    torch_predictions = np.concatenate(torch_predictions, axis=0)
    print(f"   Time: {torch_time:.2f}s")
    print(f"   Throughput: {len(samples) / torch_time:.2f} samples/sec")

    # Open TTNN device
    print(f"\n5. Setting up TTNN model on device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Convert parameters
        parameters = preprocess_parameters_for_ttnn(torch_model, device, num_layers, use_gated_attn)

        # Create TTNN model
        tt_model = TtPatchTSMixerModelForForecasting(
            device=device,
            base_address="model",
            parameters=parameters,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            num_channels=num_channels,
            d_model=d_model,
            num_layers=num_layers,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=1e-5,
        )

        # TTNN inference
        print(f"\n6. Running TTNN inference...")
        tt_predictions = []
        ttnn_time = 0

        for i in range(0, len(samples), batch_size):
            batch = torch.from_numpy(samples[i : i + batch_size])

            start = time.time()
            tt_out = tt_model(batch, dtype=ttnn.bfloat16)
            tt_out_torch = ttnn.to_torch(tt_out).squeeze(2).float().numpy()
            ttnn_time += time.time() - start

            tt_predictions.append(tt_out_torch)

        tt_predictions = np.concatenate(tt_predictions, axis=0)
        print(f"   Time: {ttnn_time:.2f}s")
        print(f"   Throughput: {len(samples) / ttnn_time:.2f} samples/sec")

    finally:
        ttnn.close_device(device)

    # Compute metrics
    print(f"\n7. Computing metrics...")

    # PyTorch vs Ground Truth
    torch_metrics = compute_metrics(torch_predictions, targets)
    print(f"\n   PyTorch vs Ground Truth:")
    print(f"      MSE:  {torch_metrics['mse']:.6f}")
    print(f"      MAE:  {torch_metrics['mae']:.6f}")
    print(f"      RMSE: {torch_metrics['rmse']:.6f}")
    print(f"      Correlation: {torch_metrics['correlation']:.6f}")

    # TTNN vs Ground Truth
    ttnn_metrics = compute_metrics(tt_predictions, targets)
    print(f"\n   TTNN vs Ground Truth:")
    print(f"      MSE:  {ttnn_metrics['mse']:.6f}")
    print(f"      MAE:  {ttnn_metrics['mae']:.6f}")
    print(f"      RMSE: {ttnn_metrics['rmse']:.6f}")
    print(f"      Correlation: {ttnn_metrics['correlation']:.6f}")

    # TTNN vs PyTorch (implementation accuracy)
    ttnn_vs_torch = compute_metrics(tt_predictions, torch_predictions)
    print(f"\n   TTNN vs PyTorch (Implementation Accuracy):")
    print(f"      MSE:  {ttnn_vs_torch['mse']:.6f}")
    print(f"      MAE:  {ttnn_vs_torch['mae']:.6f}")
    print(f"      RMSE: {ttnn_vs_torch['rmse']:.6f}")
    print(f"      Correlation: {ttnn_vs_torch['correlation']:.6f}")

    # Validation checks
    print(f"\n8. Validation:")

    mse_diff_pct = abs(ttnn_metrics["mse"] - torch_metrics["mse"]) / torch_metrics["mse"] * 100
    mae_diff_pct = abs(ttnn_metrics["mae"] - torch_metrics["mae"]) / torch_metrics["mae"] * 100

    mse_check = mse_diff_pct <= 5.0
    mae_check = mae_diff_pct <= 5.0
    corr_check = ttnn_vs_torch["correlation"] >= 0.90

    print(f"   MSE difference: {mse_diff_pct:.2f}% {'✅' if mse_check else '❌'} (target: ≤5%)")
    print(f"   MAE difference: {mae_diff_pct:.2f}% {'✅' if mae_check else '❌'} (target: ≤5%)")
    print(
        f"   TTNN-PyTorch correlation: {ttnn_vs_torch['correlation']:.4f} {'✅' if corr_check else '❌'} (target: ≥0.90)"
    )

    all_passed = mse_check and mae_check and corr_check

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ BENCHMARK PASSED - All validation criteria met!")
    else:
        print("❌ BENCHMARK FAILED - Some validation criteria not met")
    print("=" * 80)

    return {
        "dataset": dataset_name,
        "torch_metrics": torch_metrics,
        "ttnn_metrics": ttnn_metrics,
        "ttnn_vs_torch": ttnn_vs_torch,
        "torch_time": torch_time,
        "ttnn_time": ttnn_time,
        "passed": all_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PatchTSMixer on time series datasets")

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh2",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        help="Dataset to benchmark on",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (optional)")

    # Model options
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="common_channel", choices=["common_channel", "mix_channel"])
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--gated-attn", action="store_true")

    # Inference options
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    results = run_benchmark(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        device_id=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_layers=args.num_layers,
        mode=args.mode,
        expansion=args.expansion,
        use_gated_attn=args.gated_attn,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    exit(main())
