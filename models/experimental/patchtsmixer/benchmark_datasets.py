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

from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_embedding_proj,
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_linear,
    preprocess_positional_encoding,
)


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


def load_or_create_model(model_class, config, checkpoint_path=None):
    """Load a trained model or create a new one"""
    model = model_class(**config).eval()

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


def preprocess_parameters_for_ttnn(torch_model, device, num_layers, use_gated_attn, task_mode="forecasting"):
    """Convert PyTorch model parameters to TTNN format"""
    base = "model"
    sd = torch_model.state_dict()
    state_dict = {}

    # Determine the correct base prefix based on task mode
    if task_mode == "forecasting":
        embedding_key = "patch_embed"
        pos_enc_key = "pos_enc"
        encoder_key = "mixer_block"
        head_key = "head"
    else:  # regression, classification, pretraining
        embedding_key = "embedding"
        pos_enc_key = "pos_encoder"
        encoder_key = "encoder"
        head_key = "head"

    # Embedding
    state_dict[f"{base}.{embedding_key}.proj.weight"] = sd[f"{embedding_key}.proj.weight"]
    state_dict[f"{base}.{embedding_key}.proj.bias"] = sd[f"{embedding_key}.proj.bias"]

    # Positional encoding
    state_dict[f"{base}.{pos_enc_key}.pe"] = sd[f"{pos_enc_key}.pe"]

    # Encoder/Mixer layers
    for i in range(num_layers):
        prefix = f"{base}.{encoder_key}.layers.{i}"

        for mixer in ["patch_mixer", "feature_mixer"]:
            state_dict[f"{prefix}.{mixer}.norm.norm.weight"] = sd[f"{encoder_key}.layers.{i}.{mixer}.norm.norm.weight"]
            state_dict[f"{prefix}.{mixer}.norm.norm.bias"] = sd[f"{encoder_key}.layers.{i}.{mixer}.norm.norm.bias"]
            state_dict[f"{prefix}.{mixer}.mlp.fc1.weight"] = sd[f"{encoder_key}.layers.{i}.{mixer}.mlp.fc1.weight"]
            state_dict[f"{prefix}.{mixer}.mlp.fc1.bias"] = sd[f"{encoder_key}.layers.{i}.{mixer}.mlp.fc1.bias"]
            state_dict[f"{prefix}.{mixer}.mlp.fc2.weight"] = sd[f"{encoder_key}.layers.{i}.{mixer}.mlp.fc2.weight"]
            state_dict[f"{prefix}.{mixer}.mlp.fc2.bias"] = sd[f"{encoder_key}.layers.{i}.{mixer}.mlp.fc2.bias"]

            if use_gated_attn:
                state_dict[f"{prefix}.{mixer}.gate.attn_layer.weight"] = sd[
                    f"{encoder_key}.layers.{i}.{mixer}.gate.attn_layer.weight"
                ]
                state_dict[f"{prefix}.{mixer}.gate.attn_layer.bias"] = sd[
                    f"{encoder_key}.layers.{i}.{mixer}.gate.attn_layer.bias"
                ]

    # Head
    state_dict[f"{base}.{head_key}.proj.weight"] = sd[f"{head_key}.proj.weight"]
    state_dict[f"{base}.{head_key}.proj.bias"] = sd[f"{head_key}.proj.bias"]

    # Convert to TTNN
    parameters = {}

    # Helper to extract sub-state_dict for a component
    def extract_component_state(prefix):
        """Extract state dict for a specific component, removing the prefix"""
        component_dict = {}
        prefix_with_dot = f"{prefix}."
        for key, value in state_dict.items():
            if key.startswith(prefix_with_dot):
                # Remove the prefix from the key
                new_key = key[len(prefix_with_dot) :]
                component_dict[new_key] = value
        return component_dict

    # Embedding
    embed_state = extract_component_state(f"{base}.{embedding_key}")
    embed_params = preprocess_embedding_proj(embed_state, f"{base}.{embedding_key}", device=device)
    parameters.update(embed_params)

    # Positional encoding
    pos_enc_state = extract_component_state(f"{base}.{pos_enc_key}")
    pos_enc_params = preprocess_positional_encoding(pos_enc_state, f"{base}.{pos_enc_key}", device=device)
    parameters.update(pos_enc_params)

    for i in range(num_layers):
        prefix = f"{base}.{encoder_key}.layers.{i}"

        for mixer_name in ["patch_mixer", "feature_mixer"]:
            mixer_path = f"{prefix}.{mixer_name}"

            # Extract sub-state for this mixer
            mixer_state = extract_component_state(mixer_path)

            # For layernorm, extract the norm component and pass empty base
            norm_state = {}
            for key, value in mixer_state.items():
                if key.startswith("norm."):
                    # Remove "norm." prefix: "norm.norm.weight" -> "norm.weight"
                    norm_state[key[5:]] = value

            norm_params = preprocess_layernorm(norm_state, f"{mixer_path}.norm", device=device)
            parameters.update(norm_params)

            # MLP layers - extract mlp component
            mlp_state = {}
            for key, value in mixer_state.items():
                if key.startswith("mlp."):
                    # Remove "mlp." prefix
                    mlp_state[key[4:]] = value

            # Extract fc1 and fc2 separately
            fc1_state = {}
            fc2_state = {}
            for key, value in mlp_state.items():
                if key.startswith("fc1."):
                    fc1_state[key[4:]] = value  # "fc1.weight" -> "weight"
                elif key.startswith("fc2."):
                    fc2_state[key[4:]] = value  # "fc2.weight" -> "weight"

            fc1_params = preprocess_linear(fc1_state, f"{mixer_path}.mlp.fc1", device=device)
            fc2_params = preprocess_linear(fc2_state, f"{mixer_path}.mlp.fc2", device=device)
            parameters.update(fc1_params)
            parameters.update(fc2_params)

            if use_gated_attn:
                gate_state = {}
                for key, value in mixer_state.items():
                    if key.startswith("gate."):
                        # Remove "gate." prefix
                        gate_state[key[5:]] = value
                gate_params = preprocess_gated_attention(gate_state, f"{mixer_path}.gate", device=device)
                parameters.update(gate_params)

    # Head preprocessing based on task mode
    head_state = extract_component_state(f"{base}.{head_key}")

    if task_mode == "forecasting":
        from models.experimental.patchtsmixer.tt.model_processing import preprocess_forecast_head

        head_params = preprocess_forecast_head(head_state, f"{base}.{head_key}", device=device)
    elif task_mode in ["regression", "classification"]:
        from models.experimental.patchtsmixer.tt.model_processing import preprocess_linear_head

        head_params = preprocess_linear_head(head_state, f"{base}.{head_key}", device=device)
    elif task_mode == "pretraining":
        from models.experimental.patchtsmixer.tt.model_processing import preprocess_pretrain_head

        head_params = preprocess_pretrain_head(head_state, f"{base}.{head_key}", device=device)
    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    parameters.update(head_params)

    return parameters


def run_benchmark(
    dataset_name="ETTh2",
    task_mode="forecasting",  # "forecasting", "regression", "classification", "pretraining"
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
    # Task-specific parameters
    num_targets=None,  # for regression
    num_classes=None,  # for classification
    head_aggregation="avg_pool",  # for regression/classification
    output_range=None,  # for regression
):
    """Run full benchmark on dataset"""

    # Import models based on task mode
    if task_mode == "forecasting":
        from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerModelForForecasting
        from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerModelForForecasting

        PyTorchModel = PatchTSMixerModelForForecasting
        TTNNModel = TtPatchTSMixerModelForForecasting
    elif task_mode == "regression":
        from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerForRegression
        from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForRegression

        PyTorchModel = PatchTSMixerForRegression
        TTNNModel = TtPatchTSMixerForRegression
    elif task_mode == "classification":
        from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import (
            PatchTSMixerForTimeSeriesClassification,
        )
        from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForTimeSeriesClassification

        PyTorchModel = PatchTSMixerForTimeSeriesClassification
        TTNNModel = TtPatchTSMixerForTimeSeriesClassification
    elif task_mode == "pretraining":
        from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerForPretraining
        from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForPretraining

        PyTorchModel = PatchTSMixerForPretraining
        TTNNModel = TtPatchTSMixerForPretraining
    else:
        raise ValueError(
            f"Unknown task_mode: {task_mode}. Choose from: forecasting, regression, classification, pretraining"
        )

    print("=" * 80)
    print(f"PatchTSMixer Dataset Benchmark: {dataset_name} ({task_mode})")
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

    # Add task-specific config parameters
    if task_mode == "forecasting":
        config["prediction_length"] = prediction_length
    elif task_mode == "regression":
        if num_targets is None:
            num_targets = num_channels  # default: predict all channels
        config["num_targets"] = num_targets
        config["head_aggregation"] = head_aggregation
        if output_range is not None:
            config["output_range"] = output_range
    elif task_mode == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        config["num_classes"] = num_classes
        config["head_aggregation"] = head_aggregation
    # pretraining has no extra params

    # Load PyTorch model
    print(f"\n3. Loading PyTorch {task_mode} model...")
    torch_model = load_or_create_model(PyTorchModel, config, checkpoint_path)

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
    print(f"Device ID: {device.id()}")
    print(f"Available compute cores: {device.compute_with_storage_grid_size()}")

    try:
        # Convert parameters
        parameters = preprocess_parameters_for_ttnn(torch_model, device, num_layers, use_gated_attn, task_mode)

        # Create TTNN model
        ttnn_model_config = {
            "device": device,
            "base_address": "model",
            "parameters": parameters,
            "context_length": context_length,
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "num_channels": num_channels,
            "d_model": d_model,
            "num_layers": num_layers,
            "mode": mode,
            "expansion": expansion,
            "use_gated_attn": use_gated_attn,
        }

        # Add task-specific parameters
        if task_mode == "forecasting":
            ttnn_model_config["prediction_length"] = prediction_length
        elif task_mode == "regression":
            ttnn_model_config["num_targets"] = num_targets
            ttnn_model_config["head_aggregation"] = head_aggregation
            if output_range is not None:
                ttnn_model_config["output_range"] = output_range
        elif task_mode == "classification":
            ttnn_model_config["num_classes"] = num_classes
            ttnn_model_config["head_aggregation"] = head_aggregation

        tt_model = TTNNModel(**ttnn_model_config)

        # TTNN inference
        print(f"\n6. Running TTNN inference...")
        tt_predictions = []
        ttnn_time = 0

        # Warm-up run to compile kernels
        batch = torch.from_numpy(samples[0:batch_size])
        batch_tt = ttnn.from_torch(batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        _ = tt_model(batch_tt, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(device)

        for i in range(0, len(samples), batch_size):
            batch = torch.from_numpy(samples[i : i + batch_size])

            # Convert torch tensor to TTNN tensor (outside timing)
            batch_tt = ttnn.from_torch(batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            # Time only device execution
            ttnn.synchronize_device(device)
            start = time.time()
            tt_out = tt_model(batch_tt, dtype=ttnn.bfloat16)
            ttnn.synchronize_device(device)
            ttnn_time += time.time() - start

            # Convert back to torch (outside timing)
            tt_out_torch = ttnn.to_torch(tt_out).squeeze(2).float().numpy()

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

    # Task mode
    parser.add_argument(
        "--task-mode",
        type=str,
        default="forecasting",
        choices=["forecasting", "regression", "classification", "pretraining"],
        help="Task mode: forecasting, regression, classification, or pretraining",
    )

    # Model options
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96, help="For forecasting task")
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="common_channel", choices=["common_channel", "mix_channel"])
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--gated-attn", action="store_true")

    # Task-specific options
    parser.add_argument("--num-targets", type=int, default=None, help="For regression task (default: num_channels)")
    parser.add_argument("--num-classes", type=int, default=None, help="For classification task (required)")
    parser.add_argument(
        "--head-aggregation",
        type=str,
        default="avg_pool",
        choices=["avg_pool", "max_pool", "use_last"],
        help="For regression/classification",
    )
    parser.add_argument("--output-range-min", type=float, default=None, help="For regression task output range")
    parser.add_argument("--output-range-max", type=float, default=None, help="For regression task output range")

    # Inference options
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    # Build output_range tuple if specified
    output_range = None
    if args.output_range_min is not None and args.output_range_max is not None:
        output_range = (args.output_range_min, args.output_range_max)

    results = run_benchmark(
        dataset_name=args.dataset,
        task_mode=args.task_mode,
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
        num_targets=args.num_targets,
        num_classes=args.num_classes,
        head_aggregation=args.head_aggregation,
        output_range=output_range,
    )

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    exit(main())
