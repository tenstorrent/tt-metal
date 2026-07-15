#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ttnn

from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_forecast_head,
    preprocess_linear_head,
    preprocess_positional_encoding,
    preprocess_pretrain_head,
)


_ALLOWED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


def download_ett_dataset(dataset_name="ETTh2", data_dir="./data"):
    """Download ETT dataset from GitHub"""
    if dataset_name not in _ALLOWED_DATASETS:
        raise ValueError(f"dataset_name must be one of {_ALLOWED_DATASETS}, got: {dataset_name!r}")

    base_dir = Path(data_dir).resolve()
    base_dir.mkdir(exist_ok=True)

    dataset_path = (base_dir / f"{dataset_name}.csv").resolve()
    if not str(dataset_path).startswith(str(base_dir)):
        raise ValueError(f"Resolved dataset path escapes the base directory: {dataset_path}")

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
    #  timestamp_column = "date"
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


def build_task_targets(
    samples, targets, *, task_mode, prediction_length, patch_length, patch_stride, num_targets, num_classes
):
    """Build task-specific metric targets from forecast windows."""
    if task_mode == "forecasting":
        return targets

    if task_mode == "regression":
        # Use mean over prediction horizon as regression target.
        reg_targets = targets.mean(axis=1)  # (B, C)
        return reg_targets[:, :num_targets].astype(np.float32)

    if task_mode == "classification":
        # Derive classes from the mean future OT channel (last feature).
        signal = targets[:, :, -1].mean(axis=1)
        quantiles = np.linspace(0.0, 1.0, num_classes + 1)
        bins = np.quantile(signal, quantiles)
        # Ensure strictly increasing bin edges for stable digitize.
        bins = np.maximum.accumulate(bins)
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 1e-6
        labels = np.digitize(signal, bins[1:-1], right=False)
        return np.eye(num_classes, dtype=np.float32)[labels]

    if task_mode == "pretraining":
        # Patchify past context to reconstruction target: (B, C, Np, patch_length).
        bsz, context_length, channels = samples.shape
        num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        past_bcl = np.transpose(samples, (0, 2, 1))
        patch_targets = np.empty((bsz, channels, num_patches, patch_length), dtype=np.float32)
        for p in range(num_patches):
            s = p * patch_stride
            patch_targets[:, :, p, :] = past_bcl[:, :, s : s + patch_length]
        return patch_targets

    raise ValueError(f"Unknown task_mode for targets: {task_mode}")


def to_metric_output(pred, task_mode):
    """Convert model output tensor to numpy array for metric computation."""
    pred_np = pred.detach().cpu().float().numpy()

    # Forecast path returns (B, H, C). Keep compatibility with older (B, H, 1, C).
    if task_mode == "forecasting" and pred_np.ndim == 4 and pred_np.shape[2] == 1:
        pred_np = np.squeeze(pred_np, axis=2)

    if task_mode in ["regression", "classification"]:
        # TTNN linear-head outputs may include leading singleton dims (e.g. 1,1,B,N).
        pred_np = np.squeeze(pred_np)
        if pred_np.ndim == 1:
            pred_np = pred_np[None, :]

    if task_mode == "classification":
        # Compare probabilities against one-hot targets.
        pred_np = pred_np - np.max(pred_np, axis=-1, keepdims=True)
        exp = np.exp(pred_np)
        pred_np = exp / np.sum(exp, axis=-1, keepdims=True)

    return pred_np.astype(np.float32)


def load_or_create_model(model_class, config, checkpoint_path=None):
    """Load a trained model or create a new one"""
    model = model_class(**config).eval()

    if checkpoint_path:
        resolved_checkpoint = Path(checkpoint_path).resolve()
        if not resolved_checkpoint.is_file():
            raise ValueError(f"Checkpoint path does not point to a valid file: {resolved_checkpoint}")
        print(f"Loading checkpoint from {resolved_checkpoint}")

        # Prefer a safer load path when supported by the installed PyTorch.
        # Fallback preserves compatibility for older versions, but should only
        # be used with checkpoints from trusted sources.
        try:
            checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            warnings.warn(
                "This PyTorch version does not support weights_only=True. "
                "Falling back to torch.load with full deserialization. "
                "Only load checkpoints from trusted sources.",
                RuntimeWarning,
            )
            checkpoint = torch.load(resolved_checkpoint, map_location="cpu")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️  Using randomly initialized model (no checkpoint provided)")
        print("   For meaningful benchmarks, train a model first or use pretrained weights")

    return model


def preprocess_parameters_for_ttnn(
    torch_model,
    device,
    num_layers,
    use_gated_attn,
    task_mode="forecasting",
    mode="common_channel",
):
    """Convert PyTorch model parameters to TTNN format."""
    sd = torch_model.state_dict()
    parameters = {}

    # Determine model-specific prefixes.
    if task_mode == "forecasting":
        embedding_key = "patch_embed"
        pos_enc_key = "pos_enc"
        encoder_key = "mixer_block"
        head_key = "head"
    else:
        embedding_key = "embedding"
        pos_enc_key = "pos_encoder"
        encoder_key = "encoder"
        head_key = "head"

    def extract_substate(prefix):
        prefix_with_dot = f"{prefix}."
        return {k[len(prefix_with_dot) :]: v for k, v in sd.items() if k.startswith(prefix_with_dot)}

    # Embedding + positional encoding.
    embed_sd = extract_substate(embedding_key)
    parameters.update(preprocess_embedding_proj(embed_sd, f"model.{embedding_key}", device=device))

    pos_enc_sd = extract_substate(pos_enc_key)
    parameters.update(preprocess_positional_encoding(pos_enc_sd, f"model.{pos_enc_key}", device=device))

    # Encoder/mixer block. This handles channel_mixer when mode == "mix_channel".
    encoder_sd = extract_substate(encoder_key)
    parameters.update(
        preprocess_block(
            encoder_sd,
            f"model.{encoder_key}",
            device,
            num_layers=num_layers,
            mode=mode,
            use_gated_attn=use_gated_attn,
        )
    )

    # Task heads.
    head_sd = extract_substate(head_key)
    if task_mode == "forecasting":
        head_params = preprocess_forecast_head(head_sd, f"model.{head_key}", device=device)
    elif task_mode in ["regression", "classification"]:
        head_params = preprocess_linear_head(head_sd, f"model.{head_key}", device=device)
    elif task_mode == "pretraining":
        head_params = preprocess_pretrain_head(head_sd, f"model.{head_key}", device=device)
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

    # Resolve task defaults before model config and target creation.
    if task_mode == "regression" and num_targets is None:
        num_targets = num_channels
    if task_mode == "classification" and num_classes is None:
        raise ValueError("num_classes must be specified for classification task")

    metric_targets = build_task_targets(
        samples,
        targets,
        task_mode=task_mode,
        prediction_length=prediction_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_targets=num_targets,
        num_classes=num_classes,
    )
    print(f"   Metric target shape: {metric_targets.shape}")

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
        config["num_targets"] = num_targets
        config["head_aggregation"] = head_aggregation
        if output_range is not None:
            config["output_range"] = output_range
    elif task_mode == "classification":
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

            torch_predictions.append(to_metric_output(pred, task_mode))

    torch_predictions = np.concatenate(torch_predictions, axis=0)
    torch_throughput = len(samples) / torch_time
    print(f"   Time: {torch_time:.2f}s")
    print(f"   Throughput: {torch_throughput:.2f} samples/sec")

    # Open TTNN device
    print(f"\n5. Setting up TTNN model on device {device_id}...")
    device = ttnn.open_device(device_id=device_id)
    print(f"Device ID: {device.id()}")
    print(f"Available compute cores: {device.compute_with_storage_grid_size()}")

    try:
        # Convert parameters
        parameters = preprocess_parameters_for_ttnn(torch_model, device, num_layers, use_gated_attn, task_mode, mode)

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
            tt_out_torch = ttnn.to_torch(tt_out)
            tt_predictions.append(to_metric_output(tt_out_torch, task_mode))

        tt_predictions = np.concatenate(tt_predictions, axis=0)
        ttnn_throughput = len(samples) / ttnn_time
        per_sample_latency_at_batch_ms = (ttnn_time / len(samples)) * 1000.0
        print(f"   Time: {ttnn_time:.2f}s")
        print(f"   Throughput: {ttnn_throughput:.2f} samples/sec")
        print(f"   Per-sample latency at batch={batch_size}: {per_sample_latency_at_batch_ms:.2f} ms")

        # Measure true single-sequence latency with batch_size=1.
        print("\n6b. Measuring single-sequence latency (batch=1)...")
        single_batch_warmup = torch.from_numpy(samples[0:1])
        single_batch_warmup_tt = ttnn.from_torch(
            single_batch_warmup,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        _ = tt_model(single_batch_warmup_tt, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(device)

        single_seq_time = 0.0
        for i in range(len(samples)):
            single_batch = torch.from_numpy(samples[i : i + 1])
            single_batch_tt = ttnn.from_torch(single_batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            ttnn.synchronize_device(device)
            start = time.time()
            _ = tt_model(single_batch_tt, dtype=ttnn.bfloat16)
            ttnn.synchronize_device(device)
            single_seq_time += time.time() - start

        single_sequence_latency_ms = (single_seq_time / len(samples)) * 1000.0
        print(f"   Single-sequence latency (batch=1): {single_sequence_latency_ms:.2f} ms")

    finally:
        ttnn.close_device(device)

    # Compute metrics
    print(f"\n7. Computing metrics...")

    # PyTorch vs Ground Truth
    torch_metrics = compute_metrics(torch_predictions, metric_targets)
    print(f"\n   PyTorch vs Ground Truth:")
    print(f"      MSE:  {torch_metrics['mse']:.6f}")
    print(f"      MAE:  {torch_metrics['mae']:.6f}")
    print(f"      RMSE: {torch_metrics['rmse']:.6f}")
    print(f"      Correlation: {torch_metrics['correlation']:.6f}")

    # TTNN vs Ground Truth
    ttnn_metrics = compute_metrics(tt_predictions, metric_targets)
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

    checkpoint_check = checkpoint_path is not None
    mse_check = mse_diff_pct <= 5.0
    mae_check = mae_diff_pct <= 5.0
    impl_corr_check = ttnn_vs_torch["correlation"] >= 0.90

    # Bounty gates: trained checkpoint path + ground-truth correlation + device perf targets.
    gt_corr_check = ttnn_metrics["correlation"] >= 0.90
    throughput_check = ttnn_throughput >= 200.0
    latency_check = single_sequence_latency_ms < 30.0

    print(f"   Checkpoint provided: {'✅' if checkpoint_check else '❌'} (required for acceptance)")
    print(f"   MSE difference: {mse_diff_pct:.2f}% {'✅' if mse_check else '❌'} (target: ≤5%)")
    print(f"   MAE difference: {mae_diff_pct:.2f}% {'✅' if mae_check else '❌'} (target: ≤5%)")
    print(
        f"   TTNN-PyTorch correlation: {ttnn_vs_torch['correlation']:.4f} "
        f"{'✅' if impl_corr_check else '❌'} (target: ≥0.90)"
    )
    print(
        f"   TTNN-ground-truth correlation: {ttnn_metrics['correlation']:.4f} "
        f"{'✅' if gt_corr_check else '❌'} (target: >0.90)"
    )
    print(f"   Throughput: {ttnn_throughput:.2f} samples/sec {'✅' if throughput_check else '❌'} (target: ≥200)")
    print(f"   Per-sample latency at batch={batch_size}: {per_sample_latency_at_batch_ms:.2f} ms (informational)")
    print(
        f"   Single-sequence latency (batch=1): {single_sequence_latency_ms:.2f} ms "
        f"{'✅' if latency_check else '❌'} (target: <30)"
    )

    all_passed = (
        checkpoint_check
        and mse_check
        and mae_check
        and impl_corr_check
        and gt_corr_check
        and throughput_check
        and latency_check
    )

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
        "torch_throughput": torch_throughput,
        "ttnn_throughput": ttnn_throughput,
        "ttnn_latency_ms": single_sequence_latency_ms,
        "ttnn_per_sample_latency_ms_at_batch": per_sample_latency_at_batch_ms,
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint from a trusted source (optional)",
    )

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
    parser.add_argument(
        "--mode",
        type=str,
        default="common_channel",
        choices=["common_channel", "mix_channel"],
        help="Channel mode. Hybrid mode is not currently supported in this benchmark flow.",
    )
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
