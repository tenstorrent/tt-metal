"""
Compare PyTorch PatchTSMixer implementation with HuggingFace's official implementation.

This script:
1. Loads both implementations with the same configuration
2. Runs inference on the same test data
3. Compares predictions, layer outputs, and model statistics
4. Reports differences and validates correctness
"""

import argparse
import os

import pandas as pd
import torch
from pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from torch.utils.data import DataLoader
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction, set_seed
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


def compute_metrics(pred, target):
    """Compute MSE, MAE, and RMSE"""
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {"mse": mse, "mae": mae, "rmse": rmse}


def compare_tensors(tensor1, tensor2, name="tensor", rtol=1e-3, atol=1e-5):
    """Compare two tensors and report differences"""
    if tensor1.shape != tensor2.shape:
        print(f"  ❌ {name}: Shape mismatch! {tensor1.shape} vs {tensor2.shape}")
        return False

    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
    rel_diff = mean_diff / (torch.mean(torch.abs(tensor1)).item() + 1e-8)

    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    print(f"  {name}:")
    print(f"    Shape: {tensor1.shape}")
    print(f"    Max diff: {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    Relative diff: {rel_diff:.6e}")
    print(f"    Close (rtol={rtol}, atol={atol}): {'✓' if is_close else '✗'}")

    return is_close


def build_test_dataset(context_length, prediction_length, num_samples=100):
    """Build a small ETTh2 test dataset"""
    dataset = "ETTh2"
    dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
    timestamp_column = "date"
    id_columns = []
    forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    data = pd.read_csv(dataset_path, parse_dates=[timestamp_column])

    # Use test split
    test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    # Build preprocessor on test data (for simplicity)
    time_series_processor = TimeSeriesPreprocessor(
        context_length=context_length,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        scaling=True,
    )
    time_series_processor.train(test_data)

    test_dataset = ForecastDFDataset(
        time_series_processor.preprocess(test_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    return test_dataset, len(forecast_columns)


def load_pytorch_model(checkpoint_path, config, device):
    """Load custom PyTorch implementation"""
    model = PatchTSMixerModelForForecasting(
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        patch_length=config["patch_length"],
        patch_stride=config["patch_stride"],
        num_channels=config["num_channels"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        mode=config["mode"],
        expansion=2,
        dropout=0.0,  # disable dropout for comparison
        use_gated_attn=config.get("use_gated_attn", False),
        head_dropout=0.0,  # disable dropout for comparison
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded PyTorch model from {checkpoint_path}")
    else:
        print("⚠ Using randomly initialized PyTorch model")

    model.eval()
    return model


def load_huggingface_model(model_path, config, device):
    """Load HuggingFace implementation"""
    # Try loading from local path or hub
    if os.path.exists(model_path):
        model = PatchTSMixerForPrediction.from_pretrained(model_path).to(device)
        print(f"✓ Loaded HuggingFace model from {model_path}")
    else:
        # Create model with matching config
        hf_config = PatchTSMixerConfig(
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            patch_length=config["patch_length"],
            patch_stride=config["patch_stride"],
            num_input_channels=config["num_channels"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            expansion_factor=2,
            dropout=0.0,
            head_dropout=0.0,
            mode=config["mode"],
            gated_attn=config.get("use_gated_attn", False),
        )
        model = PatchTSMixerForPrediction(hf_config).to(device)
        print(f"⚠ Using randomly initialized HuggingFace model")
        print(f"  (Tried to load from: {model_path})")

    model.eval()
    return model


def compare_model_stats(pytorch_model, hf_model):
    """Compare model statistics"""
    print("\n" + "=" * 60)
    print("MODEL STATISTICS COMPARISON")
    print("=" * 60)

    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    hf_params = sum(p.numel() for p in hf_model.parameters())

    pytorch_trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    hf_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)

    print(f"PyTorch Implementation:")
    print(f"  Total parameters: {pytorch_params:,}")
    print(f"  Trainable parameters: {pytorch_trainable:,}")

    print(f"\nHuggingFace Implementation:")
    print(f"  Total parameters: {hf_params:,}")
    print(f"  Trainable parameters: {hf_trainable:,}")

    print(f"\nDifference:")
    print(f"  Parameters: {abs(pytorch_params - hf_params):,}")
    print(f"  Match: {'✓' if pytorch_params == hf_params else '✗'}")


def compare_predictions(pytorch_model, hf_model, test_loader, device, num_batches=5):
    """Compare predictions on test data"""
    print("\n" + "=" * 60)
    print("PREDICTION COMPARISON")
    print("=" * 60)

    all_pytorch_preds = []
    all_hf_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break

            past_values = batch["past_values"].to(device)
            future_values = batch["future_values"].to(device)

            # PyTorch prediction: expects (B, L, C) -> returns (B, H, C)
            pytorch_pred = pytorch_model(past_values)

            # HuggingFace prediction: expects dict -> returns object with .prediction_outputs
            hf_output = hf_model(past_values=past_values)
            hf_pred = hf_output.prediction_outputs

            all_pytorch_preds.append(pytorch_pred.cpu())
            all_hf_preds.append(hf_pred.cpu())
            all_targets.append(future_values.cpu())

            if i == 0:
                print(f"\nBatch {i+1}:")
                compare_tensors(pytorch_pred.cpu(), hf_pred.cpu(), name="Predictions", rtol=1e-3, atol=1e-5)

    # Aggregate all predictions
    pytorch_preds = torch.cat(all_pytorch_preds, dim=0)
    hf_preds = torch.cat(all_hf_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"\n" + "─" * 60)
    print("OVERALL METRICS")
    print("─" * 60)

    pytorch_metrics = compute_metrics(pytorch_preds, targets)
    hf_metrics = compute_metrics(hf_preds, targets)

    print(f"\nPyTorch Implementation:")
    print(f"  MSE: {pytorch_metrics['mse']:.6f}")
    print(f"  MAE: {pytorch_metrics['mae']:.6f}")
    print(f"  RMSE: {pytorch_metrics['rmse']:.6f}")

    print(f"\nHuggingFace Implementation:")
    print(f"  MSE: {hf_metrics['mse']:.6f}")
    print(f"  MAE: {hf_metrics['mae']:.6f}")
    print(f"  RMSE: {hf_metrics['rmse']:.6f}")

    print(f"\nPrediction Agreement:")
    compare_tensors(pytorch_preds, hf_preds, name="All predictions", rtol=1e-3, atol=1e-5)

    return pytorch_metrics, hf_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and HuggingFace PatchTSMixer implementations")
    parser.add_argument(
        "--pytorch_checkpoint",
        type=str,
        default="simple_patchtsmixer_etth2/best_model.pt",
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--hf_model", type=str, default="ibm/patchtsmixer-etth2-forecast", help="HuggingFace model path or hub ID"
    )
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--patch_length", type=int, default=8)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="common_channel")
    parser.add_argument("--use_gated_attn", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=5, help="Number of batches to compare")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PATCHTSMIXER IMPLEMENTATION COMPARISON")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num batches: {args.num_batches}")

    # Build test dataset
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    test_dataset, num_channels = build_test_dataset(args.context_length, args.prediction_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print(f"✓ Num channels: {num_channels}")

    # Model config
    config = {
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "num_channels": num_channels,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "mode": args.mode,
        "use_gated_attn": args.use_gated_attn,
    }

    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    pytorch_model = load_pytorch_model(args.pytorch_checkpoint, config, device)
    hf_model = load_huggingface_model(args.hf_model, config, device)

    # Compare model statistics
    compare_model_stats(pytorch_model, hf_model)

    # Compare predictions
    pytorch_metrics, hf_metrics = compare_predictions(pytorch_model, hf_model, test_loader, device, args.num_batches)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

    # Final summary
    print("\n📊 SUMMARY:")
    if abs(pytorch_metrics["mse"] - hf_metrics["mse"]) < 0.01:
        print("✓ Both implementations produce similar results")
    else:
        print("⚠ Implementations produce different results")
        print("  This is expected if models have different weights or architecture differences")


if __name__ == "__main__":
    main()
