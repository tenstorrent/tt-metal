#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Informer real-data assets for full validation runs.

This script downloads ETTh1 CSV from Hugging Face datasets and optionally
materializes a dual-state checkpoint expected by test_real_dataset_accuracy:

  - state_dict (TTNN key mapping)
  - torch_state_dict (native Torch model keys)
  - config
  - mean
  - std

By default this script trains a compact ETTh1 checkpoint locally. Synthetic
checkpoint generation is debug-only and not valid for accuracy claims.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.informer.reference.eval_utils import default_etth1_splits, iter_windows, resolve_eval_range
from models.demos.informer.reference.torch_reference import (
    TorchInformerModel,
    build_calendar_time_features,
    compute_metrics,
    compute_normalization,
    load_etth1_csv,
    normalize_values,
    ttnn_state_dict,
)
from models.demos.informer.tt.config import InformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Informer ETTh1 + HF assets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/demos/informer/.assets"),
        help="Directory for generated assets",
    )
    parser.add_argument(
        "--etth1-repo-id",
        type=str,
        default="pkr7098/time-series-forecasting-datasets",
        help="HF dataset repo containing ETTh1.csv",
    )
    parser.add_argument(
        "--etth1-filename",
        type=str,
        default="ETTh1.csv",
        help="Filename in the HF dataset repo",
    )
    parser.add_argument(
        "--etth1-output",
        type=Path,
        default=None,
        help="Output path for ETTh1.csv (default: <output-dir>/ETTh1.csv)",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="Output path for ETTh1 dual-state checkpoint (default: <output-dir>/etth1_ttnn.pt)",
    )
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        choices=["none", "synthetic", "train"],
        default="train",
        help="Checkpoint source: train (default), none, or synthetic (debug only).",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        type=str,
        default=None,
        help="Optional HF repo id containing a trained dual-state ETTh1 checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default=None,
        help="Optional filename in --checkpoint-repo-id for the trained checkpoint artifact.",
    )
    parser.add_argument(
        "--checkpoint-repo-type",
        type=str,
        choices=["model", "dataset", "space"],
        default="model",
        help="HF repo type for checkpoint download.",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="huggingface/informer-tourism-monthly",
        help="HF model ID to warm into local cache",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="monash_tsf",
        help="HF dataset ID to warm into local cache",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        default="tourism_monthly",
        help="HF dataset config to warm into local cache",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="test",
        help="HF dataset split to warm into local cache",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic checkpoint materialization",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=80,
        help="Number of local training steps for --checkpoint-source train.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Training batch size for --checkpoint-source train.",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=3e-3,
        help="Training learning rate for --checkpoint-source train.",
    )
    parser.add_argument(
        "--train-min-improvement",
        type=float,
        default=0.05,
        help="Minimum required MSE/MAE improvement vs persistence baseline for trained checkpoint.",
    )
    parser.add_argument(
        "--train-min-corr",
        type=float,
        default=0.20,
        help="Minimum required correlation vs ground truth for trained checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate outputs even if files already exist",
    )
    parser.add_argument(
        "--skip-hf-warmup",
        action="store_true",
        help="Skip warming HF model and dataset cache",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_etth1_csv(repo_id: str, filename: str, output_path: Path, force: bool) -> Path:
    from huggingface_hub import hf_hub_download

    ensure_parent(output_path)
    if output_path.exists() and not force:
        print(f"[prepare_assets] Reusing existing ETTh1 CSV: {output_path}")
        return output_path.resolve()

    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    shutil.copy2(cached_path, output_path)
    print(f"[prepare_assets] Downloaded ETTh1 CSV: {output_path}")
    return output_path.resolve()


def try_download_checkpoint_from_hf(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    output_path: Path,
    force: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    ensure_parent(output_path)
    if output_path.exists() and not force:
        print(f"[prepare_assets] Reusing existing checkpoint: {output_path}")
        return output_path.resolve()

    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    shutil.copy2(cached_path, output_path)
    print(f"[prepare_assets] Downloaded trained checkpoint: {output_path}")
    return output_path.resolve()


def build_default_etth1_config() -> InformerConfig:
    return InformerConfig(
        enc_in=7,
        dec_in=7,
        c_out=7,
        seq_len=96,
        label_len=48,
        pred_len=24,
        d_model=32,
        n_heads=1,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        time_feature_dim=4,
        dtype="bfloat16",
        attention_type="prob",
    )


def create_dual_state_checkpoint(csv_path: Path, checkpoint_path: Path, seed: int, force: bool) -> Path:
    ensure_parent(checkpoint_path)
    if checkpoint_path.exists() and not force:
        print(f"[prepare_assets] Reusing existing checkpoint: {checkpoint_path}")
        return checkpoint_path.resolve()

    torch.manual_seed(seed)
    cfg = build_default_etth1_config()
    torch_model = TorchInformerModel(cfg)
    torch_model.eval()

    _, values = load_etth1_csv(csv_path, features=cfg.enc_in)
    split_cfg = default_etth1_splits()
    train_length = min(split_cfg.train_len, values.shape[0])
    mean, std = compute_normalization(values, train_length)

    payload = {
        "state_dict": ttnn_state_dict(torch_model),
        "torch_state_dict": torch_model.state_dict(),
        "config": asdict(cfg),
        "mean": mean,
        "std": std,
        "checkpoint_info": {
            "source": "synthetic",
            "trained": False,
            "note": "Randomly initialized checkpoint for parity debugging only.",
        },
    }
    torch.save(payload, checkpoint_path)
    print(f"[prepare_assets] Created synthetic dual-state checkpoint: {checkpoint_path}")
    return checkpoint_path.resolve()


def checkpoint_has_trained_metadata(checkpoint_path: Path) -> bool:
    if not checkpoint_path.is_file():
        return False
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        return False
    info = payload.get("checkpoint_info", {})
    return bool(isinstance(info, dict) and info.get("trained") is True)


def build_batch_tensors(
    values: torch.Tensor, time_features: torch.Tensor, starts: torch.Tensor, cfg: InformerConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_past_values = []
    batch_past_time = []
    batch_future_time = []
    batch_future_values = []
    for start in starts.tolist():
        window = values[start : start + cfg.seq_len + cfg.pred_len]
        window_time = time_features[start : start + cfg.seq_len + cfg.pred_len]
        batch_past_values.append(window[: cfg.seq_len])
        batch_future_values.append(window[cfg.seq_len :])
        batch_past_time.append(window_time[: cfg.seq_len])
        batch_future_time.append(window_time[cfg.seq_len :])
    return (
        torch.stack(batch_past_values, dim=0),
        torch.stack(batch_past_time, dim=0),
        torch.stack(batch_future_time, dim=0),
        torch.stack(batch_future_values, dim=0),
    )


def evaluate_checkpoint_quality(
    model: TorchInformerModel,
    values: torch.Tensor,
    time_features: torch.Tensor,
    cfg: InformerConfig,
    *,
    min_improvement: float,
    min_corr: float,
) -> dict[str, float]:
    split_cfg = default_etth1_splits()
    eval_start, eval_end = resolve_eval_range(values.shape[0], split="test", split_cfg=split_cfg)
    eval_length = eval_end - eval_start
    mse_values = []
    mae_values = []
    corr_values = []
    baseline_mse_values = []
    baseline_mae_values = []
    baseline_corr_values = []

    model.eval()
    with torch.no_grad():
        for offset in iter_windows(
            eval_length,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            stride=24,
            max_windows=8,
        ):
            start = eval_start + offset
            past_values = values[start : start + cfg.seq_len].unsqueeze(0)
            future_values = values[start + cfg.seq_len : start + cfg.seq_len + cfg.pred_len].unsqueeze(0)
            past_time = time_features[start : start + cfg.seq_len].unsqueeze(0)
            future_time = time_features[start + cfg.seq_len : start + cfg.seq_len + cfg.pred_len].unsqueeze(0)

            pred = model(past_values, past_time, future_time, future_values)
            mse, mae, corr = compute_metrics(pred, future_values)
            baseline = past_values[:, -1:, :].repeat(1, cfg.pred_len, 1)
            mse_b, mae_b, corr_b = compute_metrics(baseline, future_values)
            mse_values.append(mse)
            mae_values.append(mae)
            corr_values.append(corr)
            baseline_mse_values.append(mse_b)
            baseline_mae_values.append(mae_b)
            baseline_corr_values.append(corr_b)

    avg_mse = sum(mse_values) / len(mse_values)
    avg_mae = sum(mae_values) / len(mae_values)
    avg_corr = sum(corr_values) / len(corr_values)
    avg_mse_baseline = sum(baseline_mse_values) / len(baseline_mse_values)
    avg_mae_baseline = sum(baseline_mae_values) / len(baseline_mae_values)
    avg_corr_baseline = sum(baseline_corr_values) / len(baseline_corr_values)

    mse_target = (1.0 - min_improvement) * avg_mse_baseline
    mae_target = (1.0 - min_improvement) * avg_mae_baseline
    if avg_mse > mse_target or avg_mae > mae_target or avg_corr < min_corr:
        raise RuntimeError(
            "Trained checkpoint did not meet quality gate: "
            f"MSE {avg_mse:.6f} (target <= {mse_target:.6f}), "
            f"MAE {avg_mae:.6f} (target <= {mae_target:.6f}), "
            f"Corr {avg_corr:.4f} (target >= {min_corr:.4f})."
        )

    return {
        "mse": avg_mse,
        "mae": avg_mae,
        "corr": avg_corr,
        "baseline_mse": avg_mse_baseline,
        "baseline_mae": avg_mae_baseline,
        "baseline_corr": avg_corr_baseline,
    }


def create_trained_checkpoint(
    csv_path: Path,
    checkpoint_path: Path,
    *,
    seed: int,
    force: bool,
    train_steps: int,
    train_batch_size: int,
    train_lr: float,
    train_min_improvement: float,
    train_min_corr: float,
) -> Path:
    ensure_parent(checkpoint_path)
    if checkpoint_path.exists() and not force and checkpoint_has_trained_metadata(checkpoint_path):
        print(f"[prepare_assets] Reusing trained checkpoint: {checkpoint_path}")
        return checkpoint_path.resolve()

    torch.manual_seed(seed)
    cfg = build_default_etth1_config()
    model = TorchInformerModel(cfg)
    model.train()

    timestamps, values = load_etth1_csv(csv_path, features=cfg.enc_in)
    if timestamps is None:
        raise RuntimeError("ETTh1 CSV must include a parseable timestamp column for calendar features.")
    time_features = build_calendar_time_features(timestamps, values.shape[0], cfg.time_feature_dim)

    split_cfg = default_etth1_splits()
    train_length = min(split_cfg.train_len, values.shape[0])
    mean, std = compute_normalization(values, train_length)
    values = normalize_values(values, mean, std)

    max_start = train_length - (cfg.seq_len + cfg.pred_len)
    if max_start < 0:
        raise RuntimeError("Dataset too short for configured training window.")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    loss_fn = torch.nn.MSELoss()
    log_stride = max(1, train_steps // 4)

    print(
        "[prepare_assets] Training local ETTh1 checkpoint "
        f"(steps={train_steps}, batch={train_batch_size}, lr={train_lr})..."
    )
    for step in range(1, train_steps + 1):
        starts = torch.randint(0, max_start + 1, (train_batch_size,))
        past_values, past_time, future_time, future_values = build_batch_tensors(values, time_features, starts, cfg)
        pred = model(past_values, past_time, future_time, future_values)
        loss = loss_fn(pred, future_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_stride == 0 or step == train_steps:
            print(f"[prepare_assets] step {step}/{train_steps} loss={float(loss):.6f}")

    quality = evaluate_checkpoint_quality(
        model,
        values,
        time_features,
        cfg,
        min_improvement=train_min_improvement,
        min_corr=train_min_corr,
    )
    print(
        "[prepare_assets] Trained checkpoint quality: "
        f"MSE={quality['mse']:.6f}, MAE={quality['mae']:.6f}, Corr={quality['corr']:.4f}, "
        f"baseline_MSE={quality['baseline_mse']:.6f}"
    )

    payload = {
        "state_dict": ttnn_state_dict(model),
        "torch_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "mean": mean,
        "std": std,
        "checkpoint_info": {
            "source": "trained_local",
            "trained": True,
            "seed": seed,
            "train_steps": train_steps,
            "train_batch_size": train_batch_size,
            "train_lr": train_lr,
            "quality": quality,
        },
    }
    torch.save(payload, checkpoint_path)
    print(f"[prepare_assets] Created trained dual-state checkpoint: {checkpoint_path}")
    return checkpoint_path.resolve()


def warm_hf_assets(model_id: str, dataset_id: str, dataset_config: str, split: str) -> None:
    from datasets import load_dataset
    from transformers import InformerConfig as HfInformerConfig
    from transformers import InformerForPrediction

    HfInformerConfig.from_pretrained(model_id)
    InformerForPrediction.from_pretrained(model_id)
    dataset = load_dataset(dataset_id, dataset_config, split=split)
    _ = len(dataset)
    print("[prepare_assets] Warmed HF assets: " f"model={model_id}, dataset={dataset_id}/{dataset_config}:{split}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_out = args.etth1_output.resolve() if args.etth1_output is not None else output_dir / "ETTh1.csv"
    ckpt_out = args.checkpoint_output.resolve() if args.checkpoint_output is not None else output_dir / "etth1_ttnn.pt"

    csv_path = download_etth1_csv(args.etth1_repo_id, args.etth1_filename, csv_out, args.force)
    checkpoint_path = None
    if args.checkpoint_repo_id and args.checkpoint_filename:
        checkpoint_path = try_download_checkpoint_from_hf(
            repo_id=args.checkpoint_repo_id,
            filename=args.checkpoint_filename,
            repo_type=args.checkpoint_repo_type,
            output_path=ckpt_out,
            force=args.force,
        )
    elif args.checkpoint_source == "train":
        checkpoint_path = create_trained_checkpoint(
            csv_path,
            ckpt_out,
            seed=args.seed,
            force=args.force,
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_min_improvement=args.train_min_improvement,
            train_min_corr=args.train_min_corr,
        )
    elif args.checkpoint_source == "synthetic":
        checkpoint_path = create_dual_state_checkpoint(csv_path, ckpt_out, args.seed, args.force)
    elif ckpt_out.exists():
        checkpoint_path = ckpt_out.resolve()
        print(f"[prepare_assets] Reusing existing checkpoint: {checkpoint_path}")
    else:
        print(
            "[prepare_assets] No checkpoint produced. Provide a trained checkpoint via "
            "--checkpoint-repo-id/--checkpoint-filename or set --checkpoint-source synthetic for debug only."
        )

    if not args.skip_hf_warmup:
        warm_hf_assets(args.hf_model_id, args.hf_dataset, args.hf_dataset_config, args.hf_split)

    print("[prepare_assets] Assets prepared:")
    print(f"  hf_model_id: {args.hf_model_id}")
    print(f"  etth1_csv: {csv_path}")
    if checkpoint_path is not None:
        print(f"  etth1_checkpoint: {checkpoint_path}")
    else:
        print("  etth1_checkpoint: /path/to/trained/etth1_ttnn.pt")


if __name__ == "__main__":
    main()
