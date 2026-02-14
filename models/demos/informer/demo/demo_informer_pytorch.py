# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Informer PyTorch reference script.

Modes:
  train - train a small Torch Informer and export TTNN-compatible checkpoint
  eval  - evaluate Torch Informer against ground truth on dataset
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from models.demos.informer.reference import (
    TorchInformerModel,
    build_calendar_time_features,
    build_sinusoidal_time_features,
    compute_normalization,
    default_etth1_splits,
    denormalize_values,
    iter_windows,
    load_etth1_csv,
    normalize_values,
)
from models.demos.informer.tt import InformerConfig


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", type=int, default=7, help="Number of variables")
    parser.add_argument("--time-feature-dim", type=int, default=4, help="Time feature dimension")
    parser.add_argument("--seq-len", type=int, default=96, help="Encoder input length")
    parser.add_argument("--label-len", type=int, default=48, help="Decoder known length")
    parser.add_argument("--pred-len", type=int, default=24, help="Prediction horizon")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension (multiple of 32)")
    parser.add_argument("--n-heads", type=int, default=2, help="Attention heads")
    parser.add_argument("--d-ff", type=int, default=256, help="FFN dimension")
    parser.add_argument("--e-layers", type=int, default=2, help="Encoder layers")
    parser.add_argument("--d-layers", type=int, default=1, help="Decoder layers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")


def _load_checkpoint(path: str | None) -> tuple[dict | None, dict | None, torch.Tensor | None, torch.Tensor | None]:
    if not path:
        return None, None, None, None
    state = torch.load(path, map_location="cpu")
    cfg = None
    mean = None
    std = None
    if isinstance(state, dict):
        cfg = state.get("config")
        mean = state.get("mean")
        std = state.get("std")
    return state, cfg, mean, std


def _build_config(args: argparse.Namespace, cfg_override: dict | None) -> InformerConfig:
    def pick(name: str, fallback: int) -> int:
        return int(cfg_override.get(name, fallback)) if cfg_override else fallback

    return InformerConfig(
        enc_in=pick("enc_in", args.features),
        dec_in=pick("dec_in", args.features),
        c_out=pick("c_out", args.features),
        seq_len=pick("seq_len", args.seq_len),
        label_len=pick("label_len", args.label_len),
        pred_len=pick("pred_len", args.pred_len),
        time_feature_dim=pick("time_feature_dim", args.time_feature_dim),
        d_model=pick("d_model", args.d_model),
        n_heads=pick("n_heads", args.n_heads),
        d_ff=pick("d_ff", args.d_ff),
        e_layers=pick("e_layers", args.e_layers),
        d_layers=pick("d_layers", args.d_layers),
        dropout=0.0,
    )


def _compute_metrics(pred: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = pred - ref
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    x = pred.flatten()
    y = ref.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum()) * torch.sqrt((vy * vy).sum())
    corr = float((vx * vy).sum().item() / (denom.item() + 1e-8))
    return mse, mae, corr


def _train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    timestamps, dataset = load_etth1_csv(Path(args.dataset), features=args.features)
    if dataset.shape[0] < args.seq_len + args.pred_len:
        raise ValueError("Dataset is shorter than seq_len + pred_len.")

    if args.time_features == "calendar":
        time_features = build_calendar_time_features(timestamps, dataset.shape[0], args.time_feature_dim)
    else:
        time_features = build_sinusoidal_time_features(dataset.shape[0], args.time_feature_dim)

    train_len = min(args.train_len, dataset.shape[0])
    mean, std = compute_normalization(dataset, train_len)
    dataset = normalize_values(dataset, mean, std)

    starts = list(
        iter_windows(
            train_len,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
        )
    )
    if not starts:
        raise ValueError("No training windows generated.")

    cfg = _build_config(args, None)
    model = TorchInformerModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for step in range(1, args.steps + 1):
        batch_starts = random.choices(starts, k=args.batch_size)
        past_values = []
        past_time = []
        future_time = []
        future_values = []
        for start in batch_starts:
            end = start + args.seq_len + args.pred_len
            past_values.append(dataset[start : start + args.seq_len])
            past_time.append(time_features[start : start + args.seq_len])
            future_time.append(time_features[start + args.seq_len : end])
            future_values.append(dataset[start + args.seq_len : end])

        past_values = torch.stack(past_values, dim=0)
        past_time = torch.stack(past_time, dim=0)
        future_time = torch.stack(future_time, dim=0)
        future_values = torch.stack(future_values, dim=0)

        optimizer.zero_grad(set_to_none=True)
        preds = model(past_values, past_time, future_time, future_values)
        loss = torch.mean((preds - future_values) ** 2)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"step {step:>4} loss {loss.item():.6f}")

    checkpoint = {
        "state_dict": model.state_dict_ttnn(),
        "torch_state_dict": model.state_dict(),
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "config": {
            "enc_in": cfg.enc_in,
            "dec_in": cfg.dec_in,
            "c_out": cfg.c_out,
            "seq_len": cfg.seq_len,
            "label_len": cfg.label_len,
            "pred_len": cfg.pred_len,
            "time_feature_dim": cfg.time_feature_dim,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "e_layers": cfg.e_layers,
            "d_layers": cfg.d_layers,
        },
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to {args.output}")


def _eval(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    state, cfg_override, mean_ckpt, std_ckpt = _load_checkpoint(args.checkpoint)
    if state is None:
        raise ValueError("Checkpoint is required for torch eval.")

    cfg = _build_config(args, cfg_override)
    model = TorchInformerModel(cfg)
    torch_state = state.get("torch_state_dict")
    if torch_state is None:
        raise ValueError("Checkpoint missing torch_state_dict. Recreate with --train.")
    model.load_state_dict(torch_state, strict=True)
    model.eval()

    timestamps, dataset = load_etth1_csv(Path(args.dataset), features=cfg.enc_in)
    if dataset.shape[0] < cfg.seq_len + cfg.pred_len:
        raise ValueError("Dataset is shorter than seq_len + pred_len.")

    if args.time_features == "calendar":
        time_features = build_calendar_time_features(timestamps, dataset.shape[0], cfg.time_feature_dim)
    else:
        time_features = build_sinusoidal_time_features(dataset.shape[0], cfg.time_feature_dim)

    mean = mean_ckpt
    std = std_ckpt
    if mean is None and std is None and args.normalize:
        train_len = min(args.train_len, dataset.shape[0])
        mean, std = compute_normalization(dataset, train_len)

    if mean is not None and std is not None:
        dataset = normalize_values(dataset, mean, std)

    starts = list(
        iter_windows(
            dataset.shape[0],
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            stride=args.stride,
            max_windows=args.max_windows,
        )
    )
    if not starts:
        raise ValueError("No evaluation windows generated.")

    mse_sum = 0.0
    mae_sum = 0.0
    corr_sum = 0.0
    batches = 0

    with torch.no_grad():
        for i in range(0, len(starts), args.batch):
            batch_starts = starts[i : i + args.batch]
            past_values = []
            past_time = []
            future_time = []
            future_values = []
            for start in batch_starts:
                end = start + cfg.seq_len + cfg.pred_len
                past_values.append(dataset[start : start + cfg.seq_len])
                past_time.append(time_features[start : start + cfg.seq_len])
                future_time.append(time_features[start + cfg.seq_len : end])
                future_values.append(dataset[start + cfg.seq_len : end])

            past_values = torch.stack(past_values, dim=0)
            past_time = torch.stack(past_time, dim=0)
            future_time = torch.stack(future_time, dim=0)
            future_values = torch.stack(future_values, dim=0)

            preds = model(past_values, past_time, future_time)
            out_gt = preds
            future_gt = future_values
            if mean is not None and std is not None:
                out_gt = denormalize_values(out_gt, mean, std)
                future_gt = denormalize_values(future_gt, mean, std)

            mse, mae, corr = _compute_metrics(out_gt, future_gt)
            mse_sum += mse
            mae_sum += mae
            corr_sum += corr
            batches += 1

    print(f"Batches: {batches}")
    print("Torch vs ground truth:")
    print(f"MSE: {mse_sum / batches:.6f}")
    print(f"MAE: {mae_sum / batches:.6f}")
    print(f"Corr: {corr_sum / batches:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Informer Torch reference demo")
    subparsers = p.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train Torch Informer and export checkpoint")
    _add_model_args(train)
    train.add_argument("--dataset", required=True, help="Path to ETTh1.csv")
    train.add_argument("--output", required=True, help="Checkpoint output path (.pt)")
    train.add_argument("--time-features", choices=["calendar", "sin"], default="calendar", help="Time feature type")
    train.add_argument("--train-len", type=int, default=default_etth1_splits().train_len, help="Train split length")
    train.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train.add_argument("--steps", type=int, default=200, help="Training steps")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train.add_argument("--stride", type=int, default=24, help="Window stride")

    eval_p = subparsers.add_parser("eval", help="Evaluate Torch Informer on dataset")
    _add_model_args(eval_p)
    eval_p.add_argument("--dataset", required=True, help="Path to ETTh1.csv")
    eval_p.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt)")
    eval_p.add_argument("--batch", type=int, default=2, help="Batch size")
    eval_p.add_argument("--stride", type=int, default=24, help="Window stride")
    eval_p.add_argument("--max-windows", type=int, default=32, help="Max windows to evaluate")
    eval_p.add_argument("--time-features", choices=["calendar", "sin"], default="calendar", help="Time feature type")
    eval_p.add_argument("--normalize", action="store_true", help="Normalize using training split stats")
    eval_p.add_argument("--train-len", type=int, default=default_etth1_splits().train_len, help="Train split length")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        _train(args)
    elif args.command == "eval":
        _eval(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
