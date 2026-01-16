# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Informer TTNN demo script.

Modes:
  smoke     - quick forward-pass sanity check
  benchmark - latency/throughput sweep
  eval      - dataset evaluation against torch reference + ground truth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import ttnn
from models.demos.informer.reference import (
    build_calendar_time_features,
    build_sinusoidal_time_features,
    compute_normalization,
    default_etth1_splits,
    denormalize_values,
    informer_torch_forward,
    iter_windows,
    load_etth1_csv,
    normalize_values,
)
from models.demos.informer.tt import InformerConfig, InformerModel, to_torch


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device-id", type=int, default=0, help="TT device id")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="TTNN dtype")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--features", type=int, default=7, help="Number of variables")
    parser.add_argument("--time-feature-dim", type=int, default=4, help="Time feature dimension")
    parser.add_argument("--seq-len", type=int, default=96, help="Encoder input length")
    parser.add_argument("--label-len", type=int, default=48, help="Decoder known length")
    parser.add_argument("--pred-len", type=int, default=24, help="Prediction horizon")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--e-layers", type=int, default=2, help="Encoder layers")
    parser.add_argument("--d-layers", type=int, default=1, help="Decoder layers")
    parser.add_argument("--checkpoint", type=str, default=None, help="TTNN-style checkpoint path")


def _parse_batch_sizes(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


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
        if "state_dict" in state:
            state = state["state_dict"]
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
        device_id=args.device_id,
        dtype=args.dtype,
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


def _smoke(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    state, cfg_override, _, _ = _load_checkpoint(args.checkpoint)
    cfg = _build_config(args, cfg_override)

    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=8192)
    try:
        model = InformerModel(cfg, device=device, seed=args.seed)
        if state is not None:
            model.load_state_dict(state, strict=True)
        past_values = torch.randn(args.batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
        past_time = torch.randn(args.batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(args.batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)
        print(f"Output shape: {tuple(out_torch.shape)}")
        print(f"Output sample (mean/std): {out_torch.mean():.4f} / {out_torch.std():.4f}")
    finally:
        ttnn.CloseDevice(device)


def _benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    state, cfg_override, _, _ = _load_checkpoint(args.checkpoint)
    cfg = _build_config(args, cfg_override)

    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=8192)
    try:
        model = InformerModel(cfg, device=device, seed=args.seed)
        if state is not None:
            model.load_state_dict(state, strict=True)

        print("=" * 60)
        print("Informer TTNN Performance Benchmark")
        print("=" * 60)
        print(f"Config: seq_len={cfg.seq_len}, label_len={cfg.label_len}, pred_len={cfg.pred_len}")
        print(f"        features={cfg.enc_in}, d_model={cfg.d_model}, n_heads={cfg.n_heads}")
        print(f"        e_layers={cfg.e_layers}, d_layers={cfg.d_layers}")
        print("-" * 60)
        print(f"{'Batch':>6} | {'Latency (ms)':>12} | {'Throughput (seq/s)':>18}")
        print("-" * 60)

        for batch in args.batch_sizes:
            past_values = torch.randn(batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
            past_time = torch.randn(batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
            future_time = torch.randn(batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)

            for _ in range(args.warmup):
                _ = model(past_values, past_time, future_time)
            ttnn.synchronize_device(device)

            start = time.perf_counter()
            for _ in range(args.iters):
                _ = model(past_values, past_time, future_time)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start

            latency_ms = (elapsed / args.iters) * 1000
            throughput = batch / (elapsed / args.iters)
            print(f"{batch:>6} | {latency_ms:>12.3f} | {throughput:>18.2f}")
        print("-" * 60)
    finally:
        ttnn.CloseDevice(device)


def _eval(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    state, cfg_override, mean_ckpt, std_ckpt = _load_checkpoint(args.checkpoint)
    cfg = _build_config(args, cfg_override)

    dataset_path = Path(args.dataset) if args.dataset else None
    if dataset_path is None:
        default_path = Path(__file__).with_name("ETTh1.csv")
        if default_path.exists():
            dataset_path = default_path
        else:
            raise ValueError("Dataset path is required. Provide --dataset.")

    timestamps, dataset = load_etth1_csv(dataset_path, features=cfg.enc_in)
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

    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=8192)
    try:
        model = InformerModel(cfg, device=device, seed=args.seed)
        if state is not None:
            model.load_state_dict(state, strict=True)

        mse_ref_sum = 0.0
        mae_ref_sum = 0.0
        corr_ref_sum = 0.0
        mse_gt_sum = 0.0
        mae_gt_sum = 0.0
        corr_gt_sum = 0.0
        batches = 0

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

            out_tt = model(past_values, past_time, future_time)
            out = to_torch(out_tt).float()
            ref = informer_torch_forward(model, past_values, past_time, future_time)

            out_gt = out
            future_gt = future_values
            if mean is not None and std is not None:
                out_gt = denormalize_values(out_gt, mean, std)
                future_gt = denormalize_values(future_gt, mean, std)

            mse_ref, mae_ref, corr_ref = _compute_metrics(out, ref)
            mse_gt, mae_gt, corr_gt = _compute_metrics(out_gt, future_gt)
            mse_ref_sum += mse_ref
            mae_ref_sum += mae_ref
            corr_ref_sum += corr_ref
            mse_gt_sum += mse_gt
            mae_gt_sum += mae_gt
            corr_gt_sum += corr_gt
            batches += 1

        avg_ref_mse = mse_ref_sum / batches
        avg_ref_mae = mae_ref_sum / batches
        avg_ref_corr = corr_ref_sum / batches
        avg_gt_mse = mse_gt_sum / batches
        avg_gt_mae = mae_gt_sum / batches
        avg_gt_corr = corr_gt_sum / batches

        print(f"Batches: {batches}")
        print("\nReference match (TTNN vs torch, same weights):")
        print(f"MSE: {avg_ref_mse:.6f}")
        print(f"MAE: {avg_ref_mae:.6f}")
        print(f"Corr: {avg_ref_corr:.6f}")
        print("\nGround truth (TTNN vs dataset future values):")
        print(f"MSE: {avg_gt_mse:.6f}")
        print(f"MAE: {avg_gt_mae:.6f}")
        print(f"Corr: {avg_gt_corr:.6f}")
    finally:
        ttnn.CloseDevice(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Informer TTNN demo")
    subparsers = p.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Quick forward-pass sanity check")
    _add_model_args(smoke)
    smoke.add_argument("--batch", type=int, default=2, help="Batch size")

    bench = subparsers.add_parser("benchmark", help="Latency/throughput sweep")
    _add_model_args(bench)
    bench.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=_parse_batch_sizes("1,2,4,8,16"),
    )
    bench.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    bench.add_argument("--iters", type=int, default=20, help="Measured iterations")

    eval_p = subparsers.add_parser("eval", help="Dataset evaluation against torch reference")
    _add_model_args(eval_p)
    eval_p.add_argument("--dataset", type=str, default=None, help="Path to ETTh1/Weather CSV")
    eval_p.add_argument("--batch", type=int, default=2, help="Batch size")
    eval_p.add_argument("--stride", type=int, default=24, help="Sliding window stride")
    eval_p.add_argument("--max-windows", type=int, default=32, help="Max windows to evaluate")
    eval_p.add_argument("--time-features", choices=["calendar", "sin"], default="calendar", help="Time feature type")
    eval_p.add_argument("--normalize", action="store_true", help="Normalize using training split stats")
    eval_p.add_argument("--train-len", type=int, default=default_etth1_splits().train_len, help="Train split length")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True

    if args.command == "smoke":
        _smoke(args)
    elif args.command == "benchmark":
        _benchmark(args)
    elif args.command == "eval":
        _eval(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
