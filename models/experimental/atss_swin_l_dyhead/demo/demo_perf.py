#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ATSS Swin-L DyHead performance benchmark.

Measures and compares per-component and end-to-end latency between:
  1. PyTorch reference model (CPU)
  2. TTNN hybrid model (device backbone/FPN/head + CPU DyHead)

Reports a detailed breakdown of each pipeline stage.

Usage:
  cd $TT_METAL_HOME
  source python_env/bin/activate
  export ARCH_NAME=wormhole_b0
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages

  python3 models/experimental/atss_swin_l_dyhead/demo/demo_perf.py
  python3 models/experimental/atss_swin_l_dyhead/demo/demo_perf.py --runs 5
"""

import argparse
import json
import time
from pathlib import Path

import torch


def benchmark_pytorch(checkpoint_path, input_tensor, img_shape, num_warmup=2, num_runs=3):
    """Benchmark PyTorch reference model with per-component timing."""
    from models.experimental.atss_swin_l_dyhead.reference.model import (
        build_atss_model,
        load_mmdet_checkpoint,
    )
    from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess

    print("\n" + "=" * 70)
    print("PyTorch Reference Model Benchmark")
    print("=" * 70)

    model = build_atss_model()
    load_mmdet_checkpoint(model, checkpoint_path)
    model.eval()

    x = model.preprocess(input_tensor)

    for _ in range(num_warmup):
        with torch.no_grad():
            feats = model.backbone(x)
            fpn_feats = model.fpn(tuple(feats))
            dy_feats = model.dyhead(list(fpn_feats))
            cls_s, reg_s, cent_s = model.head(tuple(dy_feats))
            _ = atss_postprocess(cls_s, reg_s, cent_s, img_shape=img_shape)

    timings = {"preprocess": [], "backbone": [], "fpn": [], "dyhead": [], "head": [], "postprocess": [], "total": []}

    for run in range(num_runs):
        with torch.no_grad():
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            x = model.preprocess(input_tensor)
            timings["preprocess"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            feats = model.backbone(x)
            timings["backbone"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            fpn_feats = model.fpn(tuple(feats))
            timings["fpn"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            dy_feats = model.dyhead(list(fpn_feats))
            timings["dyhead"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            cls_s, reg_s, cent_s = model.head(tuple(dy_feats))
            timings["head"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            results = atss_postprocess(cls_s, reg_s, cent_s, img_shape=img_shape)
            timings["postprocess"].append((time.perf_counter() - t0) * 1000)

            timings["total"].append((time.perf_counter() - t_total_start) * 1000)

    return timings, results


def benchmark_ttnn(checkpoint_path, input_tensor, img_shape, device, num_warmup=2, num_runs=3):
    """Benchmark TTNN hybrid model with per-component timing."""
    import ttnn
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
    from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess

    print("\n" + "=" * 70)
    print("TTNN Hybrid Model Benchmark")
    print("=" * 70)

    from models.experimental.atss_swin_l_dyhead.common import ATSS_PAD_SIZE_DIVISOR

    _, _, h, w = input_tensor.shape
    pad_h = (ATSS_PAD_SIZE_DIVISOR - h % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
    pad_w = (ATSS_PAD_SIZE_DIVISOR - w % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
    padded_h, padded_w = h + pad_h, w + pad_w

    print(f"[TTNN] Building model (input={padded_h}x{padded_w})...")
    ttnn_model = TtATSSModel.from_checkpoint(checkpoint_path, device, input_h=padded_h, input_w=padded_w)

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = ttnn_model.predict(input_tensor, img_shape)

    timings = {
        "preprocess": [],
        "to_device": [],
        "backbone": [],
        "fpn": [],
        "fpn_to_host": [],
        "dyhead": [],
        "head_to_device": [],
        "head": [],
        "head_to_host": [],
        "postprocess": [],
        "total": [],
    }

    for run in range(num_runs):
        with torch.no_grad():
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            x = ttnn_model.preprocess(input_tensor)
            timings["preprocess"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            x_ttnn = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            timings["to_device"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            backbone_feats = ttnn_model.backbone(x_ttnn)
            timings["backbone"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            fpn_feats = ttnn_model.fpn(backbone_feats)
            timings["fpn"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            torch_fpn = []
            for feat in fpn_feats:
                torch_fpn.append(ttnn.to_torch(ttnn.from_device(feat)).float())
            timings["fpn_to_host"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            dy_feats = ttnn_model.dyhead(torch_fpn)
            timings["dyhead"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            ttnn_dy = []
            for feat in dy_feats:
                ttnn_dy.append(
                    ttnn.from_torch(
                        feat,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            timings["head_to_device"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            cls_ttnn, reg_ttnn, cent_ttnn = ttnn_model.head(ttnn_dy)
            timings["head"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            cls_torch = [ttnn.to_torch(ttnn.from_device(c)).float() for c in cls_ttnn]
            reg_torch = [ttnn.to_torch(ttnn.from_device(r)).float() for r in reg_ttnn]
            cent_torch = [ttnn.to_torch(ttnn.from_device(c)).float() for c in cent_ttnn]
            timings["head_to_host"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            results = atss_postprocess(cls_torch, reg_torch, cent_torch, img_shape=img_shape)
            timings["postprocess"].append((time.perf_counter() - t0) * 1000)

            timings["total"].append((time.perf_counter() - t_total_start) * 1000)

    return timings, results


def print_timings(name, timings):
    """Print timing breakdown."""
    print(f"\n{'─' * 60}")
    print(f"  {name} — Timing Breakdown (ms)")
    print(f"{'─' * 60}")
    print(f"  {'Stage':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

    for stage, vals in timings.items():
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        min_v = min(vals)
        max_v = max(vals)
        marker = " ◀" if stage == "total" else ""
        print(f"  {stage:<20} {mean_v:>10.1f} {min_v:>10.1f} {max_v:>10.1f}{marker}")
    print(f"{'─' * 60}")


def print_comparison(pytorch_timings, ttnn_timings):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("  Performance Comparison Summary")
    print("=" * 70)

    shared_stages = ["preprocess", "backbone", "fpn", "dyhead", "head", "postprocess", "total"]
    print(f"\n  {'Stage':<20} {'PyTorch (ms)':>14} {'TTNN (ms)':>14} {'Speedup':>10}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*10}")

    for stage in shared_stages:
        pt_vals = pytorch_timings.get(stage, [])
        tt_vals = ttnn_timings.get(stage, [])
        if not pt_vals or not tt_vals:
            continue
        pt_mean = sum(pt_vals) / len(pt_vals)
        tt_mean = sum(tt_vals) / len(tt_vals)
        speedup = pt_mean / tt_mean if tt_mean > 0 else float("inf")
        marker = " ◀" if stage == "total" else ""
        print(f"  {stage:<20} {pt_mean:>14.1f} {tt_mean:>14.1f} {speedup:>9.2f}x{marker}")

    ttnn_transfer_stages = ["to_device", "fpn_to_host", "head_to_device", "head_to_host"]
    transfer_total = 0
    for stage in ttnn_transfer_stages:
        vals = ttnn_timings.get(stage, [])
        if vals:
            transfer_total += sum(vals) / len(vals)
    if transfer_total > 0:
        print(f"\n  TTNN data transfer overhead: {transfer_total:.1f} ms")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ATSS performance benchmark")
    parser.add_argument("--checkpoint", default=None, help="mmdet checkpoint path")
    parser.add_argument("--input-h", type=int, default=640, help="Input height")
    parser.add_argument("--input-w", type=int, default=640, help="Input width")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark runs")
    _default_out = str(Path(__file__).resolve().parent.parent / "results" / "perf")
    parser.add_argument("--output-dir", default=_default_out, help="Output directory")
    parser.add_argument("--skip-ttnn", action="store_true", help="Skip TTNN benchmark")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmark")
    args = parser.parse_args()

    from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT

    checkpoint = args.checkpoint or ATSS_CHECKPOINT
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    torch.manual_seed(42)
    input_tensor = torch.randint(0, 256, (1, 3, args.input_h, args.input_w), dtype=torch.float32)
    img_shape = (args.input_h, args.input_w)

    print(f"Input: {args.input_h}x{args.input_w}")
    print(f"Warmup: {args.warmup}, Benchmark runs: {args.runs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input_shape": [args.input_h, args.input_w],
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
    }

    pytorch_timings = None
    ttnn_timings = None

    if not args.skip_pytorch:
        pytorch_timings, _ = benchmark_pytorch(
            checkpoint,
            input_tensor,
            img_shape,
            num_warmup=args.warmup,
            num_runs=args.runs,
        )
        print_timings("PyTorch Reference", pytorch_timings)
        report["pytorch"] = {
            k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)} for k, v in pytorch_timings.items() if v
        }

    if not args.skip_ttnn:
        import ttnn

        device = ttnn.open_device(device_id=0, l1_small_size=32768)
        try:
            ttnn_timings, _ = benchmark_ttnn(
                checkpoint,
                input_tensor,
                img_shape,
                device,
                num_warmup=args.warmup,
                num_runs=args.runs,
            )
            print_timings("TTNN Hybrid", ttnn_timings)
            report["ttnn"] = {
                k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)} for k, v in ttnn_timings.items() if v
            }
        finally:
            ttnn.close_device(device)

    if pytorch_timings and ttnn_timings:
        print_comparison(pytorch_timings, ttnn_timings)

    report_path = output_dir / "perf_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
