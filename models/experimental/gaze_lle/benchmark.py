# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gaze-LLE inference benchmark harness.

Emits results in a grep-parseable format so the experiment loop can pick them up:
    inference_speed <fps>
    accuracy <pcc_percent>
    peak_dram <bytes>

Defaults to ViT-B/14 backbone at 448x448 with single-person head bbox.
Runs either the pure torch reference (``--impl torch``) or the TT-NN
implementation (``--impl ttnn``). In ttnn mode we additionally compute PCC of
the predicted heatmap versus the torch reference to report ``accuracy``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import List

import torch

from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def peak_dram_bytes(device=None) -> int:
    try:
        import ttnn

        if device is not None and hasattr(ttnn, "get_memory_info"):
            return int(ttnn.get_memory_info(device).dram.peak_allocated_bytes)
    except Exception:
        pass
    return 0


def run_torch_benchmark(variant: str, iters: int, warmup: int, inout: bool) -> dict:
    torch.manual_seed(0)
    model = build_gaze_lle(variant=variant, inout=inout).eval()
    img = torch.randn(1, 3, 448, 448)
    bboxes: List = [(0.3, 0.2, 0.6, 0.5)]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(img, bboxes)
        start = time.perf_counter()
        for _ in range(iters):
            out = model(img, bboxes)
        elapsed = time.perf_counter() - start

    fps = iters / elapsed if elapsed > 0 else 0.0
    ref_heatmap = out["heatmap"].detach().clone()

    return {
        "inference_speed": fps,
        "accuracy": 100.0,
        "peak_dram": 0,
        "ref_heatmap": ref_heatmap,
    }


def run_ttnn_benchmark(variant: str, iters: int, warmup: int, inout: bool, device_id: int) -> dict:
    import ttnn

    from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE

    torch.manual_seed(0)
    ref_model = build_gaze_lle(variant=variant, inout=inout).eval()
    ref_img = torch.randn(1, 3, 448, 448)
    bboxes: List = [(0.3, 0.2, 0.6, 0.5)]

    with torch.no_grad():
        ref_out = ref_model(ref_img, bboxes)

    device = ttnn.open_device(device_id=device_id)
    try:
        tt_model = TtGazeLLE(ref_model, device, inout=inout)

        for _ in range(warmup):
            tt_out = tt_model(ref_img, bboxes)
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)

        start = time.perf_counter()
        for _ in range(iters):
            tt_out = tt_model(ref_img, bboxes)
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        fps = iters / elapsed if elapsed > 0 else 0.0

        heatmap_pcc = pcc(ref_out["heatmap"], tt_out["heatmap"])
        accuracy = max(0.0, heatmap_pcc) * 100.0
        dram = peak_dram_bytes(device)
    finally:
        ttnn.close_device(device)

    return {
        "inference_speed": fps,
        "accuracy": accuracy,
        "peak_dram": dram,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["torch", "ttnn"], default="torch")
    parser.add_argument("--variant", default="vitb14", choices=["vitb14", "vitl14"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--inout", action="store_true", default=True)
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("GAZE_LLE_DEVICE", "2")))
    args = parser.parse_args(argv)

    torch.set_grad_enabled(False)
    if args.impl == "torch":
        res = run_torch_benchmark(args.variant, args.iters, args.warmup, args.inout)
    else:
        res = run_ttnn_benchmark(args.variant, args.iters, args.warmup, args.inout, args.device_id)

    print(f"impl {args.impl}")
    print(f"variant {args.variant}")
    print(f"inference_speed {res['inference_speed']:.4f}")
    print(f"accuracy {res['accuracy']:.4f}")
    print(f"peak_dram {res['peak_dram']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
