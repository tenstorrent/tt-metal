#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark: rm_binary_eltwise vs ttnn.add (RM vs tile) on Wan shapes.

For each (dtype, shape) pair the script runs three back-to-back profiled segments:
  Segment 3k-2 : rm_binary_eltwise   — custom row-major kernel, 3 DRAM passes
  Segment 3k-1 : ttnn.add on RM     — composite tilize + add + untilize, 5 DRAM passes
  Segment 3k   : ttnn.add on TILE   — add only (no layout conversion), 1 DRAM pass

Tracy signposts with descriptive headers delimit the segments so that
tracy_summarize.py can show per-segment op breakdowns and total kernel time.

Usage
-----
  # 1. Run with device profiler enabled:
  TT_METAL_DEVICE_PROFILER=1 python benchmark_rm_binary_eltwise.py

  # 2. Analyse the generated CSV (path printed at the end of the run):
  python tracy_tools/tracy_summarize.py \\
      --input generated/profiler/reports/<run>/ops_perf_results_*.csv

Notes
-----
- Program cache is enabled before the warmup phase so that compilation cost is
  not included in the profiled segments.
- ttnn.synchronize_device() is called before each signpost to ensure all
  outstanding kernels have retired before the segment boundary is written.
- N_WARMUP=3 is enough to compile all kernels and warm L2/DRAM caches.
- N_ITER=10 gives a stable average; each segment therefore contains 10 op rows
  per atomic kernel (e.g. 10×add only for the tile path, 10×tilize+add+untilize for RM).
"""

import sys
from pathlib import Path
import pytest

import torch
import ttnn
from tracy import signpost

# Add repo root to path so the model import works when run directly.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


# [1, 1, 60, 104, 384]
# [1, 1, 120, 208, 384]
# 1, 1, 480, 832, 96]
# [1, 2, 120, 208, 384]
# 1, 4, 240, 416, 192]
# [1, 4, 480, 832, 96]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Wan VAE decoder residual-add shapes (B, T, H, W, C).
# All C values are multiples of 32, so padded_shape == logical_shape for RM.
# WAN_SHAPES = [
#     (1, 1,  90,  160, 384),   # mid_block / resnets.0
#     (1, 2, 180,  320, 384),   # up_blocks.1 / resnets.{0,1}
#     (1, 4, 360,  640, 192),   # up_blocks.2 / resnets.0
#     (1, 4, 720, 1280,  96),   # up_blocks.3 / resnets.0
# ]


WAN_SHAPES = [
    (1, 1, 60, 104, 384),
    (1, 1, 120, 208, 384),
    (1, 1, 480, 832, 96),
    (1, 2, 120, 208, 384),
    (1, 4, 240, 416, 192),
    (1, 4, 480, 832, 96),
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

N_WARMUP = 3  # iterations to compile kernels and warm caches (not profiled)
N_ITER = 10  # iterations per profiled segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dtype_name(dtype: ttnn.DataType) -> str:
    return "bf16" if dtype == ttnn.bfloat16 else "fp32"


def _torch_dtype(dtype: ttnn.DataType) -> torch.dtype:
    return torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32


def _shape_tag(shape: tuple) -> str:
    return "x".join(str(d) for d in shape)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(device: ttnn.Device) -> list[tuple[int, str]]:
    """Run all benchmark cases and return the segment legend.

    Signpost naming convention expected by analyze_tracy_perf.py:
      "<impl_type>-<config_name>-start"  →  opens a measurement block
      "<impl_type>-<config_name>-end"    →  closes a measurement block

    parse_implementation() splits on the first '-', so:
      "rm_binary_eltwise-bf16/1x1x90x160x384"  →  impl_type="rm_binary_eltwise"
      "ttnn_add-..." / "ttnn_add_tile-..."     →  impl_type="ttnn_add" / "ttnn_add_tile"

    The aggregator groups by config_name, so all three implementations for the
    same shape/dtype appear together in the summary table.
    """
    legend: list[tuple[int, str]] = []
    seg_idx = 0

    for dtype in DTYPES:
        tdtype = _torch_dtype(dtype)
        dname = _dtype_name(dtype)

        for shape in WAN_SHAPES:
            stag = _shape_tag(shape)
            cfg = f"{dname}/{stag}"  # config_name used by parse_implementation
            print(f"  benchmarking dtype={dname}  shape={shape} …", flush=True)

            # --- allocate tensors -------------------------------------------
            A_torch = torch.randn(shape, dtype=tdtype)
            B_torch = torch.randn(shape, dtype=tdtype)

            A_rm = ttnn.from_torch(
                A_torch,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            B_rm = ttnn.from_torch(
                B_torch,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            A_tile = ttnn.from_torch(
                A_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            B_tile = ttnn.from_torch(
                B_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # --- warmup (outside measurement blocks — not captured) ----------
            for _ in range(N_WARMUP):
                ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op="add")
                ttnn.add(A_rm, B_rm)  # composite: tilize → add → untilize
                ttnn.add(A_tile, B_tile)  # add on tile (no layout conversion)
            ttnn.synchronize_device(device)

            # --- measurement block: custom rm_binary_eltwise ----------------
            signpost(header=f"rm_binary_eltwise-{cfg}-start")
            for _ in range(N_ITER):
                ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op="add")
            ttnn.synchronize_device(device)
            signpost(header=f"rm_binary_eltwise-{cfg}-end")

            seg_idx += 1
            legend.append((seg_idx, f"rm_binary_eltwise  dtype={dname}  shape={shape}"))

            # --- measurement block: ttnn.add (composite tilize+add+untilize) -
            signpost(header=f"ttnn_add-{cfg}-start")
            for _ in range(N_ITER):
                ttnn.add(A_rm, B_rm)
            ttnn.synchronize_device(device)
            signpost(header=f"ttnn_add-{cfg}-end")

            seg_idx += 1
            legend.append((seg_idx, f"ttnn.add (RM→tile→RM)  dtype={dname}  shape={shape}"))

            # --- measurement block: ttnn.add on TILE (no tilize/untilize) -----
            signpost(header=f"ttnn_add_tile-{cfg}-start")
            for _ in range(N_ITER):
                ttnn.add(A_tile, B_tile)
            ttnn.synchronize_device(device)
            signpost(header=f"ttnn_add_tile-{cfg}-end")

            seg_idx += 1
            legend.append((seg_idx, f"ttnn.add (TILE only)  dtype={dname}  shape={shape}"))

            # Free device tensors explicitly to avoid OOM on large shapes.
            ttnn.deallocate(A_rm)
            ttnn.deallocate(B_rm)
            ttnn.deallocate(A_tile)
            ttnn.deallocate(B_tile)

    # Flush device profiler data to CSV.
    ttnn.ReadDeviceProfiler(device)

    return legend


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def test_bench_eltwise():
    print("=" * 70)
    print("rm_binary_eltwise micro-benchmark")
    print(f"  shapes : {len(WAN_SHAPES)}")
    print(f"  dtypes : {[_dtype_name(d) for d in DTYPES]}")
    print(f"  warmup : {N_WARMUP} iter   profiled : {N_ITER} iter")
    print("=" * 70)

    device = ttnn.open_device(device_id=0)

    try:
        legend = run_benchmark(device)
    finally:
        ttnn.close_device(device)

    # Print legend so the user can map segment numbers to cases.
    print()
    print("Segment legend (matches SEGMENT column in tracy_summarize output):")
    print("-" * 70)
    for idx, label in legend:
        print(f"  Segment {idx:3d}  {label}")
    print()
    print("To analyse results:")
    print("  python tracy_tools/analyze_tracy_perf.py \\")
    print("      --csv generated/profiler/reports/<run>/ops_perf_results_*.csv")
    print()
    print("Or run everything in one shot:")
    print('  python tracy_tools/analyze_tracy_perf.py --run "pytest benchmark_rm_binary_eltwise.py"')
    print()
    print("The summary table groups by config (dtype/shape) and shows avg/std/min/max")
    print("for rm_binary_eltwise vs ttnn_add vs ttnn_add_tile side-by-side.")
