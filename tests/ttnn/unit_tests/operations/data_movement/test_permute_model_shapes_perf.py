# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Permute perf harness over real model attention shapes.

Goal: a single, profileable workload that exercises ``ttnn.permute`` on the
exact shapes/dims used by the attention blocks of device-perf-measured models
(ViT, BERT, SqueezeBERT, Falcon7B, Segformer), so the current permute kernel
can be measured before/after an optimization ("old vs new").

Each case is tagged with the permute *device-op variant* it is expected to
dispatch to (see permute_device_operation.cpp). The interesting rows for
optimization are the ones tagged UNOPTIMIZED: the K reshape-transpose
``(0,2,3,1)`` in ViT/BERT, which lands on ``MultiCoreTiledGeneric``. The
``(0,2,1,3)`` Q/context permutes and the ``(0,1,3,2)`` WH-swaps already hit
optimized paths (tile-row-invariant / transpose_wh) and are included as
controls — a good optimization should not regress them.

--------------------------------------------------------------------------
How to measure (old vs new)
--------------------------------------------------------------------------
Correctness + a quick wall-clock loop (no profiler needed):

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/data_movement/test_permute_model_shapes_perf.py::test_permute_workload

Per-op DEVICE KERNEL DURATION [ns] via the device profiler (this is the
number to capture before and after your kernel change):

    pytest tests/ttnn/unit_tests/operations/data_movement/test_permute_model_shapes_perf.py::test_permute_device_perf

(the device-perf test shells out to the profiler itself and prints AVG/MIN/MAX
per-op duration for each case; it does not assert a baseline so it always
reports a number you can diff across branches).
"""

import pytest
import torch

import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    from tracy import signpost
except ImportError:  # tracy not built; workload still runs, just unmarked

    def signpost(_message):
        pass


# Iterations executed *inside* the profiler signpost window. Per-op duration =
# (summed DEVICE KERNEL DURATION between signposts) / N_PROFILE_ITERS.
N_WARMUP = 5
N_PROFILE_ITERS = 30


# (id, pre-permute shape, dims, dtype, layout, predicted dispatch variant)
#
# Shapes are the tensor *fed into* ttnn.permute at the model call site.
PERMUTE_CASES = [
    # --- ViT base (google/vit-base-patch16-224), batch 8, seq 197, 12 heads x 64 ---
    ("vit_base_q", (8, 197, 12, 64), (0, 2, 1, 3), ttnn.bfloat16, "TILE", "tile_row_invariant (optimized)"),
    ("vit_base_k", (8, 197, 12, 64), (0, 2, 3, 1), ttnn.bfloat16, "TILE", "tiled_generic (UNOPTIMIZED)"),
    ("vit_base_ctx", (8, 12, 197, 64), (0, 2, 1, 3), ttnn.bfloat16, "TILE", "tile_row_invariant (optimized)"),
    # --- BERT large (phiyodr/bert-large-finetuned-squad2), batch 8, seq 384, 16 heads x 64 ---
    ("bert_large_q", (8, 384, 16, 64), (0, 2, 1, 3), ttnn.bfloat16, "TILE", "tile_row_invariant (optimized)"),
    ("bert_large_k", (8, 384, 16, 64), (0, 2, 3, 1), ttnn.bfloat16, "TILE", "tiled_generic (UNOPTIMIZED)"),
    ("bert_large_ctx", (8, 16, 384, 64), (0, 2, 1, 3), ttnn.bfloat16, "TILE", "tile_row_invariant (optimized)"),
    # --- SqueezeBERT, batch 8, seq 384, 12 heads x 64 (transpose_for_scores WH-swap) ---
    ("squeezebert_qv", (8, 12, 384, 64), (0, 1, 3, 2), ttnn.bfloat16, "TILE", "tiled_invariant WH-swap (optimized)"),
    # --- Falcon7B prefill, batch 1, seq 256, 71 heads x 64 (ttnn.transpose(k,-2,-1)) ---
    ("falcon7b_k", (1, 71, 256, 64), (0, 1, 3, 2), ttnn.bfloat16, "TILE", "tiled_invariant WH-swap (optimized)"),
    # --- Segformer-b0 stage 2, batch 1, seq 256, 5 heads x 32, bf8 ---
    ("segformer_b0_k", (1, 5, 256, 32), (0, 1, 3, 2), ttnn.bfloat8_b, "TILE", "tiled_invariant WH-swap (optimized)"),
]

CASE_IDS = [c[0] for c in PERMUTE_CASES]


def _layout(layout_str):
    return ttnn.TILE_LAYOUT if layout_str == "TILE" else ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize("case", PERMUTE_CASES, ids=CASE_IDS)
def test_permute_workload(device, case):
    """Correctness check + a profileable permute loop for one model shape.

    This is the body the device profiler / tracy measures. It runs only
    ``ttnn.permute`` so every op row in the profiler log belongs to
    PermuteDeviceOperation.
    """
    name, shape, dims, dtype, layout_str, variant = case
    logger.info(f"permute case={name} shape={shape} dims={dims} dtype={dtype} -> variant={variant}")

    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_output = torch.permute(torch_input, dims)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=_layout(layout_str),
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- correctness: a future "new" kernel must still match torch ---
    tt_output = ttnn.permute(tt_input, dims)
    out = ttnn.to_torch(tt_output).to(torch.float32)
    assert list(out.shape) == list(torch_output.shape), f"{out.shape} != {torch_output.shape}"
    pcc = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert_with_pcc(torch_output, out, pcc)

    # --- profiling window: warmup outside, measured iters inside signposts ---
    for _ in range(N_WARMUP):
        ttnn.permute(tt_input, dims)
    ttnn.synchronize_device(device)

    signpost("start")
    for _ in range(N_PROFILE_ITERS):
        ttnn.permute(tt_input, dims)
    ttnn.synchronize_device(device)
    signpost("stop")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("case", PERMUTE_CASES, ids=CASE_IDS)
def test_permute_device_perf(case):
    """Run the workload under the device profiler and report per-op duration.

    No baseline is asserted — this prints AVG/MIN/MAX DEVICE KERNEL DURATION
    [ns] for PermuteDeviceOperation so you can capture the number on the
    current kernel and again after your optimization.
    """
    from models.perf.device_perf_utils import run_device_perf

    name = case[0]
    subdir = "permute_model_shapes"
    num_iterations = 3
    test_file = "tests/ttnn/unit_tests/operations/data_movement/test_permute_model_shapes_perf.py"
    command = f"pytest {test_file}::test_permute_workload[{name}]"
    cols = ["DEVICE KERNEL"]

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size=1,
        op_name="PermuteDeviceOperation",
        has_signposts=True,
    )

    summed_ns = post_processed_results["AVG DEVICE KERNEL DURATION [ns]"]
    per_op_ns = summed_ns / N_PROFILE_ITERS
    logger.info(
        f"\n=== permute device perf [{name}] ==="
        f"\n  variant       : {case[5]}"
        f"\n  summed ({N_PROFILE_ITERS} iters): {summed_ns:.1f} ns"
        f"\n  per-op        : {per_op_ns:.1f} ns"
        f"\n  min/max summed: {post_processed_results['MIN DEVICE KERNEL DURATION [ns]']:.1f}"
        f" / {post_processed_results['MAX DEVICE KERNEL DURATION [ns]']:.1f} ns"
    )
