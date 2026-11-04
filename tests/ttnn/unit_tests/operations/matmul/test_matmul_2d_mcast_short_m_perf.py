# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip timing of the 2D-multicast matmuls that Gemma4 prefill runs per device on a BH Galaxy (8x4, CP8/TP4).

Chunk 4096 at CP8 gives 512 rows per device (16 tile rows), which Gemma4 runs on a 12x10 grid with per_core_M=2;
chunk 8192 gives 1024 rows (per_core_M=4). Shapes, dtypes, memory configs, program configs and compute configs are
copied from a Tracy capture of models/demos/gemma4_d_p (test_prefill_layer_perf_chunk_n, chunk 0), so the test has
no dependency on the model code and can be cherry-picked onto any commit. The *_in0_l1 / *_in0_dram cases are
controls that read in0 from the other memory than the model does.

Each case runs ITERS back-to-back matmuls inside one trace and reports the mean time per matmul over REPS replays
(median of replays). The test prints a table and checks PCC once per case; it has no perf threshold.

Run on two builds and compare the us/matmul column:
    pytest tests/ttnn/unit_tests/operations/matmul/test_matmul_2d_mcast_short_m_perf.py -s
"""

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

GRID = (12, 10)
ITERS = 32
REPS = 15
WARMUP_REPS = 3

# name, rows per device, K, N, per_core_N, out_subblock_w, in0 memory, output memory, fused activation
_GELU = "gelu_tanh"
CASES = [
    ("qkv_global", 4608, 12, 2, "dram", "l1", None),
    ("qkv_sliding", 4096, 11, 1, "dram", "l1", None),
    ("mlp_gate_gelu", 5376, 14, 2, "dram", "l1", _GELU),
    ("mlp_up", 5376, 14, 2, "dram", "l1", None),
    ("mlp_down", 5376, 14, 2, "l1", "dram", None),
    # Controls: same shapes with in0 read from the other memory, to separate the in0 read path from the shape.
    ("qkv_sliding_in0_l1", 4096, 11, 1, "l1", "l1", None),
    ("mlp_up_in0_l1", 5376, 14, 2, "l1", "l1", None),
    ("mlp_down_in0_dram", 5376, 14, 2, "dram", "dram", None),
]
K = 5376
ROWS = {"chunk4096": 512, "chunk8192": 1024}


def _program_config(rows, per_core_n, subblock_w, activation):
    per_core_m = -(-rows // ttnn.TILE_SIZE // GRID[1])
    # fp32 dest accumulation caps an output subblock at 4 tiles.
    subblock_h = next(h for h in (4, 3, 2, 1) if per_core_m % h == 0 and h * subblock_w <= 4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=14,  # largest divisor of K = 168 tiles up to 16
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, 1.0) if activation == _GELU else None,
    )


def _mem(kind):
    return ttnn.L1_MEMORY_CONFIG if kind == "l1" else ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("chunk", list(ROWS), ids=list(ROWS))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matmul_2d_mcast_short_m_perf(device, case, chunk):
    name, n, per_core_n, subblock_w, in0_mem, out_mem, activation = case
    rows = ROWS[chunk]
    grid = device.compute_with_storage_grid_size()
    if grid.x < GRID[0] or grid.y < GRID[1]:
        pytest.skip(f"needs a {GRID[0]}x{GRID[1]} compute grid, device has {grid.x}x{grid.y}")

    torch.manual_seed(0)
    a = torch.randn(1, 1, rows, K)
    b = torch.randn(1, 1, K, n) / K**0.5
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem(in0_mem))
    tt_b = ttnn.from_torch(
        b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    program_config = _program_config(rows, per_core_n, subblock_w, activation)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def run():
        return ttnn.linear(
            tt_a,
            tt_b,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=_mem(out_mem),
        )

    out = run()
    expected = a @ b
    if activation == _GELU:
        expected = torch.nn.functional.gelu(expected, approximate="tanh")
    assert_with_pcc(expected, ttnn.to_torch(out), 0.99)
    out.deallocate(True)

    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(ITERS):
        run().deallocate(True)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(WARMUP_REPS):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        per_matmul_us = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per_matmul_us.append((time.perf_counter() - t0) * 1e6 / ITERS)
    finally:
        ttnn.release_trace(device, trace)

    median = statistics.median(per_matmul_us)
    spread = (max(per_matmul_us) - min(per_matmul_us)) / median * 100
    logger.info(
        f"[mm2d_perf] {chunk:9} {name:18} M={rows} K={K} N={n} per_core_M={program_config.per_core_M} "
        f"per_core_N={per_core_n} in0={in0_mem} out={out_mem}: {median:8.2f} us/matmul (spread {spread:.1f}%)"
    )
