# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Standalone `minimal_matmul` block-size sweep for the 6 FLUX shapes.

Both the fused `all_gather_minimal_matmul_async` (AGMM) op and the "separate" path drive their
matmul through `ttnn.experimental.minimal_matmul`. This benchmark finds the optimal per-core block
sizing (M_block, K_block, N_block) for each FLUX shape by calling `minimal_matmul` directly on a
single device, sweeping block sizes against ONE pre-built tensor set per shape.

Per-device matmul shape: the matmul inside the fused AGMM runs per device on
`(per_device_M, K, N)`, where `per_device_M = M // SP_SIZE` (the activation is sharded with M on the
sp_axis, then K is all-gathered to full width; the weight is replicated so N is full). See
`all_gather_minimal_matmul_async_program_factory.cpp:253` (`M = ag_output.physical_volume() / K`)
and the `(per_device_M > N)` transpose heuristic in test_all_gather_minimal_matmul_async_dev.py.
So this benchmark uses `m = M // SP_SIZE` for the activation rows and full K, N.

These runs are purely about speed -- outputs are junk (no PCC check). Per-config device timings are
captured via `tracy.signpost` markers + `ttnn.ReadDeviceProfiler`; run under the device profiler to
get DEVICE KERNEL DURATION per config (see "How to run" below).

How to run (profiler build). Write artifacts to a user-owned folder via -o (sets
TT_METAL_PROFILER_DIR for both the Python tool and the C++ runtime) to avoid the shared,
sometimes root-owned generated/profiler/.logs dir. Uses trace, so fast dispatch (no
TT_METAL_SLOW_DISPATCH_MODE):
    python -m tracy -r -p -o /home/rsalman/agmm-profiler \
        -m pytest models/tt_dit/tests/models/wan2_2/test_minimal_matmul_block_sweep.py -s
    # -> /home/rsalman/agmm-profiler/reports/<ts>/ops_perf_results_*.csv
    # Each config is delimited by an "AGMM_MM_SWEEP ..." signpost row (OP CODE column);
    # MinimalMatmulDeviceOperation rows carry DEVICE KERNEL DURATION [ns]. Per config there are
    # 4 matmul rows (compile + trace-capture + 2 replays); take the min / replay rows.
Quick harness smoke (no profiler needed):
    pytest models/tt_dit/tests/models/wan2_2/test_minimal_matmul_block_sweep.py -k flux22-1 -s

Standalone `minimal_matmul` has no force_transpose flag -- it auto-selects
`transpose = (per_device_M > N)`. For these 6 shapes only flux22-3 transposes; the rest run
non-transposed. The signpost records the actual transpose state.
"""

from itertools import product

import pytest
import torch
import tracy

import ttnn

# ----------------------------- tunable knobs (edit freely) -----------------------------
SP_SIZE = 4  # = mesh.shape[sp_axis] on the 4x8 Galaxy; per_device_M = M // SP_SIZE
GRID = (11, 10)  # matches the fused AGMM no-transpose config grid (x, y); clamped to device grid
K_BLOCK_MAX = 16  # cap on K_block candidates (K is zero-padded to a multiple of K_block)
MAX_DEST_VOLUME = 4  # subblock_h * subblock_w cap with fp32_dest_acc_en=True, full-sync off
NUM_ITERS = 2  # trace replays timed per config
TRACE_REGION_SIZE = 90112

# (M, K, N) FLUX shapes from test_all_gather_minimal_matmul_async_dev.py. M here is the FULL M;
# the benchmark uses per_device_M = M // SP_SIZE for the matmul.
FLUX_SHAPES = [
    (4096, 6144, 2304),  # flux22-1
    (2048, 6144, 2304),  # flux22-2
    (4096, 6144, 768),  # flux22-3
    (2048, 6144, 768),  # flux22-4
    (4096, 6144, 4608),  # flux22-5
    (3072, 6144, 1536),  # flux22-6
]
FLUX_IDS = ["flux22-1", "flux22-2", "flux22-3", "flux22-4", "flux22-5", "flux22-6"]

TILE = 32


# --------------------------------- helpers ---------------------------------
def _round_up(a, b):
    return ((a + b - 1) // b) * b


def _divisors(n):
    """Sorted divisors of n (>= 1). n is expected >= 1."""
    n = max(int(n), 1)
    return sorted({d for i in range(1, n + 1) for d in (i,) if n % i == 0})


def valid_subblocks(m_block, n_block, max_dest=MAX_DEST_VOLUME):
    """All valid (subblock_h, subblock_w): h | m_block, w | n_block, h*w <= max_dest.

    Swept exhaustively (the op constrains h*subblock_w to the DST budget). Ordered by descending
    area then descending subblock_w so the likely-best candidates run first.
    """
    pairs = [(h, w) for h in _divisors(m_block) for w in _divisors(n_block) if h * w <= max_dest]
    return sorted(pairs, key=lambda hw: (-hw[0] * hw[1], -hw[1]))


def _candidate_blocks(m, k, n, grid_x, grid_y):
    """Shape- & grid-aware block candidates, bounded by the factory's per-core tiling.

    Returns (transpose, m_block_cands, k_block_cands, n_block_cands).
    """
    transpose = m > n  # minimal_matmul auto-transposes by (M_received > N); no force flag
    in0_cores = grid_x if transpose else grid_y  # cores parallelizing M
    in1_cores = grid_y if transpose else grid_x  # cores parallelizing N

    m_tiles = m // TILE
    n_tiles = n // TILE
    k_tiles = k // TILE

    m_tiles_per_core = _round_up(m_tiles, in0_cores) // in0_cores
    n_tiles_per_core = _round_up(n_tiles, in1_cores) // in1_cores

    # Sweep block sizes from 1 (many small blocks) up to the per-core tile count (a single block
    # covering the whole per-core range, so K is read once). per-core counts are small after the
    # grid divide, so the full range stays bounded. Non-divisors are valid (tail/padded last block).
    m_block_cands = list(range(1, m_tiles_per_core + 1))
    n_block_cands = list(range(1, n_tiles_per_core + 1))
    # K is the contraction dim (not divided across cores); zero-padded to a multiple of K_block, so
    # any value is valid. Use divisors of K_tiles (no padding waste) capped at K_BLOCK_MAX.
    k_block_cands = [d for d in _divisors(k_tiles) if d <= K_BLOCK_MAX]
    return transpose, m_block_cands, k_block_cands, n_block_cands


# --------------------------------- the sweep ---------------------------------
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("M, K, N", FLUX_SHAPES, ids=FLUX_IDS)
def test_minimal_matmul_block_sweep(device, M, K, N):
    torch.manual_seed(0)
    m = M // SP_SIZE  # per-device activation rows

    # Resolve the core grid: prefer the requested full grid, clamp to the device's actual grid.
    dev_grid = device.compute_with_storage_grid_size()
    grid_x = min(GRID[0], dev_grid.x)
    grid_y = min(GRID[1], dev_grid.y)
    core_grid = ttnn.CoreCoord(grid_x, grid_y)

    transpose, m_block_cands, k_block_cands, n_block_cands = _candidate_blocks(m, K, N, grid_x, grid_y)
    print(
        f"\n[block-sweep] fullM={M} m={m} K={K} N={N} grid={grid_x}x{grid_y} transpose={int(transpose)} "
        f"M_block_cands={m_block_cands} K_block_cands={k_block_cands} N_block_cands={n_block_cands}"
    )

    # ----- one-time tensor setup (built once per shape, reused for every block config) -----
    torch_input = torch.randn((m, K), dtype=torch.float32)
    weight_input = torch.randn((K, N), dtype=torch.float32)
    bias_input = torch.randn((1, N), dtype=torch.float32)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(bias_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Match dev.py's compute config so the winning block sizes transfer back to the fused op.
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def run_op(cfg):
        return ttnn.experimental.minimal_matmul(
            tt_input,
            tt_weight,
            bias_tensor=tt_bias,
            compute_kernel_config=compute_config,
            config=cfg,
        )

    n_run = 0
    n_skip = 0
    for M_block, K_block, N_block in product(m_block_cands, k_block_cands, n_block_cands):
        for subblock_h, subblock_w in valid_subblocks(M_block, N_block):
            try:
                matmul_config = ttnn.MinimalMatmulConfig(
                    M_block_size=M_block,
                    K_block_size=K_block,
                    N_block_size=N_block,
                    subblock_h=subblock_h,
                    subblock_w=subblock_w,
                    compute_with_storage_grid_size=core_grid,
                )

                tracy.signpost(
                    f"AGMM_MM_SWEEP fullM={M} m={m}x{K}x{N} grid={grid_x}x{grid_y} transpose={int(transpose)} "
                    f"M_block={M_block} K_block={K_block} N_block={N_block} sh={subblock_h} sw={subblock_w}"
                )

                # Compile, then trace-capture a single matmul and replay it NUM_ITERS times.
                run_op(matmul_config)
                ttnn.synchronize_device(device)
                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                run_op(matmul_config)
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
                ttnn.synchronize_device(device)
                for _ in range(NUM_ITERS):
                    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                ttnn.release_trace(device, trace_id)
                ttnn.ReadDeviceProfiler(device)
                n_run += 1
            except Exception as e:  # noqa: BLE001 - invalid combos are expected; keep sweeping
                if isinstance(e, KeyboardInterrupt):
                    raise
                n_skip += 1
                print(
                    f"[block-sweep] skip M_block={M_block} K_block={K_block} N_block={N_block} "
                    f"sh={subblock_h} sw={subblock_w}: {e}"
                )
                continue

    print(f"[block-sweep] {FLUX_IDS[FLUX_SHAPES.index((M, K, N))]}: ran {n_run} configs, skipped {n_skip}")
