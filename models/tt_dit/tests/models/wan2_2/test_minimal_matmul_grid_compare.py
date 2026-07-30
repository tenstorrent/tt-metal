# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone minimal_matmul: 12x10 grid (separate's grid) vs 11x10 grid (fused AGMM's grid).

Tests whether part of the fused-vs-separate gap is just the core-grid size. The non-transposed
fused AGMM cedes a column to the fabric mux and runs the matmul on an 11x10 grid (110 cores), while
a standalone/separate matmul can use the full 12x10 grid (120 cores).

For each FLUX shape (per-device dims: M//4 x 6144 x N) we run minimal_matmul on BOTH grids, each
with that grid's sweep-optimal block sizes (12x10 -> "separate" blocks, 11x10 -> "fused" blocks).
Single device, trace + tracy.signpost + ReadDeviceProfiler; parse with parse_mm_block_sweep.py.
"""

import pytest
import torch
import tracy

import ttnn

NUM_ITERS = 3
TRACE_REGION_SIZE = 90112
SP_SIZE = 4  # per_device_M = full_M // SP_SIZE

# (id, gx, gy, m_per_device, K, N, M_block, K_block, N_block, sh, sw) -- blocks are each grid's sweep winner.
CONFIGS = [
    ("flux22-1", 12, 10, 1024, 6144, 2304, 4, 4, 6, 2, 2),
    ("flux22-1", 11, 10, 1024, 6144, 2304, 4, 4, 7, 4, 1),
    ("flux22-2", 12, 10, 512, 6144, 2304, 2, 8, 6, 2, 2),
    ("flux22-2", 11, 10, 512, 6144, 2304, 2, 6, 7, 2, 1),
    ("flux22-3", 12, 10, 1024, 6144, 768, 3, 6, 3, 1, 3),
    ("flux22-3", 11, 10, 1024, 6144, 768, 3, 8, 3, 3, 1),
    ("flux22-4", 12, 10, 512, 6144, 768, 2, 8, 2, 2, 2),
    ("flux22-4", 11, 10, 512, 6144, 768, 2, 8, 3, 1, 3),
    ("flux22-5", 12, 10, 1024, 6144, 4608, 4, 8, 6, 2, 2),
    ("flux22-5", 11, 10, 1024, 6144, 4608, 4, 6, 7, 4, 1),
    ("flux22-6", 12, 10, 768, 6144, 1536, 3, 6, 4, 1, 4),
    ("flux22-6", 11, 10, 768, 6144, 1536, 3, 8, 5, 3, 1),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "shape_id, gx, gy, m, K, N, M_block, K_block, N_block, sh, sw",
    CONFIGS,
    ids=[f"{c[0]}-{c[1]}x{c[2]}" for c in CONFIGS],
)
def test_grid_compare(device, shape_id, gx, gy, m, K, N, M_block, K_block, N_block, sh, sw):
    torch.manual_seed(0)
    dev_grid = device.compute_with_storage_grid_size()
    if gx > dev_grid.x or gy > dev_grid.y:
        pytest.skip(f"grid {gx}x{gy} exceeds device grid {dev_grid.x}x{dev_grid.y}")
    core_grid = ttnn.CoreCoord(gx, gy)

    tt_input = ttnn.from_torch(
        torch.randn((m, K), dtype=torch.float32), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    tt_weight = ttnn.from_torch(
        torch.randn((K, N), dtype=torch.float32), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    tt_bias = ttnn.from_torch(
        torch.randn((1, N), dtype=torch.float32), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=core_grid,
    )

    def run_op():
        return ttnn.experimental.minimal_matmul(
            tt_input, tt_weight, bias_tensor=tt_bias, compute_kernel_config=compute_config, config=matmul_config
        )

    # AGMM_MM_SWEEP signpost format so parse_mm_block_sweep.py groups by (shape, grid).
    tracy.signpost(
        f"AGMM_MM_SWEEP fullM={m * SP_SIZE} m={m}x{K}x{N} grid={gx}x{gy} transpose={int(m > N)} "
        f"M_block={M_block} K_block={K_block} N_block={N_block} sh={sh} sw={sw}"
    )
    run_op()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    run_op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    for _ in range(NUM_ITERS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ttnn.release_trace(device, trace_id)
    ttnn.ReadDeviceProfiler(device)
