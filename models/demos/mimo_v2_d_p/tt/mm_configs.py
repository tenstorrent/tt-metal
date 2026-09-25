# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""2D-multicast matmul program configs for the prefill projections ([M, K] x [K, N], M = tokens per chip).

M (row tiles) is split over grid rows, N over grid cols; out subblock <= 4 tiles (fp32 dest acc)."""

import math

import ttnn

TILE = 32


def _largest_div(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def mm_2d_config(device, M, K, N, *, in0_block_w=4, grid_x=None, grid_y=None, dst_tiles=4):
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid_x or grid.x, grid_y or grid.y
    gx, gy = min(gx, grid.x), min(gy, grid.y)
    Mt, Kt, Nt = math.ceil(M / TILE), K // TILE, N // TILE
    if Kt % in0_block_w:
        return None
    # use as many grid rows as divide the M tiles evenly (avoid ragged rows)
    per_m = math.ceil(Mt / gy)
    gy = math.ceil(Mt / per_m)
    per_n = math.ceil(Nt / gx)
    gx = math.ceil(Nt / per_n)
    sub_w = _largest_div(per_n, dst_tiles)
    sub_h = _largest_div(per_m, max(1, dst_tiles // sub_w))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_m,
        out_block_w=per_n,
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


L1_BUDGET = 1_480_000  # per-core CB bytes (calibrated: est. 1.44 MB fits, 1.56 MB overflows)


def _l1_bytes(pc, in0_bytes=2048, in1_bytes=1088, out_bytes=2048, interm_bytes=4096):
    m, n, bw = pc.per_core_M, pc.per_core_N, pc.in0_block_w
    return 2 * m * bw * in0_bytes + 2 * bw * n * in1_bytes + m * n * (out_bytes + interm_bytes)


def best_mm_config(device, M, K, N):
    """Measured best for the MiMo projections on BH (M = 640 / 2048 tokens per chip, bf8 weights, HiFi2):
    full-width grid, the widest in0 block that fits L1 (see tests/perf/test_matmul_configs.py)."""
    for bw in (8, 4, 2, 1):
        pc = mm_2d_config(device, M, K, N, in0_block_w=bw)
        if pc is not None and _l1_bytes(pc) <= L1_BUDGET:
            return pc
    return None
