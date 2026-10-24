# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Explicit matmul program configs for t0's per-token linears, ``(1, 1, M, K) @ (K, N)`` (pattern from Timestral).

The auto config is a 1D config that re-reads the activation once per N block (QKV at M = 98304: 8 N blocks,
12.99 ms at 15.7 % of peak). The families here, swept per role in ``tests/perf/sweep_matmul.py``:

    2D multicast     M over grid rows, N over grid columns; the activation is fused in the packer
    minimal_matmul   blocked M / K / N with an explicit subblock

Every matmul accumulates in fp32 (fp32 dest), so a subblock holds at most 4 tiles.
"""

from __future__ import annotations

import math

MAX_DEST = 4  # tiles in DST with fp32 accumulation
_L1_TILE_BUDGET = 640
_MAX_IN0_BLOCK_W = 12
_MAX_OUT_BLOCK_H = 12


def largest_divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def subblock(block_h: int, block_w: int, max_tiles: int = MAX_DEST) -> tuple[int, int]:
    """Largest ``h * w <= max_tiles`` dividing the block, widest first on ties."""
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h == 0 and block_w % w == 0 and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _out_block_h(per_core_m: int, cap: int) -> tuple[int, int]:
    """``(out_block_h, per_core_M)``: one block for short M, else a block in ``[cap / 2, cap]`` that pads least."""
    if per_core_m <= cap:
        return per_core_m, per_core_m
    block = min(range(cap, cap // 2 - 1, -1), key=lambda b: b * math.ceil(per_core_m / b))
    return block, block * math.ceil(per_core_m / block)


def matmul_2d_config(
    grid, m_tiles: int, k_tiles: int, n_tiles: int, *, fused_activation=None, out_block_cap: int = _MAX_OUT_BLOCK_H
):
    """Each core holds its whole N slice, so the activation is read once per grid row."""
    import ttnn

    grid_x, grid_y = grid
    per_core_n = math.ceil(n_tiles / grid_x)
    out_block_h, per_core_m = _out_block_h(math.ceil(m_tiles / grid_y), out_block_cap)
    in0_block_w = 1
    for cand in range(min(k_tiles, _MAX_IN0_BLOCK_W), 0, -1):
        tiles = 2 * out_block_h * cand + 2 * cand * per_core_n + 2 * out_block_h * per_core_n
        if k_tiles % cand == 0 and tiles <= _L1_TILE_BUDGET:
            in0_block_w = cand
            break
    sub_h, sub_w = subblock(out_block_h, per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(math.ceil(n_tiles / per_core_n), math.ceil(m_tiles / per_core_m)),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


def minimal_config(grid, m_tiles: int, k_tiles: int, n_tiles: int, blocks: tuple):
    """``MinimalMatmulConfig`` from ``(M, K, N, subblock_h, subblock_w)`` caps; blocks shrink to divisors of the shape
    and the subblock falls back to the largest legal one."""
    import ttnn

    m_cap, k_cap, n_cap, sub_h, sub_w = blocks
    m_block = min(m_cap, m_tiles)
    n_block = largest_divisor_at_most(n_tiles, n_cap)
    if m_block % sub_h or n_block % sub_w or sub_h * sub_w > MAX_DEST:
        sub_h, sub_w = subblock(m_block, n_block)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=largest_divisor_at_most(k_tiles, k_cap),
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )


def fused_activation(name: str | None):
    import ttnn

    if name is None:
        return None
    return ttnn.UnaryWithParam({"silu": ttnn.UnaryOpType.SILU, "relu": ttnn.UnaryOpType.RELU}[name])
