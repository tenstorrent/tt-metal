# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Explicit matmul configs for Timestral's per-token linears, ``(1, 1, M, K) @ (K, N)``.

The auto config is a 1D config that re-reads the activation once per N block (3-6x at the paper
shape, M = 98304). ``tests/perf/sweep_matmul.py`` picked two replacements (``doc/optimize/iter_01``):

    2D multicast     M over grid rows, N over grid columns; the activation is fused in the packer
    minimal_matmul   when N leaves 2D grid columns idle (N = 768 uses 8 of 11) and nothing is fused:
                     FF-down 1.81 vs 2.42 ms; with a fused sigmoid it loses (1.71 vs 1.48 ms)
"""

from __future__ import annotations

import math

# in0 + in1 double-buffered plus the output and packer-accumulation blocks, in bf16 tiles.
_L1_TILE_BUDGET = 640
_MAX_IN0_BLOCK_W = 12
_MAX_OUT_BLOCK_H = 12
# minimal_matmul (M, K, N) blocks and subblock, from tests/perf/sweep_subblocks.py (doc/optimize/iter_04):
# wo + residual 0.80 vs 0.88 ms for 8 / 8 / 8 at 1 x 8, w_down 1.35 vs 1.37, wv 0.725 vs 0.743.
_MINIMAL_BLOCKS = ((16, 8, 4), (4, 1))
_SWIGLU_M_BLOCK = 8
# minimal_matmul_split ((M, K, N) blocks, subblock) per chunk count, from tests/perf/sweep_qkv_split.py:
#   2  [Wv | Wg], [Wk | Wv]  1.90 ms vs 2.58 ms as two linears (doc/optimize/iter_04)
#   3  [Wq | Wk | Wv]        2.79 ms vs the fused 2D linear's 2.76 ms (doc/optimize/iter_03)
#   4  [Wq | Wk | Wv | Wg]   3.63 vs 2.62 + 1.36 ms device as the 3-chunk split plus wg (doc/optimize/iter_04)
_SPLIT_BLOCKS = {2: ((16, 4, 6), (8, 1)), 3: ((16, 12, 4), (2, 4)), 4: ((16, 8, 6), (4, 2))}


def _largest_divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(block_h: int, block_w: int, max_tiles: int) -> tuple[int, int]:
    """Largest ``h * w <= max_tiles`` dividing the block, widest first on ties."""
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h == 0 and block_w % w == 0 and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _out_block_h(per_core_m: int) -> tuple[int, int]:
    """``(out_block_h, per_core_M)``: one block for short M, else a block in ``[cap / 2, cap]`` that pads
    ``per_core_M`` least (paper 308 -> 11 x 28; long_context 106 -> 12 x 9 = 108)."""
    if per_core_m <= _MAX_OUT_BLOCK_H:
        return per_core_m, per_core_m
    candidates = range(_MAX_OUT_BLOCK_H, _MAX_OUT_BLOCK_H // 2 - 1, -1)
    block = min(candidates, key=lambda b: b * math.ceil(per_core_m / b))
    return block, block * math.ceil(per_core_m / block)


def grid_columns(n_tiles: int, grid_x: int) -> int:
    """Grid columns a 2D config uses for ``n_tiles`` output columns."""
    return math.ceil(n_tiles / math.ceil(n_tiles / grid_x))


def matmul_2d_config(grid, m_tiles: int, k_tiles: int, n_tiles: int, *, fused_activation=None, max_dest: int = 8):
    """Each core holds its whole N slice, so the activation is read once."""
    import ttnn

    grid_x, grid_y = grid
    per_core_n = math.ceil(n_tiles / grid_x)
    out_block_h, per_core_m = _out_block_h(math.ceil(m_tiles / grid_y))
    in0_block_w = 1
    for cand in range(min(k_tiles, _MAX_IN0_BLOCK_W), 0, -1):
        tiles = 2 * out_block_h * cand + 2 * cand * per_core_n + 2 * out_block_h * per_core_n
        if k_tiles % cand == 0 and tiles <= _L1_TILE_BUDGET:
            in0_block_w = cand
            break
    sub_h, sub_w = _subblock(out_block_h, per_core_n, max_dest)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_columns(n_tiles, grid_x), math.ceil(m_tiles / per_core_m)),
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


def _blocked_config(grid, m_tiles: int, k_tiles: int, n_tiles: int, blocks, max_dest: int):
    """``MinimalMatmulConfig`` from ``((M, K, N) caps, preferred subblock)``; the subblock falls back to the
    largest legal one when the capped blocks or the dest size rule it out."""
    import ttnn

    (m_cap, k_cap, n_cap), (sub_h, sub_w) = blocks
    m_block = min(m_cap, m_tiles)
    n_block = _largest_divisor_at_most(n_tiles, n_cap)
    if m_block % sub_h or n_block % sub_w or sub_h * sub_w > max_dest:
        sub_h, sub_w = _subblock(m_block, n_block, max_dest)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=_largest_divisor_at_most(k_tiles, k_cap),
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )


def minimal_matmul_config(grid, m_tiles: int, k_tiles: int, n_tiles: int, *, max_dest: int = 8):
    return _blocked_config(grid, m_tiles, k_tiles, n_tiles, _MINIMAL_BLOCKS, max_dest)


def split_config(grid, m_tiles: int, k_tiles: int, chunk_tiles: int, chunks: int, *, max_dest: int = 8):
    """``minimal_matmul_split`` config for ``chunks`` outputs of ``chunk_tiles`` columns each."""
    return _blocked_config(grid, m_tiles, k_tiles, chunk_tiles, _SPLIT_BLOCKS[chunks], max_dest)


def swiglu_config(grid, m_tiles: int, k_tiles: int, n_out_tiles: int, *, max_dest: int = 8):
    """``minimal_matmul(fuse_swiglu=True)`` blocks: one gate/up tile pair per N block, K blocks up to 12
    (MLP front half 3.85 ms vs 7.96 ms split; ``tests/perf/sweep_swiglu.py``, ``doc/optimize/iter_02``);
    subblock 2 x 2 (2.51 vs 2.55 ms at 4 x 2; ``tests/perf/sweep_subblocks.py``, ``doc/optimize/iter_04``)."""
    import ttnn

    m_block = min(_SWIGLU_M_BLOCK, m_tiles)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=_largest_divisor_at_most(k_tiles, _MAX_IN0_BLOCK_W),
        N_block_size=2,
        subblock_h=_largest_divisor_at_most(m_block, min(2, max_dest // 2)),
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )


def fused_activation(name: str | None):
    import ttnn

    if name is None:
        return None
    ops = {"silu": ttnn.UnaryOpType.SILU, "sigmoid": ttnn.UnaryOpType.SIGMOID}
    return ttnn.UnaryWithParam(ops[name])
