# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""2D multicast matmul program configs for Tafsut's per-token linears (R1-2).

Ported from chronos_forecast ``tt/program_configs.py`` (599000d80f5, 0b755e48966): M split over the grid rows, N over
the grid columns, the whole compute grid in use, instead of the default auto-config. Tafsut's linears see a
``(1, 1, B * S, d)`` activation, so M is already folded.
"""

from __future__ import annotations

import math

import ttnn

TILE = 32
# in0 + in1 double-buffered, plus the output block, in bf16 tiles (2 KiB each)
L1_TILE_BUDGET = 560
MAX_IN0_BLOCK_W = 12
MAX_OUT_BLOCK_H = 16


def _largest_divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(block_h: int, block_w: int, max_tiles: int) -> tuple[int, int]:
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h == 0 and block_w % w == 0 and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def matmul_2d(grid, m_tiles: int, k_tiles: int, n_tiles: int, *, fp32_acc: bool, relu: bool = False,
              max_in0_block_w: int = MAX_IN0_BLOCK_W, max_out_block_h: int = MAX_OUT_BLOCK_H):
    grid_x, grid_y = grid
    per_core_n = math.ceil(n_tiles / grid_x)
    per_core_m = math.ceil(m_tiles / grid_y)
    cols, rows = math.ceil(n_tiles / per_core_n), math.ceil(m_tiles / per_core_m)
    out_block_h = _largest_divisor_at_most(per_core_m, max_out_block_h)
    in0_block_w = 1
    for cand in range(min(k_tiles, max_in0_block_w), 0, -1):
        # an fp32 intermediate (fp32_dest_acc_en with packer L1 accumulation) or fp32 output takes 2 tiles' room
        tiles = 2 * out_block_h * cand + 2 * cand * per_core_n + (2 if fp32_acc else 1) * out_block_h * per_core_n
        if k_tiles % cand == 0 and tiles <= L1_TILE_BUDGET:
            in0_block_w = cand
            break
    sub_h, sub_w = _subblock(out_block_h, per_core_n, 4 if fp32_acc else 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        fuse_batch=True,
    )


def for_linear(x, w, ckc, relu: bool = False, **kw):
    """The 2D config of ``x @ w`` for a tile-aligned ``(1, 1, M, K)`` DRAM activation, else None (auto-config)."""
    xs, ws = tuple(x.padded_shape), tuple(w.padded_shape)
    if x.layout != ttnn.TILE_LAYOUT or xs[-1] != ws[-2] or x.shape[-1] % TILE or math.prod(xs[:-2]) != 1:
        return None
    g = x.device().compute_with_storage_grid_size()
    return matmul_2d((g.x, g.y), xs[-2] // TILE, ws[-2] // TILE, ws[-1] // TILE,
                     fp32_acc=bool(ckc.fp32_dest_acc_en), relu=relu, **kw)
