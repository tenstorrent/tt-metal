# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Matmul program configs for the per-chip [640, K] x [K, N] projections and the per-head W_uk / W_uv (MM_CONFIG).

ttnn's automatic choice is slow for the down projections whose output is the fp32 TP partial (wo, the shared
expert's / dense MLP's w2: K 48-192 tiles, N 192 tiles). The rule, as the DeepSeek / GLM prefill
(moe/tt_shared_expert.py get_program_configs, mla/mla_config.py 640-row entries): M over the grid's 10 rows, N over
its columns (12 or 11, whichever pads N less), the largest K block <= 16 tiles that divides K, the widest 1-row
output subblock in the 4-tile fp32 DEST.
"""

import math

import ttnn

from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

MAX_IN0_BLOCK_W = 16


def _grid_x(n_t: int, gx_max: int) -> int:
    best = None
    for gx in (gx_max, gx_max - 1):
        pad = math.ceil(n_t / gx) * gx - n_t
        if best is None or pad < best[0]:
            best = (pad, gx)
    return best[1]


def config_2d(m_t: int, k_t: int, n_t: int, grid) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    gy = grid.y
    gx = min(_grid_x(n_t, grid.x), n_t)  # N narrower than the grid (the indexer's wproj, wk): one column per tile
    pm = math.ceil(m_t / gy)
    pn = math.ceil(n_t / gx)
    ibw = max(d for d in range(1, min(k_t, MAX_IN0_BLOCK_W) + 1) if k_t % d == 0)
    sw = max(w for w in range(1, min(pn, 4) + 1) if pn % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def config_batched(m_t: int, k_t: int, n_t: int, batch: int, grid):
    """Per-head matmuls (W_uk, W_uv: [1, heads, 640, K] x [1, heads, K, N]): blocks of 4 x N tiles per core."""
    pm = 4 if m_t % 4 == 0 else m_t
    pn = n_t if n_t <= 16 else max(d for d in range(1, 17) if n_t % d == 0)
    if batch * (m_t // pm) * (n_t // pn) > grid.x * grid.y:
        return None
    ibw = max(d for d in range(1, min(k_t, 8) + 1) if k_t % d == 0)
    sw = max(w for w in range(1, min(pn, 4) + 1) if pn % w == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
    )


def get(name, x, w, transpose_b: bool = True, grid=None):
    """The program config for x [1, 1, M, K] @ w ([N, K] with transpose_b), or None (ttnn picks) under MM_CONFIG auto
    or for a shape the rule does not cover (batched, fewer than 10 row tiles)."""
    if settings.get("MM_CONFIG") == "auto":
        return None
    xs, ws = x.padded_shape, w.padded_shape
    if len(xs) != 4 or len(ws) != 4 or xs[0] != 1 or ws[0] != 1:
        return None
    t = ttnn.TILE_SIZE
    m_t, k_t = xs[-2] // t, xs[-1] // t
    n_t = (ws[-2] if transpose_b else ws[-1]) // t
    grid = grid or x.device().compute_with_storage_grid_size()
    if xs[1] > 1 and ws[1] == xs[1]:
        return config_batched(m_t, k_t, n_t, xs[1], grid)
    if xs[1] != 1 or ws[1] != 1 or m_t < grid.y:
        return None
    return config_2d(m_t, k_t, n_t, grid)


def get_indexer(x, w):
    """INDEXER_MM=tuned: the indexer's projections, x [1, 1, M, K] @ w [K, N] (stored [K, N], no transpose),
    M 640 (wk) or 160 (wq_b, wproj: the TP-partitioned query rows): the 2D rule with the grid's rows cut to M's tiles."""
    if settings.get("INDEXER_MM") != "tuned":
        return None
    xs, ws = x.padded_shape, w.padded_shape
    if len(xs) != 4 or len(ws) != 4 or xs[0] * xs[1] != 1 or ws[0] * ws[1] != 1:
        return None
    t = ttnn.TILE_SIZE
    m_t, k_t, n_t = xs[-2] // t, xs[-1] // t, ws[-1] // t
    g = x.device().compute_with_storage_grid_size()
    return config_2d(m_t, k_t, n_t, ttnn.CoreCoord(g.x, min(g.y, m_t)))


def out_memory(name):
    """Output memory of a TP-partial projection: L1 under MM_CONFIG tuned_l1 (the reduce_scatter reads it there)."""
    return ttnn.L1_MEMORY_CONFIG if settings.get("MM_CONFIG") == "tuned_l1" else ttnn.DRAM_MEMORY_CONFIG
