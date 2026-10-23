# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Explicit 2D-multicast blocking for the prefill projections (MATMUL_PROGRAM=2d), after gemma4_d_p tt/matmul_config.py.

ttnn's default config for these short-M (640 rows per chip) projections runs on MATMUL_CORE_GRID; the explicit one uses
the whole compute grid: the widest column count that divides N, every core row, K read in blocks of up to 16 tiles.
"""

import ttnn

from .settings import settings

_MAX_PER_CORE_M = 4  # taller per-core blocks overflow L1 (gemma4_d_p, measured)


def linear_kwargs(mesh, x, w, *, activation=None, fp32_acc=True, grid=None, transpose_b=False):
    """kwargs for ttnn.linear(x, w, ...): the explicit program config (activation fused into it) when MATMUL_PROGRAM
    is 2d (2d_all: also the transposed [out, in] global weights) and the block fits, else core_grid=grid and the
    activation by name."""
    mode = settings.get("MATMUL_PROGRAM")
    if mode == "2d_all" or (mode == "2d" and not transpose_b):
        cfg = program_config(mesh, x, w, activation=activation, fp32_acc=fp32_acc, transpose_b=transpose_b)
        if cfg is not None:
            return {"program_config": cfg}
    return {"core_grid": grid, **({"activation": activation} if activation else {})}


def program_config(mesh, x, w, *, activation=None, fp32_acc=True, transpose_b=False):
    tile = ttnn.TILE_SIZE
    g = mesh.compute_with_storage_grid_size()
    m_tiles = x.padded_shape[-2] // tile
    k_tiles = x.padded_shape[-1] // tile
    n_tiles = w.padded_shape[-2 if transpose_b else -1] // tile
    grid_x = max(d for d in range(1, g.x + 1) if n_tiles % d == 0)
    per_core_n = n_tiles // grid_x
    per_core_m = -(-m_tiles // g.y)
    if per_core_m > _MAX_PER_CORE_M:
        return None
    grid_y = -(-m_tiles // per_core_m)
    subblock_w = 2 if per_core_n % 2 == 0 else 1
    max_subblock = 4 if fp32_acc else 8
    subblock_h = next(h for h in (4, 3, 2, 1) if per_core_m % h == 0 and h * subblock_w <= max_subblock)
    fused = None
    if activation == "gelu_tanh":
        fused = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
    elif activation is not None:
        raise ValueError(activation)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=max(d for d in range(1, min(k_tiles, 16) + 1) if k_tiles % d == 0),
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused,
    )
