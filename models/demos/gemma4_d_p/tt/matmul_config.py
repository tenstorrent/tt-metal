# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit matmul blocking for the prefill projections."""

import ttnn

# Weight columns per core in the 1D config: with 2-row output subblocks, 2 x 2 tiles fill the fp32 dest.
_PER_CORE_N_1D = 2

# Tallest per-core output block measured to fit in L1 (chunk 8192 at CP8). Chunk 16384 gives 7 tiles
# per core, whose circular buffers need 1,660,032 B against 1,572,864 B of L1, so larger shapes keep
# ttnn's default config.
_MAX_PER_CORE_M = 4


def prefill_matmul_program_config(
    hidden_states, weight, grid_x, grid_y, fused_activation=None, fp32_dest_acc=False, max_per_core_m=_MAX_PER_CORE_M
):
    """2D-multicast program config for hidden_states @ weight on a grid_x x grid_y core grid, or None
    to keep ttnn's default when the per-core block would not fit in L1.

    ttnn's automatic config blocks these short-M (chunk / CP rows) prefill shapes poorly. Reading K
    in the largest block of up to 16 tiles that divides it keeps the projections near the DRAM roof.
    fp32_dest_acc must match the compute kernel config: fp32 accumulation halves the dest registers,
    so output subblocks are capped at 4 tiles instead of 8.
    """
    tile = ttnn.TILE_SIZE
    m_tiles = hidden_states.padded_shape[-2] // tile
    k_tiles = hidden_states.padded_shape[-1] // tile
    n_tiles = weight.padded_shape[-1] // tile
    per_core_m = ttnn.core.divup(m_tiles, grid_y)
    if per_core_m > max_per_core_m:
        return None
    per_core_n = ttnn.core.divup(n_tiles, grid_x)
    subblock_w = 2 if per_core_n % 2 == 0 else 1
    max_subblock_tiles = 4 if fp32_dest_acc else 8
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=_in0_block_w(k_tiles),
        out_subblock_h=next(h for h in (4, 3, 2, 1) if per_core_m % h == 0 and h * subblock_w <= max_subblock_tiles),
        out_subblock_w=subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def prefill_1d_matmul_program_config(hidden_states, weight, grid, fused_activation=None):
    """1D in0-multicast config for short-M prefill projections, or None when it does not apply.

    With at most 8 tile rows of M, the 2D config uses only 8 of the grid's rows and reads each weight column
    block through one core. Here every core reads its own two weight columns from DRAM while the activations
    are multicast, which keeps more DRAM readers busy.
    """
    tile = ttnn.TILE_SIZE
    m_tiles = hidden_states.padded_shape[-2] // tile
    k_tiles = hidden_states.padded_shape[-1] // tile
    n_tiles = weight.padded_shape[-1] // tile
    if m_tiles > 8 or n_tiles % _PER_CORE_N_1D or n_tiles // _PER_CORE_N_1D > grid.x * grid.y:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_in0_block_w(k_tiles),
        out_subblock_h=2 if m_tiles % 2 == 0 else 1,
        out_subblock_w=_PER_CORE_N_1D,
        per_core_M=m_tiles,
        per_core_N=_PER_CORE_N_1D,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _in0_block_w(k_tiles):
    return max(d for d in range(1, min(k_tiles, 16) + 1) if k_tiles % d == 0)
