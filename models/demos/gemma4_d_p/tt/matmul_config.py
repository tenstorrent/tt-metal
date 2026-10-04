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


def prefill_matmul_program_config(hidden_states, weight, grid_x, grid_y, fused_activation=None, fp32_dest_acc=False):
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
    if per_core_m > _MAX_PER_CORE_M:
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


def prefill_1d_matmul_program_config(hidden_states, weight, grid, fused_activation=None, per_core_n=None):
    """1D in0-multicast config for short-M prefill projections, or None when it does not apply.

    With at most 8 tile rows of M, the 2D config uses only 8 of the grid's rows and reads each weight column
    block through one core. Here every core reads its own two weight columns from DRAM while the activations
    are multicast, which keeps more DRAM readers busy. A width-sharded activation (to_l1_width_sharded) is read
    one shard per K block.
    """
    per_core_n = per_core_n or _PER_CORE_N_1D
    tile = ttnn.TILE_SIZE
    m_tiles = hidden_states.padded_shape[-2] // tile
    k_tiles = hidden_states.padded_shape[-1] // tile
    n_tiles = weight.padded_shape[-1] // tile
    if not is_short_m(hidden_states) or n_tiles % per_core_n or n_tiles // per_core_n > grid.x * grid.y:
        return None
    # fp32 dest: at most 4 tiles (2 x 2) per output subblock.
    out_subblock_w = 2 if per_core_n % 2 == 0 else 1
    out_subblock_h = 2 if m_tiles % 2 == 0 else 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=(
            hidden_states.memory_config().shard_spec.shape[1] // tile
            if hidden_states.is_sharded()
            else _in0_block_w(k_tiles)
        ),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _in0_block_w(k_tiles):
    return max(d for d in range(1, min(k_tiles, 16) + 1) if k_tiles % d == 0)


# Most tile rows (per device) that the 1D projection config takes.
_MAX_SHORT_M_TILES = 8

# K tiles per core of a width-sharded short-M activation. An interleaved activation is read and multicast through a
# single core, which paces the 1D projections at chunk 2048; sharded, the cores holding its K slices multicast them
# in turn. 8 (or 7 where 8 does not divide K) keeps the K blocks deep; shallow blocks are much slower.
_SHARD_K_TILES = (8, 7, 6, 4)


def _is_short_m_rows(rows):
    return rows // ttnn.TILE_SIZE <= _MAX_SHORT_M_TILES


def is_short_m(x):
    """Whether x takes the 1D projection config: at most _MAX_SHORT_M_TILES tile rows."""
    return _is_short_m_rows(x.padded_shape[-2])


def _width_sharded_l1(device, rows, num_cores, shard_tiles):
    """L1 width-sharded layout: num_cores row-wise cores, each holding rows x shard_tiles tiles."""
    cores = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (rows, shard_tiles * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
    )


def short_m_sharded_memcfg(device, rows, k_tiles):
    """Width-sharded L1 layout of a short-M activation: K split over cores in _SHARD_K_TILES blocks."""
    grid = device.compute_with_storage_grid_size()
    shard = next((d for d in _SHARD_K_TILES if k_tiles % d == 0), None)
    if shard is None or k_tiles // shard > grid.x * grid.y:
        raise ValueError(
            f"short-M activation with {k_tiles} K tiles: needs a block depth in {_SHARD_K_TILES} that divides it "
            f"onto at most {grid.x * grid.y} cores"
        )
    return _width_sharded_l1(device, rows, k_tiles // shard, shard)


def to_l1_width_sharded(x):
    """x in the short-M sharded layout; x itself when it is already in it."""
    memcfg = short_m_sharded_memcfg(x.device(), x.padded_shape[-2], x.padded_shape[-1] // ttnn.TILE_SIZE)
    return x if x.memory_config() == memcfg else ttnn.to_memory_config(x, memcfg)


def short_m_gather_memcfg(x, tp_degree):
    """Output layout for the TP row all-gather of x: the short-M sharded layout when the gathered activation is short
    M, so the projections read it without a reshard; None (the default) otherwise."""
    rows = x.padded_shape[-2] * tp_degree
    if not _is_short_m_rows(rows):
        return None
    return short_m_sharded_memcfg(x.device(), rows, x.padded_shape[-1] // ttnn.TILE_SIZE)


def short_m_output_memcfg(x, weight, per_core_n=_PER_CORE_N_1D):
    """Width-sharded L1 output of a 1D short-M projection: per_core_n columns on each of its cores, which skips the
    interleaved write."""
    n_tiles = weight.padded_shape[-1] // ttnn.TILE_SIZE
    return _width_sharded_l1(x.device(), x.padded_shape[-2], n_tiles // per_core_n, per_core_n)
