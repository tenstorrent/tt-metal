# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ring-1D matmul helpers for Gemma4 MLP weight prefetcher.

Ports the decode-path pieces of ``tt_transformers`` ``ModelArgs.matmul_1d_ring_config``
/ DRAM-width-sharded weight layout used with ``Prefetcher``. Attention is out of
scope — MLP-only (gate / up / down).
"""

from __future__ import annotations

import math

import ttnn

TILE = 32


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pad_n_to_ring_size(n: int, ring_size: int) -> int:
    """Pad N so each ring core gets a tile-aligned shard (BH prefetcher unit-test recipe)."""
    per_core = round_up(math.ceil(n / ring_size), TILE)
    return per_core * ring_size


def create_dram_sharded_mem_config(k: int, n_padded: int, dram_cores: int) -> ttnn.MemoryConfig:
    """DRAM width-sharded weight layout; N must already be ring-padded."""
    assert n_padded % dram_cores == 0, f"n_padded {n_padded} not divisible by dram_cores {dram_cores}"
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, n_padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def matmul_1d_ring_config(
    m: int,
    k: int,
    n_padded: int,
    ring_size: int,
    num_global_cb_receivers: int,
    untilize_out: bool = False,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for prefetcher ring matmul."""
    in0_block_w = k // ring_size // TILE
    while in0_block_w > 0 and (k // TILE) % in0_block_w != 0:
        in0_block_w -= 1
    if in0_block_w == 0:
        in0_block_w = 1

    out_block_h = m // TILE
    out_block_w = n_padded // ring_size // TILE
    assert out_block_w > 0, f"n_padded {n_padded} too small for ring_size {ring_size}"

    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    # ring_size = num_receiver_cores * dram_banks; on P150, banks=8 → grid 8×8 for 64.
    assert ring_size % 8 == 0, f"ring_size {ring_size} not divisible by 8 (dram banks)"
    grid = ttnn.CoreGrid(y=ring_size // 8, x=8)
    hop_core_range_set = ttnn.CoreRangeSet(set())

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        num_global_cb_receivers=num_global_cb_receivers,
        untilize_out=untilize_out,
    )


def activation_mem_config(k_padded: int, ring_size: int, receiver_core_range_set) -> ttnn.MemoryConfig:
    """L1 width-sharded activation on prefetcher receiver cores.

    ``k_padded`` must already be ``pad_n_to_ring_size(k, ring_size)`` so each
    shard width is tile-aligned (Gemma4 hidden 5376 → 6144 at ring_size=64).
    """
    assert k_padded % ring_size == 0, f"k_padded {k_padded} not divisible by ring_size {ring_size}"
    k_per_shard = k_padded // ring_size
    assert k_per_shard % TILE == 0, f"k_per_shard {k_per_shard} not tile-aligned"
    return ttnn.create_sharded_memory_config(
        shape=(TILE, k_per_shard),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def output_mem_config(n_padded: int, ring_size: int, receiver_core_range_set) -> ttnn.MemoryConfig:
    """L1 width-sharded matmul output on receiver cores (N already ring-padded)."""
    assert n_padded % ring_size == 0, f"n_padded {n_padded} not divisible by ring_size {ring_size}"
    n_per_shard = n_padded // ring_size
    assert n_per_shard % TILE == 0, f"n_per_shard {n_per_shard} not tile-aligned"
    return ttnn.create_sharded_memory_config(
        shape=(TILE, n_per_shard),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
