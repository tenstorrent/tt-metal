# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed dual-family prefill KV cache for Gemma 4 migration.

This is **not** the interleaved demo cache in ``kv_cache.py``. That layout is
``[batch, local_heads, seq, head_dim]`` INTERLEAVED DRAM / ``ReplicateTensorToMesh``
and is not the GPT-OSS/MiniMax NdShard packing the chunk table describes.

Two families (local and global do not share a shape):

  * ``local_k`` / ``local_v`` — sliding-window K and V, per-chip shape
    ``[num_users * num_layers, nkv_per_dev, sliding_window // sp, 256]``.
    Unused slots on full-attention layers are cheap (the ring is 1024 tokens).
    User-major fold ``slot * num_layers + layer`` matches
    ``update_padded_kv_cache``'s semantic ``layer_idx``.
  * ``global_kv`` — merged ``[K_roped_rotary | V]`` row of width 640, per-chip
    shape ``[num_users * n_global_layers, 1, max_seq_len // sp, 640]``.
    Compact: do not pad sliding layers out to full context. Batch fold is
    ``slot * n_global_layers + global_index(layer)``.

Each tensor is DRAM NdShard ROUND_ROBIN_1D, shard ``[1, 1, 32, row_dim]``.
Local allocates ``nkv_per_dev`` heads on dim 1 (4 at TP=4); global allocates 1.
Allocated zeroed + ``ReplicateTensorToMesh``: every chip gets the same empty
buffer; content diverges on the first write.

See ``tt/runners/kv_chunk_table.py`` for the address table that describes this
layout. Config order is the prefill↔decode contract.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gemma4.tt.runners.kv_chunk_table import (
    DEFAULT_GLOBAL_HEAD_DIM,
    DEFAULT_GLOBAL_N_KV,
    DEFAULT_GLOBAL_ROTARY_FACTOR,
    DEFAULT_LOCAL_HEAD_DIM,
    DEFAULT_LOCAL_N_KV,
    DEFAULT_SLIDING_WINDOW,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    default_layer_types,
    global_row_dim,
    nkv_per_device,
    num_global_layers,
)


@dataclass
class Gemma4PrefillKVCache(KvCaches):
    """Externally-owned packed prefill caches for Gemma 4 (local K/V + merged global)."""

    local_k: ttnn.Tensor
    local_v: ttnn.Tensor
    global_kv: ttnn.Tensor
    num_users: int
    num_layers: int
    num_global_layers: int
    max_seq_len: int
    sliding_window: int
    sp: int
    layer_types: tuple


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    sliding_window=DEFAULT_SLIDING_WINDOW,
    local_n_kv_heads=DEFAULT_LOCAL_N_KV,
    global_n_kv_heads=DEFAULT_GLOBAL_N_KV,
    local_head_dim=DEFAULT_LOCAL_HEAD_DIM,
    global_head_dim=DEFAULT_GLOBAL_HEAD_DIM,
    global_rotary_factor=DEFAULT_GLOBAL_ROTARY_FACTOR,
    layer_types=None,
    cache_dtype=ttnn.bfloat8_b,
) -> Gemma4PrefillKVCache:
    """Allocate the three packed prefill caches (local K, local V, merged global).

    ``max_seq_len`` sizes the global tensor only. Local tensors always allocate
    the sliding-window ring (``sliding_window // sp``).
    """
    types = tuple(layer_types) if layer_types is not None else default_layer_types(num_layers)
    if len(types) != num_layers:
        raise ValueError(f"layer_types length {len(types)} != num_layers {num_layers}")

    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[1 - sp_axis]
    tile_align = ttnn.TILE_SIZE * sp
    assert max_seq_len % tile_align == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({tile_align})"
    )
    assert sliding_window % tile_align == 0, (
        f"sliding_window ({sliding_window}) must be a multiple of TILE_SIZE*sp ({tile_align})"
    )

    local_nkv = nkv_per_device(local_n_kv_heads, tp)
    global_nkv = nkv_per_device(global_n_kv_heads, tp)
    n_global = num_global_layers(types)
    g_row = global_row_dim(global_head_dim, global_rotary_factor)
    local_seq = sliding_window // sp
    global_seq = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
        for bank_id in range(get_num_dram_banks(mesh_device))
    ]

    def _alloc(batch, n_heads, seq, row_dim):
        nd_shard_spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, row_dim],
            grid=ttnn.CoreRangeSet(core_ranges),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)
        return ttnn.from_torch(
            torch.zeros(batch, n_heads, seq, row_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return Gemma4PrefillKVCache(
        local_k=_alloc(num_users * num_layers, local_nkv, local_seq, local_head_dim),
        local_v=_alloc(num_users * num_layers, local_nkv, local_seq, local_head_dim),
        global_kv=_alloc(num_users * n_global, global_nkv, global_seq, g_row),
        num_users=num_users,
        num_layers=num_layers,
        num_global_layers=n_global,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
        sp=sp,
        layer_types=types,
    )
