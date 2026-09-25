# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill KV cache: two persistent block-cyclic SP-sharded caches (K post-RoPE, V raw).

Canonical layout (recipe section 5.1), copied from ``minimax_m3/tt/attention/kv_cache.py``:

  * per-chip shape ``[num_users * num_layers, n_local_kv_heads, seq_local, head_dim]``, with
    ``n_local_kv_heads = 8 / TP = 2`` for this model (M3 / GPT-OSS carry 1; see bringup_log D2);
  * ``slot = user_id * num_layers + layer_idx`` (user-major, layers contiguous);
  * DRAM ``NdShardSpec`` shard ``[1, 1, 32, head_dim]``, ``ROUND_ROBIN_1D`` over the DRAM bank grid,
    ``NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`` tokens per bank;
  * sequence SP-sharded block-cyclic on the SP rows, ``seq_local = max_seq_len // sp``, period = chunk;
  * zeroed allocation with ``ReplicateTensorToMesh`` (content diverges on first write);
  * written by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.utils import blockcyclic_positions
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class KVCache(KvCaches):
    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    num_local_kv_heads: int
    head_dim: int


def allocate_kv_cache(
    mesh_device,
    mesh_config,
    *,
    num_layers: int,
    max_seq_len: int,
    num_users: int = 1,
    num_local_kv_heads: int = 2,
    head_dim: int = 128,
    cache_dtype=ttnn.bfloat8_b,
) -> KVCache:
    sp = mesh_config.sp
    assert (
        max_seq_len % (ttnn.TILE_SIZE * sp) == 0
    ), f"max_seq_len {max_seq_len} must be a multiple of 32 * sp ({32 * sp})"
    seq_local = max_seq_len // sp
    banks = get_num_dram_banks(mesh_device)
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)]),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def alloc():
        # Every chip gets the same zeroed buffer; which heads / tokens it holds is decided at write time.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, num_local_kv_heads, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return KVCache(
        k=alloc(),
        v=alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        num_local_kv_heads=num_local_kv_heads,
        head_dim=head_dim,
    )


def _write_one(cache_tensor, chunk, *, user_id, layer_idx, num_layers, kv_actual, sp_axis):
    # update_padded_kv_cache needs TILE input in the cache's dtype; cast a copy, keep the original live
    # for the attention op that follows.
    src = chunk if chunk.dtype == cache_tensor.dtype else ttnn.typecast(chunk, cache_tensor.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache_tensor,
        src,
        slot_idx=user_id,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if src is not chunk:
        src.deallocate(True)


def write_kv_chunk(cache: KVCache, tt_k, tt_v, *, user_id: int, layer_idx: int, kv_actual: int, sp_axis: int):
    """Write this chunk's per-chip K ``[1, n_local_kv, chunk_local, D]`` and V at global offset ``kv_actual``."""
    assert 0 <= user_id < cache.num_users and 0 <= layer_idx < cache.num_layers, (user_id, layer_idx)
    for tensor, chunk in ((cache.k, tt_k), (cache.v, tt_v)):
        _write_one(
            tensor,
            chunk,
            user_id=user_id,
            layer_idx=layer_idx,
            num_layers=cache.num_layers,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )


def read_slot_kv(mesh_device, cache: KVCache, user_id: int):
    """Host copies ``(k, v)`` of one user's slots: ``[num_layers, num_kv_heads, max_seq_len, D]`` fp32,
    heads concatenated over TP, sequence still in on-device block-cyclic order."""
    start, end = user_id * cache.num_layers, (user_id + 1) * cache.num_layers
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, 1))

    def block(tensor):
        s = list(tensor.shape)
        # DRAM_MEMORY_CONFIG on the slice: slicing into another ND-shard miscomputes the host read-back.
        sl = ttnn.slice(tensor, [start, 0, 0, 0], [end, s[1], s[2], s[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(sl, mesh_composer=composer).float()
        ttnn.deallocate(sl)
        return host

    return block(cache.k), block(cache.v)


def naturalize(block, n_tokens: int, sp: int, chunk_size: int, max_seq_len: int):
    """Un-rotate a ``[..., max_seq_len, D]`` block-cyclic host block to natural token order ``[..., n_tokens, D]``."""
    positions = blockcyclic_positions(sp, chunk_size, max_seq_len)
    natural = torch.empty_like(block)
    natural[..., positions, :] = block
    return natural[..., :n_tokens, :]
