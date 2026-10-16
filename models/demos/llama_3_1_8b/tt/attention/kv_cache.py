# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B chunked-prefill KV cache.

Layout is identical to GPT-OSS / DeepSeek: two persistent device caches, each per-chip shape
``[num_users*num_layers, n_kv_local, seq_local, head_dim]``, written by
``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``.

CRITICAL difference from GPT-OSS: n_kv_local = 2 (not 1). With TP=4 and 8 KV heads, each chip
holds 2 KV heads (8 KV heads / 4 TP columns = 2 per chip). The sequence is SP-sharded
block-cyclic on the ``sp`` rows (SP=8).

The batch dim is user-major: ``slot = user_id * num_layers + layer_idx`` so each user's layers stay
contiguous, matching ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx`` indexing.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard in allocate_kv_cache and the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class LlamaKVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Two persistent device caches, each per-chip shape ``[num_users*num_layers, n_kv_local, seq_local, head_dim]``:

      * ``k``, ``v`` — GQA K/V. Under TP=4 each chip holds 2 KV heads (heads sharded on the TP cols
        at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    n_kv_local=2,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`LlamaKVCache`.

    Args:
        num_layers: layers per user (full model = 32). All layers allocate K/V slots.
        max_seq_len: per-user cache capacity in tokens, a multiple of ``TILE_SIZE * sp`` so ``seq_local =
            max_seq_len // sp`` is tile-aligned.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        n_kv_local: number of KV heads per TP chip (2 for Llama 3.1 8B with TP=4).
        head_dim: per-head width (128 for Llama 3.1 8B).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate).
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        max_seq_len % (ttnn.TILE_SIZE * sp) == 0
    ), f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); seq_local must be tile-aligned"
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
        for bank_id in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc(dtype=cache_dtype):
        # Per-chip cache holds n_kv_local heads (2 for Llama 3.1 8B). Allocated zeroed +
        # ReplicateTensorToMesh: every chip gets the same empty buffer; content diverges on the
        # first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, n_kv_local, seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return LlamaKVCache(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache."""
    src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        src,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if src is not tensor:
        src.deallocate(True)


def write_kv_chunk(kv_cache: LlamaKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for the first/only chunk).
    """
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    assert 0 <= slot_idx < kv_cache.num_users, f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})"
    assert 0 <= layer_idx < kv_cache.num_layers, f"layer_idx {layer_idx} out of range [0, {kv_cache.num_layers})"
    assert (
        kv_actual % ttnn.TILE_SIZE == 0
    ), f"kv_actual ({kv_actual}) must be tile-aligned (multiple of {ttnn.TILE_SIZE})"
    _write_one(
        kv_cache.k,
        tt_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
    _write_one(
        kv_cache.v,
        tt_v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
