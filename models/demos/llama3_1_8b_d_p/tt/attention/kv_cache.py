# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B chunked-prefill KV cache.

Copied from ``gpt_oss_d_p/tt/attention/kv_cache.py`` (the only GQA donor; ``minimax_m3`` is
near-identical). Two persistent device caches (K and V) on the DeepSeek chunked-KV NdShard DRAM
substrate, written by
``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``.

The layout is canonical and copied verbatim, because two consumers pin it: the chunked ring SDPA
reads the block-cyclic sequence straight out of it, and the migration address walk computes DRAM
addresses from the same bank geometry. Changing a constant here does not raise — it degrades PCC or
migrates the wrong bytes.

**One departure from both donors, and it is the interesting one.** The recipe's fixed layout table
gives the per-chip shape as ``[num_users*num_layers, 1, seq_local, head_dim]`` — dim 1 is literally
``1``. That holds only when ``TP == n_kv_heads``: gpt-oss has 8 KV heads at TP=8, MiniMax-M3 has 4 at
TP=4, so both land exactly one KV head per chip. Llama 3.1 8B has **8 KV heads at TP=4**, so each
chip holds **2**, and dim 1 is ``num_kv_heads // tp``. The NdShard spec is unchanged (the shard is
``[1, 1, 32, head_dim]``, i.e. per head per 32-token block), and ``update_padded_kv_cache`` places no
constraint on the head dim outside its TP-dedup mode, which this cache does not use. See
``docs/SPEC_NOTES.md``: the spec template has no field for KV heads per chip, and the recipe's table
states a special case as a constant.

Batch dim is user-major: ``slot = user_id * num_layers + layer_idx``, so each user's layers stay
contiguous, matching ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx`` indexing and the ring
SDPA's ``kv_cache_batch_idx``.
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

    Two persistent device caches, each per-chip shape
    ``[num_users * num_layers, num_kv_heads_local, seq_local, head_dim]``:

      * ``k``, ``v`` — GQA K/V. KV heads shard on the TP cols at write time (2 per chip at TP=4);
        the sequence is SP-sharded block-cyclic on the ``sp`` rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    num_kv_heads_local: int
    head_dim: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    num_kv_heads_local=2,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`LlamaKVCache`.

    Deliberately NOT ``init_kvpe_cache`` — that is MLA-specific and allocates a single latent cache;
    a GQA model calling it gets one cache where it needs two. This owns the K/V pair and the
    user-major packing, and reuses the same DRAM NdShard spec (same bank grid, same 32-token
    contiguous shard) so ``update_padded_kv_cache`` writes into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 32). All layers allocate K/V slots.
        max_seq_len: per-user cache capacity in tokens, a multiple of ``TILE_SIZE * sp`` so
            ``seq_local = max_seq_len // sp`` is tile-aligned (matches build_indexed_rope's
            constraint and the 32-token DRAM shard).
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for Llama 3.1 8B).
        num_kv_heads_local: KV heads this chip holds = ``n_kv_heads // tp`` (2 at 8 KV heads, TP=4).
        cache_dtype: on-device cache dtype (spec: bfloat8_b).
    """
    sp = mesh_device.shape[sp_axis]
    # seq_local must be tile-aligned: the cache is TILE_LAYOUT and the DRAM NdShard is 32-token; this
    # is also build_indexed_rope's chunk constraint, so cache and rope layouts agree.
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); "
        f"seq_local must be tile-aligned"
    )
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
        for bank_id in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        # One shard per (head, 32-token block): unchanged from the donors, which is exactly why more
        # than one head per chip needs no change to the bank walk.
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc(dtype=cache_dtype):
        # WHICH KV heads a chip holds is decided at write time by how the input chunk is mesh-mapped,
        # not here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer;
        # content diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, num_kv_heads_local, seq_local, head_dim),
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
        num_kv_heads_local=num_kv_heads_local,
        head_dim=head_dim,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    The op requires TILE layout and ``input.dtype == cache.dtype``, so cast a copy to the cache dtype
    when needed (the original stays live for the attention op that follows). At
    ``kv_actual % 32 == 0`` chunk boundaries the per-device write offset is contiguous (block-cyclic
    degenerates to a reshape).
    """
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

    ``tt_k`` / ``tt_v`` are the per-device SP shards ``[1, n_kv_local, s_local, head_dim]`` — KV heads
    TP-sharded on the cols, sequence SP-sharded on the ``sp_axis`` rows — which is exactly the
    per-chip cache layout, so they write in place. ``kv_actual`` is the cumulative valid prefix before
    this chunk (0 for the first/only chunk).
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading/batch dim, so a batched tt_k/tt_v would silently write only slot_idx and drop the rest.
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    # The head dim must match what was allocated, or the write silently lands on the wrong rows.
    assert tt_k.shape[1] == kv_cache.num_kv_heads_local, (
        f"k has {tt_k.shape[1]} local KV heads but the cache was allocated for "
        f"{kv_cache.num_kv_heads_local}; n_kv_heads/tp must match allocate_kv_cache"
    )
    assert (
        tt_v.shape[1] == kv_cache.num_kv_heads_local
    ), f"v has {tt_v.shape[1]} local KV heads but the cache was allocated for {kv_cache.num_kv_heads_local}"
    # Fail loud on a bad slot/layer (otherwise a silent OOB write into another user's slot) and on a
    # misaligned chunk offset (the block-cyclic per-device write assumes a tile-aligned boundary).
    assert 0 <= slot_idx < kv_cache.num_users, f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})"
    assert 0 <= layer_idx < kv_cache.num_layers, f"layer_idx {layer_idx} out of range [0, {kv_cache.num_layers})"
    assert (
        kv_actual % ttnn.TILE_SIZE == 0
    ), f"kv_actual ({kv_actual}) must be tile-aligned (multiple of {ttnn.TILE_SIZE})"
    for cache, tensor in ((kv_cache.k, tt_k), (kv_cache.v, tt_v)):
        _write_one(
            cache,
            tensor,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=kv_cache.num_layers,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
