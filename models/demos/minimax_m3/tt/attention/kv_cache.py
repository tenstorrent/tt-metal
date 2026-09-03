# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, get_num_dram_banks


@dataclass
class MiniMaxKVCache:
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Three persistent device caches, each per-chip shape ``[num_users*num_layers, 1, seq_local, head_dim]``
    (DeepSeek chunked-KV NdShard DRAM layout). They are written by
    ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``:

      * ``k``, ``v``   — main GQA K/V. Under TP=cols each chip holds 1 head (heads sharded on the TP cols
                         at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.
      * ``index_k``    — MSA lightning-indexer key (a single shared head). Same per-chip shape; at write
                         time it is REPLICATED across the TP cols (one shared head, NOT head-sharded), and
                         only the MSA layers (3-59) populate it — dense layers leave their slots zeroed.

    The batch dim is user-major: ``slot = user_id * num_layers + layer_idx`` so each user's layers stay
    contiguous, matching ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx`` indexing.

    STEP 1: allocation + plumbing only. The write (step 2) and the cache-read attention path (step 4)
    are not wired yet — the single-shot prefill still runs the no-cache forward for its logits.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    index_k: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> MiniMaxKVCache:
    """Allocate the three external prefill KV caches (K, V, index_k). See :class:`MiniMaxKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (that is MLA-specific and allocates a single cache): this owns
    the M3 GQA triple and the user-major packing. It reuses the same DRAM NdShard spec (same bank grid +
    32-token contiguous shard) so ``update_padded_kv_cache`` can write into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 60). All three caches allocate all layers; only the MSA
            layers will write ``index_k`` (dense slots stay zeroed — capacity is cheap, packing stays simple).
        max_seq_len: per-user cache capacity in tokens, a multiple of ``sp``. ``seq_local = max_seq_len // sp``.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for M3 main K/V and the index head alike).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate + the device golden check).
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
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
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds (or whether index_k is
        # replicated across cols) is decided at write time by how the input chunk is mesh-mapped, not
        # here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer; content
        # diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # index_k feeds the indexer's HARD top-16 block selection (not a smooth softmax like K/V), so bf8's
    # ~2-3 mantissa bits perturb the block scores enough to flip many picks -> chunked vs one-shot
    # selection diverges (~7/16 overlap) -> residual drift compounding over MSA layers. Cache it in bf16
    # (M3_INDEX_CACHE_BF16=1) to keep selection stable; it's tiny (1 head) and only the indexer reads it.
    index_dtype = ttnn.bfloat16 if os.getenv("M3_INDEX_CACHE_BF16") == "1" else cache_dtype

    return MiniMaxKVCache(
        k=_alloc(),
        v=_alloc(),
        index_k=_alloc(index_dtype),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    The op requires TILE layout and input.dtype == cache.dtype, so cast a copy to the cache's dtype when
    needed (the original stays live for the attention op that follows). At ``kv_actual % 32 == 0`` chunk
    boundaries the per-device write offset is contiguous (block-cyclic degenerates to a reshape).
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


def write_kv_chunk(kv_cache: MiniMaxKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer type).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for non-chunked).
    """
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


def write_index_k_chunk(kv_cache: MiniMaxKVCache, tt_index_k, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-norm/post-RoPE MSA index_k (MSA layers only).

    tt_index_k is the single shared index head [1, 1, s_local, head_dim], SP-sharded on the rows and
    REPLICATED across the TP cols (so each col writes the same data into its replicated cache slot).
    """
    _write_one(
        kv_cache.index_k,
        tt_index_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
