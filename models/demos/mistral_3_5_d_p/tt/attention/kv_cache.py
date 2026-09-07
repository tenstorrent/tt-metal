# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 chunked-prefill KV cache — the ``kv_cache`` coupled cluster's
``layout_and_allocation`` + ``write_op`` roles.

Donor package: ``gpt_oss_d_p`` for ALL SEVEN roles (the cluster must come from one package; sourcing
the allocator from one and the ring SDPA from another produces a silently broken model). gpt-oss is
the only GQA two-cache prefill package: ``minimax_m3`` allocates three (k, v, index_k for MSA) and
``deepseek_v3_d_p`` allocates one MLA latent — both the wrong cache count for dense GQA.

=====================================================================================
THE LAYOUT (recipe §5). Fixed by convention — do NOT invent it.
=====================================================================================
Read by both the chunked ring SDPA (``dense_sp.py``) and the migration address walk
(``runners/kv_chunk_table.py``), so a deviation degrades PCC or corrupts addresses rather than
raising. Copied verbatim from the donor:

  per-chip shape        [num_users * num_layers, 1, seq_local, head_dim]
  slot packing          slot = user_id * num_layers + layer_idx   (user-major, layers contiguous)
  DRAM memory config    NdShardSpec, shard [1, 1, 32, head_dim], ROUND_ROBIN_1D over the bank grid
  contiguous tokens     NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
  sequence sharding     SP-sharded block-cyclic on the SP axis; seq_local = capacity // sp
  alignment             capacity % (TILE_SIZE * sp) == 0
  allocation            zeroed, ReplicateTensorToMesh (content diverges on the first write)
  write op              ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ..)
  bank count            get_num_dram_banks(mesh_device)   (NOT a hardcoded 8)

=====================================================================================
THE FOUR PER-MODEL DECISIONS (recipe §5.2)
=====================================================================================
1. **Number of cache tensors: 2 — ``k`` and ``v``.** Mistral-Medium-3.5 is dense GQA: one K and one
   V row per token per KV head. Not MLA (no single ``kvpe`` latent) and not sparse (no ``index_k``
   indexer cache). ``init_kvpe_cache`` is deliberately NOT called — it is MLA-specific and allocates
   a single latent — but its NdShard spec is reused verbatim so ``update_padded_kv_cache`` writes
   into these tensors unchanged.
2. **head_dim: 128**, the model's head dim as-is (``config.text_config.head_dim``). Unlike MLA there
   is no latent width to reconstruct. This is the one dimension that changes from the donor's 64.
3. **cache_dtype: bfloat8_b**, from the BINDING spec (``dataformats.kv_cache.default``). The ring
   cache-read path asserts on it — ``dense_sp`` fails loud rather than silently degrading.
4. **Auxiliary caches: none.** Nothing outside K/V needs cache state: no sinks (not a cache), no
   sliding-window halo buffer (the CCL manager owns the ring scratch), no indexer.

Not a decision: **which KV head a chip holds.** That is set at WRITE time by how the chunk is
mesh-mapped — every chip is allocated the same zeroed buffer. At TP=8 with 8 KV heads, column ``c``
holds head ``c``: exactly the donor's 1-KV-head-per-TP-column layout, unchanged, which is why
``num_kv_heads`` needed no adaptation and the slot formula, the 32-token bank walk, ROUND_ROBIN_1D
and the SP block-cyclic sequence all carry over as they are.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C

# Must match the DRAM NdShard in allocate_kv_cache and the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class MistralKVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Two persistent device caches, each per-chip shape
    ``[num_users*num_layers, 1, seq_local, head_dim]``:

      * ``k``, ``v`` — dense GQA K/V. Under TP=cols each chip holds 1 KV head (heads sharded on the
        TP cols at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.

    ``max_seq_len`` here is the ALLOCATED capacity in tokens (``spec.cache_capacity``), which the
    ring SDPA passes as ``cache_global``; the servable context is the spec's ``max_seq_len`` and the
    runtime asserts chunks stay inside it.
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
    head_dim=C.HEAD_DIM,
    cache_dtype=None,
) -> MistralKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`MistralKVCache`.

    Args:
        num_layers: layers per user (full model = 88). Every layer allocates K/V slots.
        max_seq_len: per-user cache CAPACITY in tokens — a multiple of ``TILE_SIZE * sp`` so
            ``seq_local = max_seq_len // sp`` is tile-aligned, and a whole number of chunks so the
            block-cyclic rope tiles it. Pass ``spec.cache_capacity``.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache.
        head_dim: per-head width (128 for Mistral-Medium-3.5).
        cache_dtype: on-device cache dtype; defaults to the spec's ``dataformats.kv_cache.default``.
    """
    from models.demos.mistral_3_5_d_p.spec import SPEC

    cache_dtype = SPEC.kv_cache_dtype if cache_dtype is None else cache_dtype
    sp = mesh_device.shape[sp_axis]
    # seq_local must be tile-aligned: the cache is TILE_LAYOUT and the DRAM NdShard is 32-token; this
    # is the same constraint build_indexed_rope carries, so cache + rope layouts agree by construction.
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
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc(dtype=cache_dtype):
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds is decided at write time
        # by how the input chunk is mesh-mapped, not here. Allocated zeroed +
        # ReplicateTensorToMesh: every chip gets the same empty buffer; content diverges on the
        # first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return MistralKVCache(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    The op requires TILE layout and ``input.dtype == cache.dtype``, so cast a copy to the cache's
    dtype when needed (the original stays live for the attention op that follows). At
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


def write_kv_chunk(kv_cache: MistralKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer).

    ``tt_k`` / ``tt_v`` are the per-device SP shards ``[1, n_kv_local, s_local, head_dim]`` (heads
    TP-sharded on the cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache
    layout, so they write in place. ``kv_actual`` is the cumulative valid prefix before this chunk
    (0 for the first/only chunk).
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading/batch dim, so a batched tt_k/tt_v would silently write only slot_idx and drop the rest.
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
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
