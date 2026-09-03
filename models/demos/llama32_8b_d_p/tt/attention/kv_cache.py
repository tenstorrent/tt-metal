# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B chunked-prefill KV cache. **The KV cache is the output of prefill**, so this file's
correctness is the point of the whole package.

Template: ``models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27``
(``NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32``), ``:31`` (the dataclass), ``:48``
(``allocate_kv_cache``), ``:77`` (the tile-alignment assert), ``:80`` (``seq_local``), ``:86-91``
(``NdShardSpec``, ``shard_shape`` at ``:87``), ``:104`` (``ReplicateTensorToMesh`` — content diverges
on the first write), ``:117`` (``_write_one``), ``:125`` (``update_padded_kv_cache``), ``:138``
(``write_kv_chunk``), asserts ``:149`` / ``:155-159``.

Two persistent device caches, each per-chip shape ``[num_users*num_layers, 1, seq_local, head_dim]``
in the DeepSeek chunked-KV DRAM ``NdShard`` layout, written by
``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``. Under
TP=cols each chip holds exactly **one** KV head (8 KV heads over the 8 TP cols at write time); the
sequence is SP-sharded **block-cyclic** on the ``sp`` rows. The batch dim is user-major
(``slot = user_id * num_layers + layer_idx``), matching ``update_padded_kv_cache``'s indexing.

**The one substantive change from the template is ``head_dim`` 64 -> 128** (Appendix F.6). Zero
gpt-oss baggage otherwise: it is already a pure GQA cache, and ``n_kv = 1`` per chip at TP=8 is the
*production-exercised* configuration on this exact hardware. But ``head_dim`` is the one number in
the DRAM shard spec that Llama changes, and it **doubles the shard row**
(``shard_shape=[1, 1, 32, head_dim]``, ``kv_cache.py:87``), so ``G-KV`` runs at the real
``head_dim=128`` and asserts the **block-cyclic read-back and the untouched regions**, not just PCC.

The block geometry is kept exactly, because that is what lets P10 reuse the producer's existing
packed-K/V reader (``BRINGUP_RECIPE.md:711-716``, ``bringup_log/08_PREFILL_INTEGRATION.md``).

``cache_dtype`` defaults to ``bfloat8_b`` (``DEC-017``). This is effectively forced, not chosen:
``models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:77-81`` asserts a bf8_b cache for the chunked
ring path, so ``bfloat16`` is a measurement-only mode that cannot ship.

**K is stored post-RoPE, V raw** — as in every template
(``models/demos/gpt_oss_d_p/tt/attention/prefill.py:162-165`` comment, write at ``:168`` after the
RoPE at ``:151``). P7's golden-KV script must match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard in allocate_kv_cache and the address-table bank walk. Do NOT change:
# P10's producer-side reader assumes this geometry (08_PREFILL_INTEGRATION.md).
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32

# Llama-3.1-8B. gpt-oss is 64; this is the one number the layout changes (Appendix F.6).
LLAMA_HEAD_DIM = 128


@dataclass
class LlamaKVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Two persistent device caches, each per-chip shape
    ``[num_users*num_layers, 1, seq_local, head_dim]``:

      * ``k``, ``v`` — GQA K/V. Under TP=cols each chip holds 1 KV head (heads sharded on the TP
        cols at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    head_dim: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=LLAMA_HEAD_DIM,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`LlamaKVCache`.

    Args:
        num_layers: layers per user (full model = 32). All layers allocate K/V slots.
        max_seq_len: per-user cache capacity in tokens, a multiple of ``TILE_SIZE * sp`` so
            ``seq_local = max_seq_len // sp`` is tile-aligned (matching ``build_indexed_rope``'s
            ``chunk_size % (TILE_SIZE*sp)`` constraint and the 32-token DRAM shard).
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width. **128 for Llama-3.1** (Appendix F.6).
        cache_dtype: on-device cache dtype. ``bfloat8_b`` (``DEC-017``); ``bfloat16`` measures only.
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); "
        f"seq_local must be tile-aligned"
    )
    # head_dim must fill whole tiles along the last dim, or the NdShard row is ragged. 128 = 4 tiles.
    assert (
        head_dim % ttnn.TILE_SIZE == 0
    ), f"head_dim ({head_dim}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE}) for the DRAM NdShard row"
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

    def _alloc():
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds is decided at write time
        # by how the input chunk is mesh-mapped, not here. Allocated zeroed + ReplicateTensorToMesh:
        # every chip gets the same empty buffer; content diverges on the first
        # update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=cache_dtype,
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
        head_dim=head_dim,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via ``update_padded_kv_cache``.

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


def write_kv_chunk(kv_cache: LlamaKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's **post-RoPE K** and **raw V** into the packed cache (every layer).

    ``tt_k`` / ``tt_v`` are the per-device SP shards ``[1, n_kv_local, s_local, head_dim]`` (heads
    TP-sharded on the cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache
    layout, so they write in place. ``kv_actual`` is the cumulative valid prefix **before** this
    chunk (0 for the first/only chunk).
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading/batch dim, so a batched tt_k/tt_v would silently write only slot_idx and drop the rest.
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    assert (
        tt_k.shape[-1] == tt_v.shape[-1] == kv_cache.head_dim
    ), f"chunk head_dim {tt_k.shape[-1]}/{tt_v.shape[-1]} != cache head_dim {kv_cache.head_dim}"
    # Fail loud on a bad slot/layer (an OOB write would silently land in another user's slot) and on
    # a misaligned chunk offset (the block-cyclic per-device write assumes a tile-aligned boundary).
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
