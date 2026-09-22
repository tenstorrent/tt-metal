# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 chunked-prefill KV cache. Ported from ``gpt_oss_d_p/tt/attention/kv_cache.py``.

GQA, so **two** caches (``k``, ``v``) per the recipe's §5.2 table. The §5.1 fixed layout is copied
verbatim: user-major slot packing ``slot = user_id * num_layers + layer_idx``, an ``NdShardSpec``
with a ``[1, 1, 32, head_dim]`` shard round-robin over the DRAM bank grid, block-cyclic SP sharding
of the sequence, zeroed ``ReplicateTensorToMesh`` allocation, and
``ttnn.experimental.deepseek_prefill.update_padded_kv_cache`` as the write op.

Two places this model departs from both borrowed packages, each deliberate:

**1. Two KV heads per chip, not one.** §5.1's per-chip shape is written
``[num_users*num_layers, 1, seq_local, head_dim]`` because in every existing GQA package the KV head
count equals TP (GPT-OSS: 8 KV heads on 8 TP cols). Here there are 8 KV heads over TP=4, so a chip
holds **2**, and the per-chip shape is ``[num_users*num_layers, 2, seq_local, head_dim]``. The `1`
in the recipe table is that coincidence, not a constraint — the table's own "varies per model" half
says which heads a chip holds is decided at write time by the mesh mapping. Verified against both
consuming ops before adopting:

  * ``update_padded_kv_cache`` requires ``cache_shape[1] == input_shape[1]`` (any value), and only
    forces ``input.padded_shape()[1] == 1`` when ``tp_factor > 1``, i.e. when the cache is
    TP-deduped via ``tp_axis``. We pass ``cluster_axis=sp_axis`` and no ``tp_axis``, so
    ``tp_factor == 1`` and the multi-head write is in contract.
  * ``ring_joint_scaled_dot_product_attention`` accepts grouped GQA KV
    (``NKH == NVH < NQH && NQH % NKH == 0``); per chip that is ``2 == 2 < 24``, ``24 % 2 == 0``.
    (The ``NKH == 1`` restriction in that file is latent-V / MLA mode only.)

**2. Capacity is rounded up to a whole number of chunks.** The spec's ``max_seq_len`` (262144) is
not a multiple of its ``chunk_size`` (5120) — 51.2 chunks. The block-cyclic write derives its
per-chip offset from ``kv_actual_global`` and the chunk size, so a capacity that is not a whole
number of chunk periods lets the final partial period alias onto the start of the cache. We round
capacity **up** to 266240 tokens (52 chunks), which is >= the spec's max_seq_len and still a
multiple of ``TILE_SIZE * sp``. See :func:`round_cache_capacity`.

There is no sliding-window split here: ``text_config.sliding_window`` is ``null`` for this model, so
every layer is full attention and the GPT-OSS bounded/circular cache machinery is dropped entirely.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard in allocate_kv_cache and the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def round_cache_capacity(max_seq_len: int, chunk_size: int) -> int:
    """Round a requested cache capacity up to a whole number of chunk periods.

    The block-cyclic layout's addressing period is ``chunk_size``; the cache must hold an integral
    number of periods or the last partial one aliases onto the first. Rounding up (rather than down)
    keeps the cache able to hold the full requested ``max_seq_len``.
    """
    assert chunk_size > 0 and max_seq_len > 0
    return -(-int(max_seq_len) // int(chunk_size)) * int(chunk_size)


def cache_row_index(cache_global: int, sp: int, chunk_size: int) -> torch.Tensor:
    """Map each global token position to its row in the row-concatenated host read of the cache.

    Composing a cache tensor with ``ConcatMesh2dToTensor(dims=(2, 1))`` yields
    ``[slots, num_kv_heads, sp * seq_local, head_dim]`` where row ``chip * seq_local + local_row``
    holds whatever chip ``chip`` has at its local row. Indexing that with this table gives the
    cache contents back in natural sequence order, which is what every KV PCC check needs.

    The layout it inverts: ``update_padded_kv_cache`` places a chunk block-cyclically, but at a
    chunk-aligned ``kv_actual`` — the only kind this model writes, since every chunk starts at a
    multiple of ``chunk_size`` — the rotation degenerates to a contiguous split of each chunk
    across the ``sp`` chips. So global position ``p`` in chunk ``p // chunk_size`` lands on chip
    ``(p % chunk_size) // chunk_local`` at that chip's row ``(p // chunk_size) * chunk_local +
    (p % chunk_local)``.
    """
    assert cache_global % chunk_size == 0, (
        f"cache capacity {cache_global} must be a whole number of {chunk_size}-token chunks; "
        f"use round_cache_capacity"
    )
    assert chunk_size % sp == 0, f"chunk_size {chunk_size} must divide evenly over sp={sp}"
    chunk_local = chunk_size // sp
    seq_local = cache_global // sp
    p = torch.arange(cache_global, dtype=torch.long)
    chip = (p % chunk_size) // chunk_local
    local_row = (p // chunk_size) * chunk_local + (p % chunk_local)
    return chip * seq_local + local_row


@dataclass
class MistralKVCache:
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Two persistent device caches, each per-chip shape
    ``[num_users*num_layers, n_kv_local, seq_local, head_dim]``. Heads are sharded on the TP cols at
    write time (2 of the 8 KV heads per chip at TP=4); the sequence is SP-sharded block-cyclic on
    the ``sp`` rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int  # the ROUNDED capacity actually allocated, in global tokens
    sp: int
    n_kv_local: int
    head_dim: int

    def layer_view(self, user_id, layer_idx):
        """Single source of truth for where a (user, layer) lives.

        Returns ``(k, v, batch_idx, capacity_tokens)``. Drive the ops with
        ``slot_idx=batch_idx, layer_idx=0, num_layers=1`` — the kernels linearize
        ``slot*num_layers + layer``, so folding the layer in here keeps one convention.

        Folding the layer into the index is load-bearing on the READ side: passing the bare user
        slot makes every layer read layer 0's cache, which is correct by coincidence at layer 0 and
        silently corrupt from layer 1 on.
        """
        assert 0 <= user_id < self.num_users, f"user_id {user_id} out of range [0, {self.num_users})"
        assert 0 <= layer_idx < self.num_layers, f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
        return self.k, self.v, user_id * self.num_layers + layer_idx, self.max_seq_len


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    chunk_size,
    sp_axis=0,
    tp=4,
    num_users=1,
    num_kv_heads=8,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> MistralKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`MistralKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (MLA-specific, single latent cache): this owns the GQA K/V
    pair and the user-major packing, while reusing the same DRAM NdShard spec so
    ``update_padded_kv_cache`` writes into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 88).
        max_seq_len: requested per-user capacity in tokens. Rounded UP to a whole number of
            ``chunk_size`` periods (see :func:`round_cache_capacity`); the rounded value is what
            lands in ``MistralKVCache.max_seq_len``.
        chunk_size: the block-cyclic addressing period.
        sp_axis: mesh axis the sequence is sharded over (rows).
        tp: tensor-parallel degree — the KV heads are sharded across it, so each chip holds
            ``num_kv_heads // tp`` heads.
        num_users: independent user slots sharing the cache (1 for bring-up).
        num_kv_heads: the model's global KV head count (8).
        head_dim: per-head width (128).
        cache_dtype: on-device cache dtype (bfloat8_b, per the spec's dataformats block).
    """
    sp = mesh_device.shape[sp_axis]

    capacity = round_cache_capacity(max_seq_len, chunk_size)
    # seq_local must be tile-aligned: the cache is TILE_LAYOUT and the DRAM NdShard is 32-token.
    # This also matches build_indexed_rope's chunk alignment so cache + rope layouts agree.
    assert capacity % (ttnn.TILE_SIZE * sp) == 0, (
        f"cache capacity ({capacity}, rounded up from max_seq_len={max_seq_len} to a multiple of "
        f"chunk_size={chunk_size}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); "
        f"seq_local must be tile-aligned"
    )
    assert chunk_size % (ttnn.TILE_SIZE * sp) == 0, (
        f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}) — the "
        f"spec requires this too; a misaligned period corrupts addresses silently"
    )
    seq_local = capacity // sp

    assert num_kv_heads % tp == 0, (
        f"num_kv_heads ({num_kv_heads}) must divide evenly across TP ({tp}); a ragged KV head split "
        f"has no valid mesh mapping"
    )
    n_kv_local = num_kv_heads // tp

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
        # WHICH heads a chip holds is decided at write time by how the input chunk is mesh-mapped,
        # not here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer;
        # content diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, n_kv_local, seq_local, head_dim),
            dtype=cache_dtype,
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
        max_seq_len=capacity,
        sp=sp,
        n_kv_local=n_kv_local,
        head_dim=head_dim,
    )


def _write_one(cache, tensor, *, slot_idx, kv_actual, sp_axis):
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
        layer_idx=0,
        num_layers=1,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if src is not tensor:
        src.deallocate(True)


def write_kv_chunk(kv_cache: MistralKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer, even one-shot).

    ``tt_k`` / ``tt_v`` are the per-device SP shards ``[1, n_kv_local, s_local, head_dim]`` (heads
    TP-sharded on the cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache
    layout, so they write in place. ``kv_actual`` is the cumulative valid prefix BEFORE this chunk
    (0 for the first/only chunk).
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading/batch dim, so a batched tt_k/tt_v would silently write only slot_idx and drop the rest.
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    assert tt_k.shape[1] == kv_cache.n_kv_local and tt_v.shape[1] == kv_cache.n_kv_local, (
        f"chunk carries {tt_k.shape[1]} KV heads per chip but the cache was allocated for "
        f"{kv_cache.n_kv_local}; update_padded_kv_cache requires cache_shape[1] == input_shape[1]"
    )
    # A misaligned chunk offset would make the block-cyclic per-device write land off-tile.
    assert (
        kv_actual % ttnn.TILE_SIZE == 0
    ), f"kv_actual ({kv_actual}) must be tile-aligned (multiple of {ttnn.TILE_SIZE})"

    k_cache, v_cache, batch_idx, _capacity = kv_cache.layer_view(slot_idx, layer_idx)
    for cache, tensor in ((k_cache, tt_k), (v_cache, tt_v)):
        _write_one(cache, tensor, slot_idx=batch_idx, kv_actual=kv_actual, sp_axis=sp_axis)
