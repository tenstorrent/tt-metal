# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS chunked-prefill KV cache. Mirrors ``minimax_m3/tt/attention/kv_cache.py`` but for GQA
(not MLA): the cache holds head-sharded K and V (no MLA latent, no MSA ``index_k``).

Layout is otherwise identical to M3 / DeepSeek: two persistent device caches, each per-chip shape
``[num_users*num_layers, 1, seq_local, head_dim]`` (the DeepSeek chunked-KV NdShard DRAM layout),
written by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``.
Under TP=cols each chip holds exactly ONE KV head (8 KV heads sharded on the 8 TP cols at write
time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.

The batch dim is user-major: ``slot = user_id * num_layers + layer_idx`` so each user's layers stay
contiguous, matching ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx`` indexing.

**Bounded sliding-window KV (PR1, ``bounded_sliding_kv_cache=True``)**: GPT-OSS alternates sliding
(window 128) and full attention per ``hf_config.layer_types``; on a sliding layer, positions older
than window+halo are never read, so packing them at full ``max_seq_len`` wastes DRAM. With the flag
on, allocation splits into TWO packed caches — full layers keep the full-length slots
(``num_users * n_full`` batch), sliding layers get small circular slots (``num_users * n_slide``
batch, ``sliding_capacity`` tokens = 2 chunk slabs) — remapped per layer via
:func:`build_layer_map` / :meth:`GptOssKVCache.layer_view`. Sliding writes wrap via a host-side
modulo on ``kv_actual`` (see :func:`write_kv_chunk`); the on-device cache-READ of a bounded layer is
NOT supported yet (PR2, a C++ change) and callers must fail loud. Flag off => byte-identical to the
legacy single packed cache.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard in allocate_kv_cache and the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def build_layer_map(layer_types) -> List[Tuple[bool, int, int]]:
    """Per-layer ``(is_sliding, ordinal, n_type)`` remap for the split full/sliding packed caches.

    ``layer_types`` is this rank's slice of ``hf_config.layer_types``: "sliding_attention" marks a
    sliding layer, anything else is full. ``ordinal`` is the layer's index WITHIN its type and
    ``n_type`` the total layers of that type, so a layer's batch slot in its type's cache is
    ``user * n_type + ordinal`` (user-major per type, mirroring the legacy ``user * num_layers +
    layer`` packing). Pure host math — unit-tested in tests/unit/test_bounded_kv_math.py.
    """
    flags = [t == "sliding_attention" for t in layer_types]
    n_slide = sum(flags)
    n_full = len(flags) - n_slide
    counts = {True: 0, False: 0}
    layer_map = []
    for is_sliding in flags:
        ordinal = counts[is_sliding]
        counts[is_sliding] += 1
        layer_map.append((is_sliding, ordinal, n_slide if is_sliding else n_full))
    return layer_map


def sliding_capacity_tokens(max_seq_len, chunk_size, sp, sliding_window=128) -> int:
    """Global token capacity of one bounded sliding-layer cache slot.

    Chunked configs (``max_seq_len > chunk_size``): 2 chunk slabs — the chunk being written plus the
    previous one, which always holds the whole window+halo a sliding read needs (``sliding_window <=
    chunk_size`` is asserted). ``capacity % chunk_global == 0`` by construction, which is what lets
    :func:`write_kv_chunk`'s host-side modulo land each chunk exactly on a slab.

    One-shot (``max_seq_len == chunk_size``) would take ``capacity = sp * align32(window + 128)``
    slack with ``cap_local`` dividing ``chunk_local`` — but ``update_padded_kv_cache`` cannot wrap
    WITHIN a single chunk (that is the C++ ``wrap_seq`` change, not in PR1), so one-shot + bounded
    is rejected here.
    """
    assert chunk_size is not None and chunk_size > 0, "bounded_sliding_kv_cache needs a chunk_size"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    if max_seq_len == chunk_size:
        raise NotImplementedError(
            "bounded_sliding_kv_cache is unsupported for one-shot prefill (max_seq_len == chunk_size): "
            "the circular write would have to wrap WITHIN the single chunk, which needs the C++ "
            "update_padded_kv_cache wrap_seq change (not in PR1). Run chunked (max_seq_len > "
            "chunk_size) or leave bounded_sliding_kv_cache off."
        )
    assert sliding_window <= chunk_size, (
        f"sliding_window ({sliding_window}) must fit one chunk slab ({chunk_size}) so the previous "
        f"slab always holds the whole window"
    )
    del sp  # kept for the one-shot capacity formula when wrap_seq lands
    return 2 * chunk_size


def bounded_blockcyclic_positions(sp, chunk_size_global, capacity_global, written_tokens) -> torch.Tensor:
    """Bounded-cache inverse of the block-cyclic writer: the global natural position held by each
    shard row (device-major, ``[sp * cap_local]``), ``-1`` for rows never written.

    The circular write puts chunk group ``g`` into slab ``j = g mod m`` (``m = capacity / chunk``
    slabs), so cache row (chip ``c``, slab ``j``, offset ``o``) holds global position
    ``g*C + c*chunk_local + o`` where ``g`` is the LARGEST chunk-group index with ``g mod m == j``
    among the ``G = ceil(written_tokens / C)`` groups written so far. The unbounded layout is the
    ``G <= m`` degenerate case (matches ``blockcyclic_positions`` on the written rows). Pure host
    math — unit-tested in tests/unit/test_bounded_kv_math.py.
    """
    assert (
        capacity_global % chunk_size_global == 0
    ), f"capacity ({capacity_global}) must be a multiple of chunk_global ({chunk_size_global})"
    assert chunk_size_global % sp == 0 and capacity_global % sp == 0, "chunk/capacity must SP-shard evenly"
    chunk_local = chunk_size_global // sp
    cap_local = capacity_global // sp
    m = capacity_global // chunk_size_global
    num_groups = -(-int(written_tokens) // chunk_size_global)  # whole (padded) chunks written
    c = torch.arange(sp).repeat_interleave(cap_local)
    lr = torch.arange(cap_local).repeat(sp)
    slab, off = lr // chunk_local, lr % chunk_local
    g = slab + m * torch.div(num_groups - 1 - slab, m, rounding_mode="floor")
    pos = g * chunk_size_global + c * chunk_local + off
    pos[slab >= num_groups] = -1  # this slab has never been written
    return pos


@dataclass
class GptOssKVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Two persistent device caches, each per-chip shape ``[num_users*num_layers, 1, seq_local, head_dim]``:

      * ``k``, ``v`` — GQA K/V. Under TP=cols each chip holds 1 KV head (heads sharded on the TP cols
        at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.

    With ``bounded_sliding=True`` (PR1) the packing splits per layer TYPE: ``k``/``v`` hold only the
    FULL-attention layers (``num_users * n_full`` slots, full ``max_seq_len``) and ``k_sliding``/
    ``v_sliding`` the sliding layers (``num_users * n_slide`` slots, ``sliding_capacity`` tokens,
    written circularly). :meth:`layer_view` is the single source of truth for where a (user, layer)
    lives; all fields below it are ``None``/``False`` when the flag is off (legacy layout).
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    # --- bounded sliding-window split (PR1); None/False when the flag is off ---
    k_sliding: Optional[ttnn.Tensor] = None
    v_sliding: Optional[ttnn.Tensor] = None
    sliding_capacity: Optional[int] = None  # global tokens per sliding slot (m slabs of chunk_size)
    layer_map: Optional[List[Tuple[bool, int, int]]] = None  # per-layer (is_sliding, ordinal, n_type)
    bounded_sliding: bool = False

    def layer_view(self, user_id, layer_idx):
        """Single source of truth for where a (user, layer) lives.

        Returns ``(k, v, batch_idx, capacity_tokens, bounded)``: the cache tensor pair, the flat
        batch slot inside it (drive ops with ``slot_idx=batch_idx, layer_idx=0, num_layers=1`` — the
        kernels just linearize ``slot*num_layers + layer``), that slot's token capacity, and whether
        it is a bounded (circular) sliding cache. Flag off => the legacy view: the single packed
        all-layers cache at ``slot = user * num_layers + layer`` with ``max_seq_len`` capacity.
        """
        if not self.bounded_sliding:
            return self.k, self.v, user_id * self.num_layers + layer_idx, self.max_seq_len, False
        is_sliding, ordinal, n_type = self.layer_map[layer_idx]
        batch_idx = user_id * n_type + ordinal
        if is_sliding:
            return self.k_sliding, self.v_sliding, batch_idx, self.sliding_capacity, True
        return self.k, self.v, batch_idx, self.max_seq_len, False


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=64,
    cache_dtype=ttnn.bfloat8_b,
    layer_types=None,
    bounded_sliding_kv_cache=False,
    chunk_size=None,
    sliding_window=128,
) -> GptOssKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`GptOssKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (that is MLA-specific and allocates a single latent cache):
    this owns the GQA K/V pair and the user-major packing. It reuses the same DRAM NdShard spec (same
    bank grid + 32-token contiguous shard) so ``update_padded_kv_cache`` can write into these tensors
    unchanged.

    Args:
        num_layers: layers per user (full model = 36). All layers allocate K/V slots.
        max_seq_len: per-user cache capacity in tokens, a multiple of ``TILE_SIZE * sp`` so ``seq_local =
            max_seq_len // sp`` is tile-aligned (matches build_indexed_rope + the 32-token DRAM shard).
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (64 for gpt-oss).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate + the device golden check).
        layer_types: this rank's slice of ``hf_config.layer_types`` (required when bounded); ignored
            when the flag is off.
        bounded_sliding_kv_cache: split sliding layers into a small circular cache (PR1). Off =>
            byte-identical to the legacy single packed cache.
        sliding_window: the sliding layers' window size (``hf_config.sliding_window``, 128 for gpt-oss).
        chunk_size: prefill chunk size in tokens (required when bounded — sizes the circular slabs).
    """
    sp = mesh_device.shape[sp_axis]
    # seq_local must be tile-aligned: the cache is TILE_LAYOUT and the DRAM NdShard is 32-token; also
    # matches build_indexed_rope's chunk_size % (TILE_SIZE*sp) constraint so cache + rope layouts agree.
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

    def _alloc(batch_slots, slot_seq_local, dtype=cache_dtype):
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds is decided at write time by
        # how the input chunk is mesh-mapped, not here. Allocated zeroed + ReplicateTensorToMesh: every
        # chip gets the same empty buffer; content diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(batch_slots, 1, slot_seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    if not bounded_sliding_kv_cache:
        # Legacy layout: ONE packed all-layers cache pair. layer_types / chunk_size / sliding_window
        # are deliberately ignored so the flag-off allocation stays byte-identical to before.
        return GptOssKVCache(
            k=_alloc(num_users * num_layers, seq_local),
            v=_alloc(num_users * num_layers, seq_local),
            num_users=num_users,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            sp=sp,
        )

    # --- bounded sliding split (PR1) ---
    assert layer_types is not None, "bounded_sliding_kv_cache=True needs hf_config.layer_types"
    assert len(layer_types) >= num_layers, (
        f"layer_types has {len(layer_types)} entries but num_layers={num_layers}; pass this rank's "
        f"slice (layer_types[first_layer_idx : first_layer_idx + num_layers])"
    )
    layer_map = build_layer_map(list(layer_types)[:num_layers])
    n_slide = sum(1 for is_sliding, _, _ in layer_map if is_sliding)
    n_full = num_layers - n_slide
    # Raises NotImplementedError on one-shot configs (needs the C++ wrap_seq change, not in PR1).
    capacity = sliding_capacity_tokens(max_seq_len, chunk_size, sp, sliding_window)
    assert (
        capacity % (ttnn.TILE_SIZE * sp) == 0
    ), f"sliding capacity ({capacity}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp})"
    cap_local = capacity // sp
    logger.info(
        f"bounded_sliding_kv_cache: {n_full} full layers @ {max_seq_len} tok + {n_slide} sliding "
        f"layers @ {capacity} tok (window {sliding_window}, {capacity // chunk_size} slabs of "
        f"{chunk_size}); num_users={num_users}"
    )
    return GptOssKVCache(
        k=_alloc(num_users * n_full, seq_local) if n_full > 0 else None,
        v=_alloc(num_users * n_full, seq_local) if n_full > 0 else None,
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        k_sliding=_alloc(num_users * n_slide, cap_local) if n_slide > 0 else None,
        v_sliding=_alloc(num_users * n_slide, cap_local) if n_slide > 0 else None,
        sliding_capacity=capacity,
        layer_map=layer_map,
        bounded_sliding=True,
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


def write_kv_chunk(kv_cache: GptOssKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for the first/only chunk).

    Bounded sliding layers (``kv_cache.bounded_sliding`` + a sliding ``layer_idx``) write CIRCULARLY
    via a host-side modulo: the writer kernel derives its per-chip offset from ``kv_actual_global``
    and the chunk's OWN size — never the cache capacity — so with ``capacity = m * chunk_global`` and
    chunk-aligned ``kv_actual``, passing ``kv_actual mod capacity`` lands the chunk exactly on slab
    ``(kv_actual / chunk_global) mod m``, overwriting the oldest slab. No C++ change needed.
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading/batch dim, so a batched (batch>1) tt_k/tt_v would silently write only slot_idx and drop the
    # rest. Require batch==1 and fail loud; batched multi-user prefill must loop this per user (slot_idx+b)
    # at the call site. The scatter-to-slots write lands with the runtime multi-user path (P4).
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    # Fail loud on a bad slot/layer (would otherwise be a silent OOB write into another user's slot) and
    # on a misaligned chunk offset (the block-cyclic per-device write assumes a tile-aligned boundary).
    assert 0 <= slot_idx < kv_cache.num_users, f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})"
    assert 0 <= layer_idx < kv_cache.num_layers, f"layer_idx {layer_idx} out of range [0, {kv_cache.num_layers})"
    assert (
        kv_actual % ttnn.TILE_SIZE == 0
    ), f"kv_actual ({kv_actual}) must be tile-aligned (multiple of {ttnn.TILE_SIZE})"
    if not kv_cache.bounded_sliding:
        # Legacy packed cache: identical op arguments to before the bounded split existed.
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
        return
    # Bounded split: route through layer_view (single source of truth) and drive the op with the flat
    # batch slot (slot_idx=batch_idx, layer_idx=0, num_layers=1 — the kernel just linearizes
    # slot*num_layers + layer, and validates slot against cache_batch/num_layers).
    k_cache, v_cache, batch_idx, capacity_tokens, bounded = kv_cache.layer_view(slot_idx, layer_idx)
    if bounded:
        chunk_global = tt_k.shape[-2] * kv_cache.sp
        # Both MUST hold or the modulo write is not slab-exact: the writer's offset math would step
        # past this slot's cap_local rows and corrupt the NEXT batch slot's rows. (When variable
        # chunk sizes / extra_chunk_sizes land, guard each non-default size cs here the same way:
        # capacity % (cs_global) == 0 — this assert already enforces it per write.)
        assert capacity_tokens % chunk_global == 0, (
            f"bounded sliding write needs capacity ({capacity_tokens}) to be a multiple of this "
            f"chunk's global size ({chunk_global})"
        )
        assert kv_actual % chunk_global == 0, (
            f"bounded sliding write needs a chunk-aligned kv_actual ({kv_actual}); " f"chunk_global={chunk_global}"
        )
        kv_actual = kv_actual % capacity_tokens
    for cache, tensor in ((k_cache, tt_k), (v_cache, tt_v)):
        _write_one(
            cache,
            tensor,
            slot_idx=batch_idx,
            layer_idx=0,
            num_layers=1,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
