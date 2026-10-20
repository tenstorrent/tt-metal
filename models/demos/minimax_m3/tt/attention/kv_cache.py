# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

# DRAM ND-shard geometry for the packed prefill KV cache — M3-local (decoupled from the DeepSeek substrate
# so they can diverge). The sequence is tiled into NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK-token blocks that
# round-robin across BH_NUM_DRAM_BANKS DRAM banks. The address-table builder must match both values.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
BH_NUM_DRAM_BANKS = 8

# Page-table entry of a logical page with no physical page (the ops' paged_kv_table.hpp).
PAGE_UNMAPPED = -1
# Tokens per page of a paged cache. A whole number of 32-row tiles that divides a chunk's per-chip run
# (chunk_size / sp), so a page never straddles two SP ranks.
PAGE_SIZE = 64


@dataclass
class MiniMaxKVCache(KvCaches):
    """M3's on-device prefill KV cache: three persistent, user-major packed device caches, each per-chip
    shape ``[num_users*num_layers, 1, seq_local, head_dim]`` on the DRAM ND-shard substrate, written in
    place by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``:

      * ``k`` / ``v``  — GQA K/V. Under TP=cols each chip holds one head (heads sharded on the TP cols);
                         the sequence is SP-sharded block-cyclic on the ``sp`` rows.
      * ``index_k``    — MSA lightning-indexer key (one shared head); only the MSA layers populate it —
                         dense layers leave their slots zeroed. REPLICATED across the TP cols by default;
                         with ``index_k_tp_axis`` set (KV dedup, ``M3_INDEX_K_TP_SHARD=1``) it is striped
                         across all sp*tp chips instead: chip (s, t) holds tokens
                         ``[n*chunk + s*chunk_local + t*chunk_local/tp, +chunk_local/tp)`` of every chunk n,
                         so its per-chip shape is ``[.., 1, seq_local/tp, ..]``.

    Batch dim is slot-major (``cache_row = slot_id * num_layers + layer_idx``) so each slot's layers stay
    contiguous, matching ``update_padded_kv_cache``'s indexing. The adapter allocates this once and the
    engine holds it as an opaque handle, passing it back into every runtime call that touches it.

    Paged (``page_table`` set): each cache is a pool of ``num_pages`` pages per chip,
    ``[num_pages*num_layers, 1, PAGE_SIZE, head_dim]`` with entry ``page * num_layers + layer``, so one page id
    holds that page of every layer. ``page_table`` ``[num_users, width]`` (INT32, DRAM, replicated) maps row
    ``slot_id``'s global logical page q (positions ``[q*PAGE_SIZE, (q+1)*PAGE_SIZE)``) to a physical page in the
    pool of the SP rank that holds it (``page_owner``), or ``PAGE_UNMAPPED``. Before a request's first chunk the
    caller sets its whole row: the request's own pages, PAGE_UNMAPPED everywhere else. A final chunk writes its
    pad rows too, and the writer drops only those on unmapped pages, so a stale entry left in the row would put
    them into another request's page. Not paged: ring_joint reads (a stage holding a dense layer, M3's layers
    0-2, keeps slot caches), index_k TP dedup, KV migration.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    index_k: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    index_k_tp_axis: Optional[int] = None  # None = index_k TP-replicated; else the mesh axis it is deduped over
    page_table: Optional[ttnn.Tensor] = None  # None = slot cache
    num_pages: int = 0  # pages in each chip's pool when paged

    def paging_kwargs(self) -> dict:
        """The page-table argument of the KV write and gather ops; empty for a slot cache."""
        return {} if self.page_table is None else {"page_table": self.page_table}

    def gather_source(self, cache_row: int, chunk_local: int) -> dict:
        """high_bw_all_gather arguments that read ``cache_row`` (slot_id * num_layers + layer) of this cache.
        A paged read passes the same flat row, which the op splits into the page-table row and the layer with the
        pool's layer count, plus the block-cyclic period (``chunk_local`` rows per rank) that places pages on
        ranks."""
        if self.page_table is None:
            return {"input_batch_index": cache_row}
        return {
            "input_batch_index": cache_row,
            "batch_slot_num_layers": self.num_layers,
            "gathered_slab_global": chunk_local * self.sp,
            **self.paging_kwargs(),
        }

    def deallocate(self) -> None:
        """Free the device caches (e.g. to re-allocate at a different ``max_seq_len`` while the
        model stays resident). The handle is dead afterwards; do not pass it into the runtime again."""
        for t in (self.k, self.v, self.index_k, self.page_table):
            if t is not None:
                ttnn.deallocate(t)


def page_owner(q, *, chunk_local, sp):
    """SP rank holding global logical page ``q``: positions are block-cyclic over the ranks in runs of
    ``chunk_local``, and a page never straddles two runs (``chunk_local % PAGE_SIZE == 0``)."""
    return (q * PAGE_SIZE // chunk_local) % sp


def page_table_width(max_seq_len: int) -> int:
    """Entries per page-table row: the longest request's pages."""
    return -(-max_seq_len // PAGE_SIZE)


def set_page_table(kv_cache: MiniMaxKVCache, table: torch.Tensor) -> None:
    """Overwrite the whole device page table, in place, with a host ``[num_users, width]`` table."""
    mesh_device = kv_cache.page_table.device()
    host = ttnn.from_torch(
        table.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host, kv_cache.page_table)


def read_paged_slot_block(kv_cache: MiniMaxKVCache, tensor, slot_id: int, *, rows: int, chunk_size: int):
    """Host copy of ``slot_id``'s first ``rows`` local rows per chip of one paged cache tensor, in the slot-cache
    layout ``[num_layers, heads, sp*rows, head_dim]`` (SP ranks along dim 2, TP cols along dim 1); unmapped pages
    read as zeros. The slot's pages are read in runs of consecutive physical pages, so the copy is bounded by the
    pages it maps. Rank s's local page i holds global page ``q = ((i // per_run) * sp + s) * per_run + i % per_run``,
    ``per_run`` pages per rank per chunk."""
    mesh_device = tensor.device()
    sp, n_layers = kv_cache.sp, kv_cache.num_layers
    per_run = chunk_size // sp // PAGE_SIZE
    table = ttnn.to_torch(ttnn.get_device_tensors(kv_cache.page_table)[0]).to(torch.int64)
    i = torch.arange(rows // PAGE_SIZE)
    q = ((i // per_run)[None, :] * sp + torch.arange(sp)[:, None]) * per_run + (i % per_run)[None, :]
    phys = table[slot_id][q]  # [sp, local pages]
    mapped = (phys >= 0) & (phys < kv_cache.num_pages)
    shape = list(tensor.shape)
    out = torch.zeros(n_layers, shape[1] * mesh_device.shape[1], sp * rows, shape[3])
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    pages = sorted(set(phys[mapped].tolist()))
    host = {}  # physical page -> its entries [n_layers, tp, sp*page, hd]
    while pages:
        lo = hi = pages.pop(0)
        while pages and pages[0] == hi + 1:
            hi = pages.pop(0)
        sl = ttnn.slice(
            tensor, [lo * n_layers, 0, 0, 0], [(hi + 1) * n_layers, *shape[1:]], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        run = ttnn.to_torch(sl, mesh_composer=composer).float()
        ttnn.deallocate(sl)
        for p in range(lo, hi + 1):
            host[p] = run[(p - lo) * n_layers : (p - lo + 1) * n_layers]
    for rank, local in mapped.nonzero().tolist():
        dst = rank * rows + local * PAGE_SIZE
        out[:, :, dst : dst + PAGE_SIZE] = host[int(phys[rank, local])][:, :, rank * PAGE_SIZE : (rank + 1) * PAGE_SIZE]
    return out


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
    index_k_tp_shard=None,
    num_pages=None,
) -> MiniMaxKVCache:
    """Allocate the three external prefill KV caches (K, V, index_k), and with ``num_pages`` their page table. See
    :class:`MiniMaxKVCache`.

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
        index_k_tp_shard: KV dedup — stripe index_k across the TP cols instead of replicating it (4x less
            index_k memory at TP=4). None reads ``M3_INDEX_K_TP_SHARD=1``. K / V are unaffected.
        num_pages: allocate a paged cache instead: a pool of this many ``PAGE_SIZE``-token pages per chip, and
            a ``num_users``-row page table, all unmapped. ``max_seq_len`` is then the longest request, and
            every chunk's per-chip run (chunk_size / sp) must be whole pages.
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
    seq_local = max_seq_len // sp
    if index_k_tp_shard is None:
        index_k_tp_shard = os.getenv("M3_INDEX_K_TP_SHARD") == "1"
    index_k_tp_axis = 1 - sp_axis if index_k_tp_shard else None
    paged = num_pages is not None
    if paged:
        assert index_k_tp_axis is None, "a paged KV cache does not support index_k TP dedup"
        # The gather lands rank r's prefix at r*seq_local in whole pages.
        assert seq_local % PAGE_SIZE == 0, f"max_seq_len/sp ({seq_local}) must be whole {PAGE_SIZE}-token pages"
    if index_k_tp_axis is not None:
        # The cache-read gathers TP-inner then SP-outer (or one row-major snake), so chip (s, t) lands at
        # (s*tp + t)*rows — the sp*tp linearization only holds with SP on the mesh rows.
        assert sp_axis == 0, f"index_k TP dedup needs sp_axis == 0 (got {sp_axis})"
        tp = mesh_device.shape[index_k_tp_axis]
        assert (
            seq_local % (tp * ttnn.TILE_SIZE) == 0
        ), f"index_k TP dedup needs seq_local ({seq_local}) divisible by tp ({tp}) into whole tiles"

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    # The topology ReplicateTensorToMesh would give: 1-D Replicate over the physical 2-D coordinates.
    replicated = ttnn.TensorTopology(
        ttnn.MeshShape([mesh_device.shape[0] * mesh_device.shape[1]]),
        [ttnn.PlacementReplicate()],
        [
            ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
            for coord in ttnn.MeshCoordinateRange(ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]))
        ],
    )

    entries, cache_rows = (num_pages * num_layers, PAGE_SIZE) if paged else (num_users * num_layers, seq_local)

    def _alloc(dtype=cache_dtype, rows=cache_rows):
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds (or whether index_k is
        # replicated across cols) is decided at write time by how the input chunk is mesh-mapped, not
        # here. Every chip gets the same zeroed buffer, allocated and zeroed on device: a host tensor of
        # the full cache does not fit in host RAM at production slot counts.
        cache = ttnn.allocate_tensor_on_device(
            ttnn.Shape([entries, 1, rows, head_dim]), dtype, ttnn.TILE_LAYOUT, mesh_device, mem_config
        )
        DRAMZeroFill.op(cache)
        cache.update_tensor_topology(replicated)
        return cache

    # index_k feeds the indexer's HARD top-16 block selection (not a smooth softmax like K/V), so bf8's
    # ~2-3 mantissa bits perturb the block scores enough to flip many picks -> chunked vs one-shot
    # selection diverges (~7/16 overlap) -> residual drift compounding over MSA layers. Cache it in bf16
    # (M3_INDEX_CACHE_BF16=1) to keep selection stable; it's tiny (1 head) and only the indexer reads it.
    index_dtype = ttnn.bfloat16 if os.getenv("M3_INDEX_CACHE_BF16") == "1" else cache_dtype

    if index_k_tp_axis is None:
        index_k = _alloc(index_dtype)
    else:
        index_k = _alloc(index_dtype, rows=seq_local // mesh_device.shape[index_k_tp_axis])
        # Declare the real distribution (dim 2 sharded over both mesh axes, row-major = the sp*tp
        # linearization), as DeepSeek's init_kvpe_cache does for its deduped cache: high_bw_all_gather
        # validates cluster_axis against the declared rank, and a 1-D Replicate topology rejects the TP leg.
        dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
        coords = [
            ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
            for coord in ttnn.MeshCoordinateRange(dist_shape)
        ]
        index_k.update_tensor_topology(
            ttnn.TensorTopology(dist_shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
        )

    page_table = None
    if paged:
        page_table = ttnn.from_torch(
            torch.full((num_users, page_table_width(max_seq_len)), PAGE_UNMAPPED, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return MiniMaxKVCache(
        k=_alloc(),
        v=_alloc(),
        index_k=index_k,
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        index_k_tp_axis=index_k_tp_axis,
        page_table=page_table,
        num_pages=num_pages or 0,
    )


def _write_one(cache, tensor, *, slot_id, layer_idx, num_layers, kv_actual, sp_axis, paging, tp_axis=None):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    ``tp_axis`` (KV dedup): ``tensor`` is TP-replicated and each chip persists only its own 1/tp window
    of its SP shard (the op picks the window from its coordinate on ``tp_axis``).

    The op requires TILE layout and input.dtype == cache.dtype, so cast a copy to the cache's dtype when
    needed (the original stays live for the attention op that follows). At ``kv_actual % 32 == 0`` chunk
    boundaries the per-device write offset is contiguous (block-cyclic degenerates to a reshape).
    """
    src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        src,
        slot_idx=slot_id,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
        tp_axis=tp_axis,
        **paging,
    )
    if src is not tensor:
        src.deallocate(True)


def write_kv_chunk(kv_cache: MiniMaxKVCache, tt_k, tt_v, *, slot_id, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer type).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for non-chunked).
    """
    _write_one(
        kv_cache.k,
        tt_k,
        slot_id=slot_id,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
        paging=kv_cache.paging_kwargs(),
    )
    _write_one(
        kv_cache.v,
        tt_v,
        slot_id=slot_id,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
        paging=kv_cache.paging_kwargs(),
    )


def write_index_k_chunk(kv_cache: MiniMaxKVCache, tt_index_k, *, slot_id, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-norm/post-RoPE MSA index_k (MSA layers only).

    tt_index_k is the single shared index head [1, 1, s_local, head_dim], SP-sharded on the rows and
    REPLICATED across the TP cols. A TP-replicated cache gets the same data on every col; a TP-deduped one
    (``kv_cache.index_k_tp_axis``) keeps only col t's ``s_local/tp`` rows on col t.
    """
    _write_one(
        kv_cache.index_k,
        tt_index_k,
        slot_id=slot_id,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
        paging=kv_cache.paging_kwargs(),
        tp_axis=kv_cache.index_k_tp_axis,
    )
