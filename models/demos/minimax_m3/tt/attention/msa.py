# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 MSA (sparse) attention — the real model forward for the sparse layers (3-59).

Unlike the dense path (ring_joint, which reads the KV cache + gathers across SP *internally*),
``sparse_sdpa_msa`` is a pure dense-context kernel: it takes full-length K/V tensors and has no
cache-read. So the cache + cross-device gather live in THIS wrapper:

  each SP device holds a sequence shard of K / V / index_k (from the chunked-KV cache read).
  We AllGather those across the SP axis so every device materialises the full context, then:

    indexer_score_msa(index_q, index_k_full, chunk_start_idx=cached_len)  -> block scores
    topk_large_indices(topk_blocks)                                      -> block-ids   (the op
                                              already force-locals the current block, +inf)
    sparse_sdpa_msa(q, k_full, v_full, block-ids)                        -> attention out

SP sharding is CONTIGUOUS, no zigzag/balancing (per the op authors: chunked prefill needs no causal
load-balancing — MSA work per query is a fixed top-k, not the dense causal triangle).

``chunk_start_idx`` is the global position of query row 0. Because the indexer scores over the
gathered full context and the current chunk's queries all begin at ``cached_len``, this scalar is
uniform across SP devices (no per-device offset needed). Causality is encoded entirely by the block
selection; sparse_sdpa_msa applies no token mask.

Entry points: ``msa_sp_attention_nocache`` (a first chunk: gather the chunk's own K/V), ``msa_sp_attention_cache_read``
(a later chunk: gather its cache slot) and, for packed forwards, ``msa_chunks_pipelined`` (several chunks, the next
one's gather overlapped under this one's attention on two sub-devices; ``MSA_CHUNK_OVERLAP`` and
``msa_overlap_chunks`` decide when it runs, and only sub-device-confined ops may run inside its window).
"""

from dataclasses import dataclass

from loguru import logger

import ttnn
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone

from .operations import apply_qk_norm_per_head, apply_rope

# sparse_sdpa_msa's per-core L1 block cache for the gathered K/V (as many slots as fit in free L1). On: at small SP
# consecutive queries on a core re-select mostly the same blocks, so a resident block saves a DRAM re-read. Output
# is identical either way. Not the model's DRAM KV cache.
MSA_KV_BLOCK_CACHE = True

# Packed forwards (TtPrefillRuntime.prefill_forward, tt/prefill_chunk.py): gather the next warm chunk's K/V/index_k
# prefix on a strip of cores while this chunk's block selection and sparse_sdpa_msa run on the rest
# (CCLManager.get_msa_overlap). A warm chunk has cached_len > 0: it attends a cache prefix and needs the gather; a
# cold one (cached_len == 0) attends only itself. Output is identical to the serial chain. Needs at least two warm
# chunks in the forward. Eager only: the device-side join is an event record, which trace capture rejects.
MSA_CHUNK_OVERLAP = True


def _ensure_dram(t):
    """high_bw_all_gather streams its source from DRAM; move an L1-resident activation there first."""
    if t.memory_config().buffer_type != ttnn.BufferType.DRAM:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _overlap_kwargs(overlap):
    """high_bw_all_gather kwargs that confine a gather to ``overlap``'s gather sub-device with its persistent
    semaphores (an MsaOverlap, see tt/ccl.py); {} outside a window."""
    if overlap is None:
        return {}
    return dict(
        subdevice_id=overlap.gather_sd_id,
        sub_core_grids=overlap.gather_cores,
        ready_semaphore=overlap.ready_semaphore,
        data_valid_semaphore=overlap.data_valid_semaphore,
    )


def high_bw_sp_gather(
    t, mesh_config, ccl_manager, out_buf, *, input_batch_index=None, gathered_dim_size=None, overlap=None
):
    """SP all-gather of ``t`` on dim 2 with ``ttnn.experimental.high_bw_all_gather`` into the persistent
    ``out_buf`` (CCLManager.get_high_bw_gather_buffer); the returned tensor aliases it, do not deallocate.
    ``input_batch_index`` selects one slot of a multi-slot [B, 1, rows, D] cache; ``gathered_dim_size``
    bounds the rows moved per rank (rank r still lands at its fixed slot r*rows). ``overlap`` (an
    MsaOverlap, see tt/ccl.py) confines the workers to its gather sub-device with its persistent semaphores,
    so the gather can run beside compute on the other sub-device."""
    return ttnn.experimental.high_bw_all_gather(
        t,
        dim=2,
        output_tensor=out_buf,
        cluster_axis=mesh_config.sp_axis,
        num_links=ccl_manager.num_links,
        input_batch_index=input_batch_index,
        gathered_dim_size=gathered_dim_size,
        **_overlap_kwargs(overlap),
    )


def _split_index_heads(t, index_dim):
    """[1, 1, S, n*index_dim] -> [1, n, S, index_dim] (head-major split, like the main QKV split)."""
    s = t.shape[2]
    n = t.shape[-1] // index_dim
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [1, s, n, index_dim])
    t = ttnn.permute(t, (0, 2, 1, 3))  # [1, n, S, index_dim]
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def index_branch_forward(
    hidden_states,
    weights,
    rope_mats,
    transformation_mat,
    *,
    index_dim,
    rms_norm_eps,
    kv_actual_global=None,
    cluster_axis=None,
):
    """The MSA index branch: pre-roped index_q (n index heads, 1/TP col) + index_k (single shared head).

    proj -> split heads -> per-head RMSNorm -> RoPE. index_q_proj is column-parallel (4 index heads ->
    1/TP col); index_k_proj is replicated (shared head). Per device this yields index_q [1, n_idx_local,
    S, index_dim] (n_idx_local=1 at TP=4) and index_k [1, 1, S, index_dim] -> the MSA indexer.

    VERIFIED 2026-06-26 against transformers-main MiniMaxM3VLIndexer source (not just a summary):
      * order proj -> q_norm/k_norm -> apply_rotary_pos_emb — matches.
      * rope on BOTH index_q AND index_k — matches.
      * rotary width: reference does apply_rotary_pos_emb(idx_q, idx_k, cos[..,:index_head_dim], ...);
        with head_dim=128, partial_rotary_factor=0.5 the model cos/sin are 64-wide, and index_head_dim=
        sparse_index_dim=128, so the slice yields the full 64 -> PARTIAL-64 (rotate first 64 of the
        128-wide index head), same as main attention. Our apply_rope(rope_mats=main 64-wide) matches.
      * norm: index_q_norm/index_k_norm gains ship in the checkpoint, applied per-head.
    """
    iq = _split_index_heads(
        ttnn.linear(hidden_states, weights.index_q_proj), index_dim
    )  # [1, n_idx_local, S, index_dim]
    iq = apply_qk_norm_per_head(iq, weights.index_q_norm, rms_norm_eps)
    iq = apply_rope(
        iq,
        rope_mats,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=kv_actual_global,
        cluster_axis=cluster_axis,
    )

    ik = _split_index_heads(
        ttnn.linear(hidden_states, weights.index_k_proj), index_dim
    )  # [1, 1, S, index_dim] (shared)
    ik = apply_qk_norm_per_head(ik, weights.index_k_norm, rms_norm_eps)
    ik = apply_rope(
        ik,
        rope_mats,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=kv_actual_global,
        cluster_axis=cluster_axis,
    )
    return iq, ik


def msa_select_blocks(
    index_q,
    index_k,
    *,
    chunk_start_idx,
    scale,
    num_groups,
    block_size,
    topk_blocks,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    kv_len=None,
    overlap=None,
    index_k_tp_sharded=False,
):
    """Indexer scores + top-k -> block ids [1, num_groups, Sq, topk] (uint32, row-major). Arguments as for
    msa_indexer_sparse. ``overlap`` confines the indexer and the top-k to the compute sub-device."""
    # Block scores: scaled dot, causal -inf for future, group-sum, block-max-pool. bf16 row-major out.
    with zone("indexer"):
        block_scores = ttnn.experimental.indexer_score_msa(
            index_q,
            index_k,
            chunk_start_idx=chunk_start_idx,
            scale=scale,
            num_groups=num_groups,
            block_size=block_size,
            program_config=ttnn.IndexerScoreProgramConfig(
                q_chunk_size=64,  # work units: q rows per group, keys per band (tokens)
                k_chunk_size=1024,
                head_group_size=0,
            ),
            seq_shard_axes=[cluster_axis] if cluster_axis is not None else [],
            block_cyclic_sp_axis=block_cyclic_sp_axis,
            block_cyclic_chunk_local=block_cyclic_chunk_local,
            kv_len=kv_len,
            sub_core_grids=overlap.compute_cores if overlap is not None else None,
            block_cyclic_cache_tp_sharded=index_k_tp_sharded,
        )

        # Top-k block ids (uint32 row-major) — the block selection that encodes causality. The op already
        # force-locals the current (diagonal) block; upstream minimax_m3_vl forces ONLY the local block.
        # valid_length bounds the search to the populated block columns when K sits in a worst-case buffer.
        topk_kwargs = {}
        if kv_len is not None:
            assert kv_len % block_size == 0, f"kv_len ({kv_len}) must be a whole number of {block_size}-token blocks"
            topk_kwargs["valid_length"] = kv_len // block_size
        if overlap is not None:
            # topk_large_indices takes the sub-device explicitly and the rectangle within it; indexer_score_msa and
            # sparse_sdpa_msa take only the rectangle, and the runtime infers their sub-device from its cores.
            topk_kwargs["subdevice_id"] = overlap.compute_sd_id
            topk_kwargs["sub_core_grids"] = overlap.compute_cores
        block_ids = ttnn.experimental.topk_large_indices(block_scores, k=topk_blocks, **topk_kwargs)
    return block_ids


def msa_sparse_attend(
    q_rm,
    k,
    v,
    block_ids,
    *,
    chunk_start_idx,
    scale,
    block_size,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    overlap=None,
):
    """sparse_sdpa_msa over the selected blocks: row-major q in, row-major out [1, Hq, Sq, head_dim]. The caller
    owns the layout conversions (the overlap window admits only ops that stay on one sub-device)."""
    # sparse_sdpa_msa: q + block-ids row-major, K/V tiled; expands blocks->tokens internally. chunk_start_idx +
    # cluster_axis drive the token-level diagonal-block causal mask with the per-device SP start (chunk_start =
    # chunk_start_idx + rank*Sq); q must be bf16 (the op rejects fp8 q under causal).
    return ttnn.transformer.sparse_sdpa_msa(
        q_rm,
        k,
        v,
        block_ids,
        scale=scale,
        block_size=block_size,
        chunk_start_idx=chunk_start_idx,
        cluster_axis=cluster_axis,
        block_cyclic_sp_axis=block_cyclic_sp_axis,
        block_cyclic_chunk_local=block_cyclic_chunk_local,
        enable_kv_block_cache=MSA_KV_BLOCK_CACHE,
        sub_core_grids=overlap.compute_cores if overlap is not None else None,
    )


def msa_indexer_sparse(
    index_q,
    index_k,
    q,
    k,
    v,
    *,
    chunk_start_idx,
    scale,
    num_groups,
    block_size,
    topk_blocks,
    return_block_ids=False,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    kv_len=None,
    index_k_tp_sharded=False,
):
    """The MSA op chain over a FULL-context (already-gathered) K/V; index_q/q may stay SP-sharded.

    index_q [1, num_groups, Sq, index_dim]   index_k [1, 1, T, index_dim]   (1 shared index-k head)
    q       [1, Hq, Sq, head_dim]            k, v    [1, n_kv, T, head_dim]  (TILE layout)
    cluster_axis: when set, the ops derive a PER-DEVICE causal chunk_start from the device's mesh
      coordinate along that axis -> chunk_start = chunk_start_idx + rank*Sq (Sq = q's S/sp rows), so
      q/index_q stay SP-sharded. None -> uniform chunk_start_idx (single-device / gathered query).
    block_cyclic_sp_axis / block_cyclic_chunk_local (set together): the AllGather'd K/V/index_k stay in
      block-cyclic ("slab") SP order; the indexer (K reader) and sparse_sdpa_msa (K reader / V writer)
      remap each natural block-id to its physical block in-kernel, so nothing is reordered on the host.
      None -> the tensors are already natural (single-chunk / pre-reordered).
    kv_len: when K/index_k are allocated at a FIXED worst-case T (the persistent high_bw_all_gather
      buffers, T = cache capacity) this is the valid natural-position prefix actually written (a multiple
      of block_size). The indexer scores/writes only columns [0, kv_len) (hash-excluded runtime arg, so a
      growing prefix reuses one program) and top-k ranks only those kv_len/block_size block columns; the
      stale tail past kv_len is never read. None -> T is the valid length (the legacy exact-size gather).
    index_k_tp_sharded: index_k was gathered from a TP-deduped cache, so it is striped across all sp*tp chips
      (chip (s, t) at (s*tp + t)*rows) rather than sp; the indexer's key remap follows. K / V keep the SP layout.
    -> out  [1, Hq, Sq, head_dim]
    """
    geometry = dict(
        chunk_start_idx=chunk_start_idx,
        cluster_axis=cluster_axis,
        block_cyclic_sp_axis=block_cyclic_sp_axis,
        block_cyclic_chunk_local=block_cyclic_chunk_local,
    )
    block_ids = msa_select_blocks(
        index_q,
        index_k,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        kv_len=kv_len,
        index_k_tp_sharded=index_k_tp_sharded,
        **geometry,
    )
    with zone("sparse_sdpa"):
        out = msa_sparse_attend(
            ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT), k, v, block_ids, scale=scale, block_size=block_size, **geometry
        )
        # sparse_sdpa_msa returns ROW_MAJOR; the model's concat_heads (prefill.py) needs TILE — match the
        # dense (ring_joint) output so the shared post-attention path works for MSA layers too.
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    return (out, block_ids) if return_block_ids else out


def msa_sp_attention_nocache(
    q,
    k,
    v,
    index_q,
    index_k,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    s_local,
    scale,
    block_size,
    topk_blocks,
    num_groups=1,
):
    """Sharded-query MSA under SP: AllGather only the KEYS; q/index_q stay sharded (S/sp rows/device).

    Each device scores ONLY its own S/sp query rows against the gathered full context, with per-device
    causality from the op's native mesh-coord chunk_start (cluster_axis=sp_axis -> rank*s_local on top of
    chunk_start_idx=cached_len). Output stays SP-sharded [1, Hq, s_local, head_dim] — no replication, no
    reshard — which is what the SP residual stream needs. This is the deployed path (vs the gather-everything
    golden, which gathers the query too). index_q is the device's group's index head; q is its TP head-slice.
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device

    sp = device.shape[sp_axis]
    if sp == 1:
        # Nothing to gather: one SP row already holds the full context (the collective rejects a one-device axis).
        def gather(key, t):
            return t

    else:
        # Exact-size activations: the worst-case buffer IS the gathered shape, so no kv_len bound is needed.
        def gather(key, t):
            src = _ensure_dram(t)
            buf = ccl_manager.get_high_bw_gather_buffer(
                key, (src.shape[0], src.shape[1], src.shape[2] * sp, src.shape[3]), src.dtype, src.layout
            )
            full = high_bw_sp_gather(src, mesh_config, ccl_manager, buf)
            if src is not t:
                src.deallocate(True)  # the DRAM staging copy; the caller still owns `t`
            return full

    # ag_kv feeds sparse_sdpa_msa; ag_index_k feeds the indexer — split so the two AG costs are
    # attributable to their consumer (see tests/perf/profile_prefill.py).
    with zone("ag_kv"):
        k_full = gather("msa_nocache_k", k)
        v_full = gather("msa_nocache_v", v)
    with zone("ag_index_k"):
        index_k_full = gather("msa_nocache_index_k", index_k)
    # Per-device causality from the mesh coordinate: device r derives chunk_start = cached_len + r*Sq (Sq = q's
    # s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        cluster_axis=sp_axis,
    )


def msa_cache_read_extent(cached_len, chunk_local, sp, block_size, tp=1):
    """(kv_len, n_rows) for the cross-chunk MSA read after the chunk at ``cached_len`` has been written.

    The KV writer places global position g on SP rank (g // chunk_local) % sp, so a chunk that starts
    mid-slab (``cached_len`` not a whole number of ``chunk_local * sp`` chunks, e.g. a multi-turn resume at a
    32-token boundary) leaves the ranks unevenly filled. ``n_rows`` is the FULLEST rank's local row count
    covering the written prefix [0, cached_len + chunk_global) -- rank 0, which owns the first block of every
    slab -- since ``high_bw_all_gather`` gathers the same local prefix from every rank. ``kv_len`` is that prefix
    rounded up to whole ``block_size`` blocks (the indexer pools and top-k selects in blocks). Positions past a
    rank's written prefix -- including [end, kv_len) -- hold zeros or stale finite KV in the persistent gather
    buffer (zeroed at allocation, see CCLManager.get_high_bw_gather_buffer), all future to every query of the
    chunk, so the indexer's causal mask and sparse_sdpa_msa's diagonal-block mask never attend them.

    ``tp`` > 1 sizes the read of a TP-deduped cache (index_k under ``M3_INDEX_K_TP_SHARD``): each SP rank's
    ``chunk_local`` rows of a slab are split into ``tp`` stripes of ``chunk_local / tp``, one per chip, so
    ``n_rows`` is chip (0, 0)'s local row count, which owns the first stripe of every slab. ``kv_len`` is unchanged.
    """
    assert (
        cached_len % ttnn.TILE_SIZE == 0
    ), f"cached_len={cached_len} must be a multiple of {ttnn.TILE_SIZE} (the KV writer's tile grid)"
    assert chunk_local % block_size == 0, f"chunk_local={chunk_local} must be a whole number of {block_size} blocks"
    assert (
        chunk_local % (tp * ttnn.TILE_SIZE) == 0
    ), f"chunk_local={chunk_local} must split into {tp} whole-tile stripes"
    chunk_global = chunk_local * sp
    stripe = chunk_local // tp
    end = cached_len + chunk_global  # the chunk (incl. its pad tail) is written up to here
    kv_len = (end + block_size - 1) // block_size * block_size
    full_slabs, rem = divmod(end, chunk_global)
    n_rows = full_slabs * stripe + min(rem, stripe)
    return kv_len, n_rows


_FABRIC_2D_CONFIGS = (
    ttnn.FabricConfig.FABRIC_2D,
    ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
)

# Route messages already logged by _gather_tp_sharded_index_k: once per process, not per layer x chunk.
_logged_index_k_routes = set()


def _log_index_k_route_once(is_2d, *, sp, tp, rows):
    if is_2d in _logged_index_k_routes:
        return
    _logged_index_k_routes.add(is_2d)
    fabric = ttnn.get_fabric_config()
    if is_2d:
        logger.info(f"[msa] index_k TP-dedup gather route: 2D full-mesh snake (fabric={fabric})")
        return
    logger.info(f"[msa] index_k TP-dedup gather route: 1D TP-then-SP (fabric={fabric})")
    # The SP leg's extent is set by the slot capacity, not the written prefix (see the docstring above).
    logger.warning(
        f"[msa] index_k TP-dedup on 1D fabric: the SP leg moves ~(tp-1)/tp of the slot capacity "
        f"({rows * sp * tp} tokens) per MSA layer per chunk, whatever the context length. Prefer a 2D fabric "
        f"when max_seq_len is well above typical prompt lengths."
    )


def _index_k_route_is_2d():
    return ttnn.get_fabric_config() in _FABRIC_2D_CONFIGS


def _gather_tp_sharded_index_k(
    index_k_cache,
    bufs,
    *,
    cache_row,
    mesh_config,
    ccl_manager,
    tp_axis,
    cached_len,
    chunk_local,
    block_size,
    overlap,
):
    """Rebuild one cache row (slot_id, layer) of the TP-deduped index_k cache as the full context on every chip.

    Each chip holds ``rows = seq_local/tp`` rows (see MiniMaxKVCache). The result is ``bufs["index_k"]``, the
    persistent worst-case buffer ``[1, 1, rows*sp*tp, hd]`` with chip (s, t) at ``(s*tp + t)*rows``: the
    stripe-major layout the indexer decodes in-kernel with ``block_cyclic_cache_tp_sharded``. Two routes give
    the same buffer:

      * 2D fabric: one full-mesh snake ``high_bw_all_gather`` (row-major over the mesh IS the sp*tp order),
        moving only the written prefix of every chip.
      * otherwise: a TP-inner gather into ``bufs["index_k_row"]`` (chip t at t*rows inside each SP row), then
        an SP-outer gather of that. The SP leg has to move every TP stripe's fixed slot up to the last one's
        prefix, so it carries close to the whole capacity rather than the prefix.

    ``overlap`` (None outside a window) runs the route, both legs, on the gather sub-device. The gathered tensor
    aliases the persistent buffers: never deallocate.
    """
    device = ccl_manager.mesh_device
    sp_axis = mesh_config.sp_axis
    sp, tp = device.shape[sp_axis], device.shape[tp_axis]
    rows = index_k_cache.shape[2]
    _, n_rows = msa_cache_read_extent(cached_len, chunk_local, sp, block_size, tp=tp)
    assert n_rows <= rows, f"index_k cache read past capacity: {n_rows} > {rows} rows"
    full = bufs["index_k"]

    is_2d = _index_k_route_is_2d()
    _log_index_k_route_once(is_2d, sp=sp, tp=tp, rows=rows)
    if is_2d:
        return ttnn.experimental.high_bw_all_gather(
            index_k_cache,
            dim=2,
            output_tensor=full,
            cluster_axis=None,
            num_links=ccl_manager.num_links,
            input_batch_index=cache_row,
            gathered_dim_size=n_rows * sp * tp,
            **_overlap_kwargs(overlap),
        )

    row = ttnn.experimental.high_bw_all_gather(
        index_k_cache,
        dim=2,
        output_tensor=bufs["index_k_row"],
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        input_batch_index=cache_row,
        gathered_dim_size=n_rows * tp,
        **_overlap_kwargs(overlap),
    )
    return high_bw_sp_gather(
        row, mesh_config, ccl_manager, full, gathered_dim_size=((tp - 1) * rows + n_rows) * sp, overlap=overlap
    )


def msa_cache_gather(
    kv_cache, *, cache_row, mesh_config, ccl_manager, cached_len, chunk_local, block_size, set_idx=0, overlap=None
):
    """Gather ``cache_row`` (= slot_id * num_layers + layer) of the packed cache across SP into the persistent
    worst-case buffers ``(k_full, v_full, index_k_full, kv_len)``. ``set_idx`` selects one of two buffer sets so a
    gather can land while the other set is still being read (the packed-forward overlap); ``overlap`` runs it on
    the gather sub-device. The gathered tensors alias the persistent buffers: never deallocate."""
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    sp = device.shape[sp_axis]
    assert sp > 1, "msa_cache_gather needs sp > 1 (high_bw_all_gather rejects a one-device axis)"
    # The replicated cache tensors carry a 1-D tensor topology, and high_bw_all_gather requires
    # cluster_axis < that rank, so the SP gather only works with SP on mesh axis 0.
    assert sp_axis == 0, f"msa_cache_gather needs sp_axis == 0 (got {sp_axis})"
    seq_local = kv_cache.k.shape[2]  # per-device cache capacity (rows)
    kv_len, n_rows = msa_cache_read_extent(cached_len, chunk_local, sp, block_size)
    assert (
        n_rows <= seq_local and kv_len <= seq_local * sp
    ), f"cache read past capacity: {n_rows} rows / kv_len {kv_len} > {seq_local} rows x {sp}"
    bufs = msa_gather_buffers(kv_cache, ccl_manager, sp, set_idx)

    def gather(name, cache_t):
        return high_bw_sp_gather(
            cache_t,
            mesh_config,
            ccl_manager,
            bufs[name],
            input_batch_index=cache_row,
            gathered_dim_size=n_rows * sp,
            overlap=overlap,
        )

    # ag_kv feeds sparse_sdpa_msa; ag_index_k feeds the indexer — split so the two AG costs are
    # attributable to their consumer (see tests/perf/profile_prefill.py).
    with zone("ag_kv"):
        k_full = gather("k", kv_cache.k)
        v_full = gather("v", kv_cache.v)
    with zone("ag_index_k"):
        if kv_cache.index_k_tp_axis is None:
            index_k_full = gather("index_k", kv_cache.index_k)
        else:
            index_k_full = _gather_tp_sharded_index_k(
                kv_cache.index_k,
                bufs,
                cache_row=cache_row,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                tp_axis=kv_cache.index_k_tp_axis,
                cached_len=cached_len,
                chunk_local=chunk_local,
                block_size=block_size,
                overlap=overlap,
            )
    return k_full, v_full, index_k_full, kv_len


def msa_gather_buffers(kv_cache, ccl_manager, sp, set_idx=0):
    """Buffer set ``set_idx`` of the persistent worst-case gather buffers for this cache: ``k`` / ``v`` /
    ``index_k`` -> [1, 1, seq_local*sp, hd], plus ``index_k_row`` -> [1, 1, rows*tp, hd] (the TP leg's landing)
    when a TP-deduped index_k is rebuilt over a 1D fabric. Allocated on first use by a full-grid op (so never
    inside the overlap window)."""
    seq_local = kv_cache.k.shape[2]

    def buf(name, cache_t, rows):
        return ccl_manager.get_high_bw_gather_buffer(
            # set 0 keeps the serial path's key, so both paths share those buffers
            f"msa_cache_{name}" if set_idx == 0 else f"msa_cache_{name}/{set_idx}",
            (1, 1, rows, cache_t.shape[3]),
            cache_t.dtype,
        )

    bufs = {"k": buf("k", kv_cache.k, seq_local * sp), "v": buf("v", kv_cache.v, seq_local * sp)}
    if kv_cache.index_k_tp_axis is None:
        bufs["index_k"] = buf("index_k", kv_cache.index_k, seq_local * sp)
        return bufs
    tp = ccl_manager.mesh_device.shape[kv_cache.index_k_tp_axis]
    rows = kv_cache.index_k.shape[2]  # seq_local / tp: the rebuild lands the same seq_local*sp rows
    bufs["index_k"] = buf("index_k_tp", kv_cache.index_k, rows * sp * tp)
    if not _index_k_route_is_2d():
        bufs["index_k_row"] = buf("index_k_tp_row", kv_cache.index_k, rows * tp)
    return bufs


def msa_sp_attention_cache_read(
    q,
    index_q,
    kv_cache,
    *,
    cache_row,
    mesh_config,
    ccl_manager,
    cached_len,
    chunk_local,
    scale,
    block_size,
    topk_blocks,
    num_groups=1,
):
    """Cross-chunk MSA: the current chunk's queries attend the accumulated context, gathered across SP
    straight out of ``cache_row`` (= slot_id * num_layers + layer) of the packed
    ``[num_users*num_layers, 1, seq_local, hd]`` ND-sharded cache by ``high_bw_all_gather``
    (``input_batch_index=cache_row``, ``gathered_dim_size`` = the
    written prefix, incl. the current chunk that prefill.py wrote before calling us). The output is the
    persistent worst-case buffer ``[1, 1, seq_local*sp, hd]`` with rank r at the fixed slot r*seq_local,
    which is the block-cyclic layout the indexer / sparse_sdpa_msa decode in-kernel (stride T/sp ==
    seq_local); ``kv_len`` bounds them to the written prefix. Returns the chunk's SP-sharded attention out
    ``[1, Hq_local, s_local, hd]``. The gathered tensors alias the persistent buffers: never deallocate.
    A TP-deduped index_k cache (``kv_cache.index_k_tp_axis``) is rebuilt by ``_gather_tp_sharded_index_k``.
    """
    sp_axis = mesh_config.sp_axis
    k_full, v_full, index_k_full, kv_len = msa_cache_gather(
        kv_cache,
        cache_row=cache_row,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        cached_len=cached_len,
        chunk_local=chunk_local,
        block_size=block_size,
    )
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        cluster_axis=sp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
        kv_len=kv_len,
        index_k_tp_sharded=kv_cache.index_k_tp_axis is not None,
    )


@dataclass
class MsaChunk:
    """One warm chunk of a packed forward as msa_chunks_pipelined consumes it: its post-RoPE queries
    (tiled, this chip's rows) and the cache position they attend from."""

    q: ttnn.Tensor
    index_q: ttnn.Tensor
    slot_id: int  # KV-cache slot of the request; the layer is the call's
    cached_len: int
    s_local: int


def msa_overlap_chunks(cached_lens, kv_cache, mesh_config, ccl_manager):
    """Indices of a packed forward's chunks (given by their ``cached_lens``) that msa_chunks_pipelined takes:
    the warm ones (cached_len > 0) when the overlap is on, a cache exists, at least two are warm and there is
    an SP axis to gather over; [] otherwise, and every chunk then runs its serial chain."""
    warm = [i for i, c in enumerate(cached_lens) if c > 0]
    if not (MSA_CHUNK_OVERLAP and kv_cache is not None and len(warm) >= 2):
        return []
    return warm if ccl_manager.mesh_device.shape[mesh_config.sp_axis] >= 2 else []


def msa_chunks_pipelined(
    chunks, *, kv_cache, layer_idx, mesh_config, ccl_manager, scale, block_size, topk_blocks, num_groups
):
    """The MSA chains of a packed forward's warm chunks (``MsaChunk`` records in forward order), with the
    next chunk's cache gather overlapped under this chunk's block selection and sparse attention. Reads
    cache layer ``layer_idx`` of each chunk's slot. Returns the tiled attention outputs in chunk order.

    Schedule, with G(i) = chunk i's K/V/index_k gather (gather sub-device, buffer set i % 2) and C(i) = its
    indexer -> top-k -> sparse_sdpa_msa (compute sub-device):
        G(0); for i: join(G(i), C(i-1)); G(i+1); C(i)
    The join before G(i+1) covers both sub-devices, so G(i+1) and C(i), enqueued after it, start only once G(i)
    has landed and C(i-1) has released the buffer set G(i+1) overwrites; G(i+1) does not wait for C(i), which
    is the overlap (a program stalls only on its own sub-device's predecessors). G(i+1) goes first because the
    dispatcher launches programs in queue order and blocks on each one's own sub-device: queued behind C(i)'s
    three ops it would launch only when sparse_sdpa does, queued before them it runs under the whole of C(i).
    A join over one sub-device would race that overwrite. Only sub-device-confined ops may run inside the window: a program whose cores
    span both sub-devices (a layout change, an allocation that runs as a program) is rejected at enqueue
    (fd_mesh_command_queue.cpp, "Programs must be executed on a single sub-device"); one whose cores happen to
    fit a single sub-device runs there, stalling only on that sub-device's predecessors.
    test_msa_chunks_pipelined_vs_serial (bit-exact against the serial chain) is the guard. The gather
    allocates nothing (persistent buffers, caller-owned semaphores), so a tensor
    freed on the compute side is only ever reused by a later compute-side program, after its reader. Eager only:
    the join is an event record, which trace capture rejects. Contract: tech_reports/SubDevices/SubDevices.md.
    """
    overlap = ccl_manager.get_msa_overlap()
    sp_axis = mesh_config.sp_axis
    sp = ccl_manager.mesh_device.shape[sp_axis]
    for set_idx in (0, 1):  # both buffer sets exist before the window (their allocation is a full-grid op)
        msa_gather_buffers(kv_cache, ccl_manager, sp, set_idx)
    q_rm = [ttnn.to_layout(c.q, ttnn.ROW_MAJOR_LAYOUT) for c in chunks]

    def gather(i):
        c = chunks[i]
        with zone(f"gather{i}", FINE):
            return msa_cache_gather(
                kv_cache,
                cache_row=kv_cache.cache_row(c.slot_id, layer_idx),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                cached_len=c.cached_len,
                chunk_local=c.s_local,
                block_size=block_size,
                set_idx=i % 2,
                overlap=overlap,
            )

    outs = []
    block_ids = None
    try:
        with overlap.window():
            pending = gather(0)
            for i, c in enumerate(chunks):
                overlap.join()
                k_full, v_full, index_k_full, kv_len = pending
                if i + 1 < len(chunks):
                    pending = gather(i + 1)
                geometry = dict(
                    chunk_start_idx=c.cached_len,
                    cluster_axis=sp_axis,
                    block_cyclic_sp_axis=sp_axis,
                    block_cyclic_chunk_local=c.s_local,
                )
                with zone(f"chunk{i}", FINE):
                    block_ids = msa_select_blocks(
                        c.index_q,
                        index_k_full,
                        scale=scale,
                        num_groups=num_groups,
                        block_size=block_size,
                        topk_blocks=topk_blocks,
                        kv_len=kv_len,
                        overlap=overlap,
                        index_k_tp_sharded=kv_cache.index_k_tp_axis is not None,
                        **geometry,
                    )
                    with zone("sparse_sdpa"):
                        outs.append(
                            msa_sparse_attend(
                                q_rm[i],
                                k_full,
                                v_full,
                                block_ids,
                                scale=scale,
                                block_size=block_size,
                                overlap=overlap,
                                **geometry,
                            )
                        )
                    ttnn.deallocate(block_ids)
                    block_ids = None
    except BaseException:
        if block_ids is not None:  # the block ids of the chunk whose attention raised
            ttnn.deallocate(block_ids)
        for out_rm in outs:  # the finished chunks' outputs, which an aborted window never tiles
            ttnn.deallocate(out_rm)
        raise
    finally:
        for t in q_rm:
            ttnn.deallocate(t)
    # The tiling the serial chain charges to sparse_sdpa, outside the window (a full-grid op).
    tiled = []
    for i, out_rm in enumerate(outs):
        with zone(f"chunk{i}", FINE), zone("sparse_sdpa"):
            tiled.append(ttnn.to_layout(out_rm, ttnn.TILE_LAYOUT))
        ttnn.deallocate(out_rm)
    return tiled
