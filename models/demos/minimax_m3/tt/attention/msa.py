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
"""

import os

import torch

import ttnn

from .operations import apply_qk_norm_per_head, apply_rope


def _dbg_msa_save(t, device, layer_idx, name):
    """DBG_MSA_BLOCKIDS=1: save a per-query MSA tensor (block_ids or block_scores), reassembled across SP
    rows on the query axis (col-0 device of each row), so we can diff chunked vs one-shot per global query
    position. -> /tmp/m3_msa/L{idx}_{name}.pt, shape [1, num_groups, S_global, last]."""
    if os.getenv("DBG_MSA_BLOCKIDS") != "1" or layer_idx is None:
        return
    try:
        import os as _os

        rows, cols = tuple(device.shape)
        dts = ttnn.get_device_tensors(t)
        shards = [ttnn.to_torch(dts[r * cols]) for r in range(rows)]  # col-0 of each SP row
        full = torch.cat(shards, dim=2)  # concat on the query (Sq) axis -> global order
        _os.makedirs("/tmp/m3_msa", exist_ok=True)
        torch.save(full, f"/tmp/m3_msa/L{layer_idx}_{name}.pt")
        from loguru import logger

        logger.info(f"[MSA L{layer_idx} {name}] saved {tuple(full.shape)}")
    except Exception as e:
        from loguru import logger

        logger.info(f"[MSA L{layer_idx} {name}] failed: {e}")


# M3 sparse_attention_config (configs/MiniMax-M3/config.json).
BLOCK_SIZE = 128
TOPK_BLOCKS = 16
NUM_INDEX_HEADS = 4  # sparse_num_index_heads (1 per GQA group; 1 per device at TP=4)
INDEX_DIM = 128  # sparse_index_dim


def _split_index_heads(t):
    """[1, 1, S, n*INDEX_DIM] -> [1, n, S, INDEX_DIM] (head-major split, like the main QKV split)."""
    s = t.shape[2]
    n = t.shape[-1] // INDEX_DIM
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [1, s, n, INDEX_DIM])
    t = ttnn.permute(t, (0, 2, 1, 3))  # [1, n, S, INDEX_DIM]
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def index_branch_forward(hidden_states, weights, rope_mats, transformation_mat, *, rms_norm_eps):
    """The MSA index branch: pre-roped index_q (n index heads, 1/TP col) + index_k (single shared head).

    proj -> split heads -> per-head RMSNorm -> RoPE. index_q_proj is column-parallel (4 index heads ->
    1/TP col); index_k_proj is replicated (shared head). Per device this yields index_q [1, n_idx_local,
    S, INDEX_DIM] (n_idx_local=1 at TP=4) and index_k [1, 1, S, INDEX_DIM] -> the MSA indexer.

    VERIFIED 2026-06-26 against transformers-main MiniMaxM3VLIndexer source (not just a summary):
      * order proj -> q_norm/k_norm -> apply_rotary_pos_emb — matches.
      * rope on BOTH index_q AND index_k — matches.
      * rotary width: reference does apply_rotary_pos_emb(idx_q, idx_k, cos[..,:index_head_dim], ...);
        with head_dim=128, partial_rotary_factor=0.5 the model cos/sin are 64-wide, and index_head_dim=
        sparse_index_dim=128, so the slice yields the full 64 -> PARTIAL-64 (rotate first 64 of the
        128-wide index head), same as main attention. Our apply_rope(rope_mats=main 64-wide) matches.
      * norm: index_q_norm/index_k_norm gains ship in the checkpoint, applied per-head.
    """
    iq = _split_index_heads(ttnn.linear(hidden_states, weights.index_q_proj))  # [1, n_idx_local, S, IDX_DIM]
    iq = apply_qk_norm_per_head(iq, weights.index_q_norm, rms_norm_eps)
    iq = apply_rope(iq, rope_mats, transformation_mat, is_decode_mode=False)

    ik = _split_index_heads(ttnn.linear(hidden_states, weights.index_k_proj))  # [1, 1, S, IDX_DIM] (shared)
    ik = apply_qk_norm_per_head(ik, weights.index_k_norm, rms_norm_eps)
    ik = apply_rope(ik, rope_mats, transformation_mat, is_decode_mode=False)
    return iq, ik


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
    device,
    return_block_ids=False,
    cluster_axis=None,
    layer_idx=None,
    dbg_tag=None,
):
    """The MSA op chain over a FULL-context (already-gathered) K/V; index_q/q may stay SP-sharded.

    index_q [1, num_groups, Sq, INDEX_DIM]   index_k [1, 1, T, INDEX_DIM]   (1 shared index-k head)
    q       [1, Hq, Sq, head_dim]            k, v    [1, n_kv, T, head_dim]  (TILE layout)
    cluster_axis: when set, the merged op derives a PER-DEVICE causal chunk_start from the device's
      mesh coordinate along that axis -> chunk_start = chunk_start_idx + rank*Sq (Sq = q's S/sp rows),
      so q/index_q stay SP-sharded. None -> uniform chunk_start_idx (single-device / gathered query).
      (Replaces the old host-built per-device chunk_offset tile; mesh-coord approach, #47939.)
    -> out  [1, Hq, Sq, head_dim]
    """
    # DBG: dump the indexer's actual inputs to localize a chunked-vs-oneshot score divergence.
    if os.getenv("DBG_MSA_BLOCKIDS") == "1" and layer_idx is not None and dbg_tag is not None:
        _dbg_msa_save(index_q, device, layer_idx, f"{dbg_tag}_idxq")  # sharded on Sq -> global order
        try:  # index_k is the full gathered context, replicated across rows -> dev0 has it all
            import os as _os

            ik0 = ttnn.to_torch(ttnn.get_device_tensors(index_k)[0]).float()
            _os.makedirs("/tmp/m3_msa", exist_ok=True)
            torch.save(ik0, f"/tmp/m3_msa/L{layer_idx}_{dbg_tag}_idxk.pt")
        except Exception:
            pass

    # Block scores: scaled dot, causal -inf for future, group-sum, block-max-pool. bf16 row-major out.
    block_scores = ttnn.experimental.indexer_score_msa(
        index_q,
        index_k,
        chunk_start_idx=chunk_start_idx,
        scale=scale,
        num_groups=num_groups,
        block_size=BLOCK_SIZE,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
        cluster_axis=cluster_axis,
    )

    # The op already force-locals the current block (+inf); the real MiniMax-M3 indexer forces ONLY the
    # local block (upstream minimax_m3_vl: index_local_blocks, no init/sink block), so nothing else is added.
    _dbg_msa_save(block_scores, device, layer_idx, f"{dbg_tag}_scores" if dbg_tag else None)

    # Top-k block ids (uint32 row-major) — the block selection that encodes causality.
    block_ids = ttnn.experimental.topk_large_indices(block_scores, k=TOPK_BLOCKS)
    _dbg_msa_save(block_ids, device, layer_idx, dbg_tag)

    # sparse_sdpa_msa: q + block-ids row-major, K/V tiled; expands blocks->tokens internally.
    out = ttnn.transformer.sparse_sdpa_msa(
        ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT),
        k,
        v,
        block_ids,
        scale=scale,
        block_size=BLOCK_SIZE,
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
    num_groups=1,
    return_block_ids=False,
    layer_idx=None,
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
    k_full = mesh_config.allgather(k, ccl_manager, axis=sp_axis, dim=2)
    v_full = mesh_config.allgather(v, ccl_manager, axis=sp_axis, dim=2)
    index_k_full = mesh_config.allgather(index_k, ccl_manager, axis=sp_axis, dim=2)
    # Per-device causality via the merged op's native mesh-coord chunk_start (#47939): device r derives
    # chunk_start = cached_len + r*Sq (Sq = q's s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        device=device,
        cluster_axis=sp_axis,
        layer_idx=layer_idx,
        dbg_tag="nocache",
    )


def _blockcyclic_to_natural(t, sp, n_chunks, chunk_local):
    """Reorder an AllGathered block-cyclic context [1, H, T, hd] to natural token order.

    ``update_padded_kv_cache`` stores chip r's slice as ``[chunk0_r, chunk1_r, ...]`` (chunk_local tokens
    each), so AllGather over the SP axis yields chip-major order — index ``(chip, chunk, c)``. Natural
    order is ``(chunk, chip, c)``. At chunk-aligned offsets that is exactly a transpose of the (chip, chunk)
    axes: reshape T -> [sp, n_chunks, chunk_local*hd], swap dims, reshape back. (Row-major for the middle
    transpose; the indexer/sparse re-tilize.)
    """
    H, T, hd = t.shape[1], t.shape[2], t.shape[3]
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [H, sp, n_chunks, chunk_local * hd])
    t = ttnn.transpose(t, 1, 2)  # (chip, chunk) -> (chunk, chip)
    t = ttnn.reshape(t, [1, H, T, hd])
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def msa_sp_attention(
    q,
    k_acc,
    v_acc,
    index_q,
    index_k_acc,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    s_local,
    n_chunks,
    chunk_local,
    scale,
    num_groups=1,
    layer_idx=None,
):
    """Cross-chunk MSA: the CURRENT chunk's queries attend the ACCUMULATED context read from the
    block-cyclic SP cache (the multi-chunk read path; ``msa_sp_attention_nocache`` is its single-chunk,
    contiguous-context sibling).

    Args (per device, on the (sp, tp) mesh):
        q, index_q          CURRENT chunk's CONTIGUOUS SP shards: q [1, Hq_local, s_local, hd],
                            index_q [1, num_groups, s_local, hd]  (chip r owns chunk positions
                            [cached_len + r*s_local : ...]).
        k_acc, v_acc        ACCUMULATED context's BLOCK-CYCLIC SP shards (as the cache stores them):
                            [1, n_kv_local, n_chunks*chunk_local, hd].
        index_k_acc         accumulated index_k block-cyclic shard [1, 1, n_chunks*chunk_local, hd].
        cached_len          valid prefix length BEFORE the current chunk (= (n_chunks-1)*chunk_local*sp).
        n_chunks            total chunks now in the cache (incl. current); chunk_local = tokens/chip/chunk.

    AllGather K/V/index_k across SP -> full block-cyclic context -> reorder to NATURAL token order (so the
    indexer's block-pool + causal offset see true positions) -> indexer (per-device chunk_offset) ->
    sparse_sdpa. Returns the current chunk's SP-sharded attention out [1, Hq_local, s_local, hd].
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    sp = device.shape[sp_axis]

    # KNOWN BUG (chunked-only): this device gather mis-orders the context (index_k PCC ~0.517 vs golden)
    # because to_memory_config does not delinearize the NdShard ("slab") cache to logical order, AND a
    # host-reorder workaround (from_torch) makes the indexer/sparse TensorAccessor read an invalid DRAM
    # bank (TT_FATAL "core 8-0"). PROPER FIX is a slab-aware kernel cache-read (see KERNEL TODO in msa
    # docstring / memory): make sparse_sdpa_msa + indexer_score_msa read the NdShard cache directly via
    # kv_cache_batch_idx/actual_isl like ring_joint. Until then the chunked MSA path is not correct.
    def gather_natural(t):
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        full_bc = mesh_config.allgather(t, ccl_manager, axis=sp_axis, dim=2)
        if full_bc.dtype != ttnn.bfloat16:
            full_bc = ttnn.typecast(full_bc, ttnn.bfloat16)
        return _blockcyclic_to_natural(full_bc, sp, n_chunks, chunk_local)

    k_full = gather_natural(k_acc)
    v_full = gather_natural(v_acc)
    index_k_full = gather_natural(index_k_acc)
    # Per-device causality via the merged op's native mesh-coord chunk_start (#47939): device r derives
    # chunk_start = cached_len + r*Sq (Sq = q's s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        device=device,
        cluster_axis=sp_axis,
        layer_idx=layer_idx,
        dbg_tag="cacheread",
    )
