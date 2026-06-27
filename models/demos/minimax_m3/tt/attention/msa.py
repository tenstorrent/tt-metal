# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 MSA (sparse) attention — the real model forward for the sparse layers (3-59).

Unlike the dense path (ring_joint, which reads the KV cache + gathers across SP *internally*),
``sparse_sdpa_msa`` is a pure dense-context kernel: it takes full-length K/V tensors and has no
cache-read. So the cache + cross-device gather live in THIS wrapper:

  each SP device holds a sequence shard of K / V / index_k (from the chunked-KV cache read).
  We AllGather those across the SP axis so every device materialises the full context, then:

    indexer_score_msa(index_q, index_k_full, chunk_start_idx=cached_len)  -> block scores
    inject sink (block 0) + topk_large_indices(topk_blocks)              -> block-ids   (the op
                                              already force-locals the current block, +inf)
    sparse_sdpa_msa(q, k_full, v_full, block-ids)                        -> attention out

SP sharding is CONTIGUOUS, no zigzag/balancing (per the op authors: chunked prefill needs no causal
load-balancing — MSA work per query is a fixed top-k, not the dense causal triangle).

``chunk_start_idx`` is the global position of query row 0. Because the indexer scores over the
gathered full context and the current chunk's queries all begin at ``cached_len``, this scalar is
uniform across SP devices (no per-device offset needed). Causality is encoded entirely by the block
selection; sparse_sdpa_msa applies no token mask.
"""

import torch

import ttnn

from .operations import apply_qk_norm_per_head, apply_rope

# M3 sparse_attention_config (configs/MiniMax-M3/config.json).
BLOCK_SIZE = 128
TOPK_BLOCKS = 16
SINK_BLOCK = 0  # sparse_init_block — the attention sink, always selected
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

    TODO(REVISIT) — confidence on the norm+rope here is uneven:
      * norm: HIGH — index_q_norm/index_k_norm weights ship in the checkpoint, so they are applied.
      * rope: MEDIUM — taken from a WebFetch *summary* of transformers-main MiniMaxM3VLIndexer
        (proj->norm->apply_rotary_pos_emb(cos/sin[..,:head_dim])), matches DeepSeek-DSA. NOT yet
        verified against the actual modeling source: confirm rope is on BOTH index_q AND index_k, the
        rotary width (partial-64 like main, via cos[..,:head_dim] on a 64-wide cos, vs full-128), and
        the order. Re-check before trusting the end-to-end token output. See PREFILL_PROPOSAL.md.
    """
    iq = _split_index_heads(ttnn.linear(hidden_states, weights.index_q_proj))  # [1, n_idx_local, S, IDX_DIM]
    iq = apply_qk_norm_per_head(iq, weights.index_q_norm, rms_norm_eps)
    iq = apply_rope(iq, rope_mats, transformation_mat, is_decode_mode=False)

    ik = _split_index_heads(ttnn.linear(hidden_states, weights.index_k_proj))  # [1, 1, S, IDX_DIM] (shared)
    ik = apply_qk_norm_per_head(ik, weights.index_k_norm, rms_norm_eps)
    ik = apply_rope(ik, rope_mats, transformation_mat, is_decode_mode=False)
    return iq, ik


def _sink_mask(num_groups, nblk, device):
    """A [1, num_groups, 1, nblk] additive mask with +inf at the sink block (broadcast over queries)."""
    m = torch.zeros(1, num_groups, 1, nblk, dtype=torch.float32)
    m[:, :, :, SINK_BLOCK] = float("inf")
    kwargs = {}
    if isinstance(device, ttnn.MeshDevice):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(
        m.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, **kwargs
    )


def msa_indexer_sparse(
    index_q, index_k, q, k, v, *, chunk_start_idx, scale, num_groups, device, return_block_ids=False, chunk_offset=None
):
    """The MSA op chain over a FULL-context (already-gathered) K/V; index_q/q may stay SP-sharded.

    index_q [1, num_groups, Sq, INDEX_DIM]   index_k [1, 1, T, INDEX_DIM]   (1 shared index-k head)
    q       [1, Hq, Sq, head_dim]            k, v    [1, n_kv, T, head_dim]  (TILE layout)
    chunk_offset: optional per-device uint32 tile (causal chunk-start in TILES) — when set, each SP
      device's Sq query rows use their own absolute positions, so q/index_q can stay sharded (Sq=S/sp).
    -> out  [1, Hq, Sq, head_dim]
    """
    # Block scores: scaled dot, causal -inf for future, group-sum, block-max-pool. bf16 row-major out.
    block_scores = ttnn.experimental.indexer_score_msa(
        index_q,
        index_k,
        chunk_start_idx=chunk_start_idx,
        scale=scale,
        num_groups=num_groups,
        block_size=BLOCK_SIZE,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
        chunk_offset=chunk_offset,
    )

    # Force the attention sink (block 0) to always be selected: +inf at block 0 before top-k. The op
    # already force-locals the current block (+inf), so we only add the sink here.
    nblk = block_scores.shape[-1]
    block_scores = ttnn.add(block_scores, _sink_mask(num_groups, nblk, device))

    # Top-k block ids (uint32 row-major) — the block selection that encodes causality.
    block_ids = ttnn.experimental.topk_large_indices(block_scores, k=TOPK_BLOCKS)

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


def _per_device_chunk_offset(cached_len, s_local, mesh_config, device):
    """Per-device causal chunk-start tile (uint32, one 32x32 tile/device) for the sharded-query path.

    SP device at rank r owns query rows [r*s_local : (r+1)*s_local) of the current chunk, whose absolute
    start is `cached_len + r*s_local`. We pass that (in TILES) at element [0,0] of a per-device tile,
    sharded across the SP axis and replicated across TP. (Functionally main's mesh-coord #47939 derives
    the same value from the device coordinate; here we build it host-side.)
    """
    rows, cols = tuple(device.shape)
    sp_axis = mesh_config.sp_axis
    assert sp_axis == 0, "sharded-query chunk_offset helper assumes SP on the row axis"
    sp = rows
    ts = ttnn.TILE_SIZE
    vals = torch.zeros(sp, 1, ts, ts, dtype=torch.int32)
    for r in range(sp):
        vals[r, 0, 0, 0] = (cached_len + r * s_local) // ts
    return ttnn.from_torch(
        vals,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=[0, None]),
    )


def msa_sp_attention_nocache(
    q, k, v, index_q, index_k, *, mesh_config, ccl_manager, cached_len, s_local, scale, num_groups=1
):
    """Sharded-query MSA under SP: AllGather only the KEYS; q/index_q stay sharded (S/sp rows/device).

    Each device scores ONLY its own S/sp query rows against the gathered full context, with a per-device
    causal `chunk_offset` (cached_len + rank*s_local). Output stays SP-sharded [1, Hq, s_local, head_dim]
    — no replication, no reshard — which is what the SP residual stream needs. This is the deployed path
    (vs the gather-everything golden, which gathers the query too). index_q is the device's group's index head; q is
    its TP head-slice.
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    k_full = mesh_config.allgather(k, ccl_manager, axis=sp_axis, dim=2)
    v_full = mesh_config.allgather(v, ccl_manager, axis=sp_axis, dim=2)
    index_k_full = mesh_config.allgather(index_k, ccl_manager, axis=sp_axis, dim=2)
    chunk_offset = _per_device_chunk_offset(cached_len, s_local, mesh_config, device)
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=0,
        scale=scale,
        num_groups=num_groups,
        device=device,
        chunk_offset=chunk_offset,
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

    def gather_natural(t):
        # Cache slices come out ND_SHARDED (the persistent cache's DRAM NdShard layout), which AllGather
        # rejects — convert to interleaved DRAM first. No-op for already-interleaved op-test inputs.
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        full_bc = mesh_config.allgather(t, ccl_manager, axis=sp_axis, dim=2)
        # The persistent cache is bf8 (a tile-only block format); the reorder needs ROW_MAJOR, so cast to
        # bf16 first. No-op for the bf16 op-test inputs; required for the bf8 cache slices from the model.
        if full_bc.dtype != ttnn.bfloat16:
            full_bc = ttnn.typecast(full_bc, ttnn.bfloat16)
        return _blockcyclic_to_natural(full_bc, sp, n_chunks, chunk_local)

    k_full = gather_natural(k_acc)
    v_full = gather_natural(v_acc)
    index_k_full = gather_natural(index_k_acc)
    chunk_offset = _per_device_chunk_offset(cached_len, s_local, mesh_config, device)
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=0,
        scale=scale,
        num_groups=num_groups,
        device=device,
        chunk_offset=chunk_offset,
    )
