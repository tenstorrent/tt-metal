# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Missing-op workarounds for DSA (status.md "Missing op APIs").

Each function carries the agreed API; the body is a workaround per status.md
"Approach to missing ops" (composed ttnn ops > CPU fallback > stub). Real C++
ops are follow-up work; replacing a body must not change the signature/shapes.

Conventions (match v3): activations [1, B, S, ·], TILE_LAYOUT, bf16, indexer
replicated across TP, B=1 prefill.
"""

import torch

import ttnn


def indexer_program_config(skv: int, head_group: int = 16) -> "ttnn.IndexerScoreProgramConfig":
    """Indexer kernel work-unit knobs for the DeepSeek-V3.2 (H_idx=64) indexer:
    QC=2 (q_chunk=64), KC=8 (k_chunk=256).

    ``k_chunk`` is capped to the key length because the op requires KC <= Skv/32; at the
    model's DSA K (end_pos > index_topk=2048) the cap is inert and KC stays 8. Shape unit
    tests with tiny Skv get the largest valid KC (e.g. Skv=128 -> KC=4).

    ``head_group`` (HB) = heads resident at once (0 = all). With the indexer run replicated at
    all 64 heads, HB=0 overflows Blackhole L1 at KC>=4, so the replicated default streams (HB=16).
    Under the TP-head-sharded path (change B, mla.py::_indexer_topk) each chip holds H_idx/tp
    heads (<=32 at tp>=2), which fits resident, so that path passes head_group=0. See
    INDEXER_OP.md "head residency".
    """
    return ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=min(256, skv), head_group_size=head_group)


# Replicated full-model default (H_idx=64, no TP head-shard): QC=2 / KC=8 / HB=16.
INDEXER_FULL_MODEL_CONFIG = indexer_program_config(256)


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    """First-shard readback for replicated mesh tensors (test / diagnostic readback only;
    the compute path is fully on device)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def indexer_logits(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    w: ttnn.Tensor,
    chunk_start_idx: int = 0,
    program_config: "ttnn.IndexerScoreProgramConfig | None" = None,
) -> ttnn.Tensor:
    """
    Index scores per (query, key), causality fused:
      logits[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t])  for t <= chunk_start_idx + s,
      logits[s, t] = -inf                                   for future/pad columns.

    Backed by the merged ``ttnn.experimental.indexer_score`` C++ op
    (replaces the composed matmul+relu+head-sum workaround). The op applies the
    causal -inf mask itself from ``chunk_start_idx`` — the caller no longer adds a
    triu mask (status.md missing-op (11)).

    Args:
        q: [1, H_idx, Sq, D_idx] index queries (non-interleaved RoPE preapplied), tiled bf16
        k: [1, 1, Skv, D_idx] index keys (shared across heads), tiled bf16
        w: [1, 1, Sq, H_idx] per-head weights (weights_proj output, scales pre-folded)
        chunk_start_idx: global position of query row 0 (causal offset for chunked prefill)
        program_config: indexer kernel work-unit knobs; defaults to the full-model
            QC=2 / KC=8 / HB=16 (INDEXER_FULL_MODEL_CONFIG). Tests with tiny Skv pass their
            own via indexer_program_config(Skv) (max valid KC for that key length).
    Returns:
        logits [1, 1, Sq, Skv] bf16 ROW_MAJOR; future/pad columns -inf.
    """
    # The op takes per-head weights as [1, H_idx, Sq, 1]; weights_proj gives [1, 1, Sq, H_idx].
    weights = ttnn.permute(w, (0, 3, 2, 1))  # [1, H_idx, Sq, 1]
    if program_config is None:
        program_config = INDEXER_FULL_MODEL_CONFIG
    return ttnn.experimental.indexer_score(
        q, k, weights, chunk_start_idx=chunk_start_idx, program_config=program_config
    )


def topk_indices(logits: ttnn.Tensor, k: int) -> ttnn.Tensor:
    """
    Top-k key indices per query row. Out [1, 1, Sq, k] uint32 ROW_MAJOR, sorted
    descending (FlashMLA contract: indices replace the causal mask).

    Backed by the merged ``ttnn.experimental.topk_large_indices`` C++ op
    (Blackhole-only; ROW_MAJOR bf16 in → ROW_MAJOR uint32 out). It chains directly
    off ``indexer_logits`` (same ROW_MAJOR bf16 layout). Causal -inf columns from
    indexer_score survive as the sentinel index 0xFFFFFFFF when a row has fewer
    than ``k`` valid keys; ``sparse_mla`` drops those via its index > row_pos mask.

    k constraints (topk_large_indices): 16 <= k <= 2048 and k a multiple of 16.
    """
    return ttnn.experimental.topk_large_indices(logits, k=k)


LATENT_DIM = 512  # kv_lora_rank — V is kvpe[..., :LATENT_DIM]; K is the full 576 width


def sparse_mla(
    q: ttnn.Tensor,
    kvpe: ttnn.Tensor,
    indices: ttnn.Tensor,
    scale: float,
    start_pos: int = 0,
    sp_axis: int = 0,
    tp_axis: int = 1,
) -> ttnn.Tensor:
    """
    Absorbed MQA over the top-k selected latents only (FlashMLA sparse contract:
    no causal mask — indices already encode it).

    Backed by the merged ``ttnn.transformer.sparse_sdpa`` C++ op (Blackhole, single
    chip). The op is invoked SPMD on the SP×TP mesh: each chip runs the single-chip
    kernel over its own q shard, so q's distribution is preserved end-to-end —
    q **SP-sharded on sequence (dim2) and TP-sharded on heads (dim1)**, output the
    same. The op requires per-chip H a multiple of 32 (prod: 128 heads / tp=4 = 32).

    Masking is fully baked into ``indices`` via the ``0xFFFFFFFF`` sentinel
    (``indexer_score`` -inf'd future columns → ``topk_indices`` emits the sentinel as
    a contiguous tail). There is therefore NO position/causal math here; ``start_pos``
    is accepted only for signature parity with the old host fallback and is ignored
    (matches ``reference_cpu/sparse_sdpa_prefill.py``).

    Args:
        q: [1, H/tp, S/sp, 576] absorbed queries (nope·wkv_b ++ rope), TILE bf16
        kvpe: [1, 1, T, 576] full latent prefix **on device**, ROW_MAJOR bf16, replicated
            across the mesh (the caller gathers it full-T on device — no host round-trip;
            backlog 9). K = full 576, V = the leading 512 cols. Not deallocated here.
        indices: [1, 1, S_global, k] uint32 (global key positions), replicated;
            0xFFFFFFFF = masked. Re-sharded onto SP (dim2) to match q when sp > 1.
        scale: softmax scale (with YaRN mscale)
        start_pos: IGNORED — causality is encoded in ``indices`` (kept for parity)
    Returns:
        out [1, H/tp, S/sp, 512] bf16 — heads TP-sharded, sequence SP-sharded
    """
    del start_pos  # causality is in `indices` (sentinel), not positions
    mesh = q.device()
    sp = list(mesh.shape)[sp_axis]
    assert sp_axis == 0 and tp_axis == 1, "sparse_mla assumes sp_axis=0 (outer), tp_axis=1"

    # The op is ROW_MAJOR-only (unpadded); q comes in TILE.
    q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)

    # indices must align with q's per-chip sequence shard [1, 1, S/sp, k]. Replicated
    # (sp == 1) already matches q's full S; for sp > 1 redistribute the replicated rows
    # onto the SP axis on device.
    idx = indices
    if sp > 1:
        # Re-shard the replicated rows onto the SP axis on device — the inverse of all_gather
        # (mesh_partition): chip sp_i keeps queries [i·S/sp, (i+1)·S/sp), matching its q seq shard.
        idx = ttnn.mesh_partition(indices, dim=2, cluster_axis=sp_axis)

    # k_chunk_size must be a multiple of 32 that divides TOPK (prod TOPK=2048 → 128).
    topk = idx.shape[-1]
    k_chunk = next((c for c in (128, 64, 32) if topk % c == 0), 32)

    out = ttnn.transformer.sparse_sdpa(q_rm, kvpe, idx, v_dim=LATENT_DIM, scale=scale, k_chunk_size=k_chunk)
    ttnn.deallocate(q_rm)
    if idx is not indices:
        ttnn.deallocate(idx)
    ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # back to TILE for the downstream wkv_b2 linear
    ttnn.deallocate(out)
    return ret
