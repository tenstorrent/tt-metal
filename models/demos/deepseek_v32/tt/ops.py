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


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    """First-shard readback for replicated mesh tensors (host fallbacks only)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def indexer_logits(q: ttnn.Tensor, k: ttnn.Tensor, w: ttnn.Tensor, chunk_start_idx: int = 0) -> ttnn.Tensor:
    """
    Index scores per (query, key), causality fused:
      logits[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t])  for t <= chunk_start_idx + s,
      logits[s, t] = -inf                                   for future/pad columns.

    Backed by the merged ``ttnn.experimental.deepseek.indexer_score`` C++ op
    (replaces the composed matmul+relu+head-sum workaround). The op applies the
    causal -inf mask itself from ``chunk_start_idx`` — the caller no longer adds a
    triu mask (status.md missing-op (11)).

    Args:
        q: [1, H_idx, Sq, D_idx] index queries (non-interleaved RoPE preapplied), tiled bf16
        k: [1, 1, Skv, D_idx] index keys (shared across heads), tiled bf16
        w: [1, 1, Sq, H_idx] per-head weights (weights_proj output, scales pre-folded)
        chunk_start_idx: global position of query row 0 (causal offset for chunked prefill)
    Returns:
        logits [1, 1, Sq, Skv] bf16 ROW_MAJOR; future/pad columns -inf.
    """
    # The op takes per-head weights as [1, H_idx, Sq, 1]; weights_proj gives [1, 1, Sq, H_idx].
    weights = ttnn.permute(w, (0, 3, 2, 1))  # [1, H_idx, Sq, 1]
    return ttnn.experimental.deepseek.indexer_score(q, k, weights, is_causal=True, chunk_start_idx=chunk_start_idx)


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


def sparse_mla(
    q: ttnn.Tensor,
    kvpe_host: torch.Tensor,
    indices: ttnn.Tensor,
    scale: float,
    start_pos: int = 0,
    sp_axis: int = 0,
    tp_axis: int = 1,
) -> ttnn.Tensor:
    """
    Absorbed MQA over the top-k selected latents only (FlashMLA sparse contract:
    no causal mask — indices already encode it).

    Distribution (matches v3's q/out layout): q is **SP-sharded on sequence (dim2)
    and TP-sharded on heads (dim1)**; kvpe is gathered full-T (replicated) by the
    caller; indices are full-T (replicated). Output is re-sharded the same way as q
    (heads on TP, sequence on SP).

    Args:
        q: [1, H/tp, S/sp, 576] absorbed queries (nope·wkv_b ++ rope)
        kvpe_host: [T, 576] full latent prefix on host (caller already gathered it
            full-T — single-shot via SP all-gather, chunked via kv_cache_to_host —
            so there is NO device→host→device re-upload here; backlog 9)
        indices: [1, 1, S_global, k] uint32 (global key positions), replicated
        scale: softmax scale (with YaRN mscale)
        start_pos: global position of the chunk's first query row
    Returns:
        out [1, H/tp, S/sp, 512] bf16 — heads TP-sharded, sequence SP-sharded
    CPU FALLBACK (gather+SDPA on host). On-device path is feasible via ttnn.gather
    (per-row index, TILE) → matmul/softmax/matmul — backlog 8.
    """
    mesh = q.device()
    shape = list(mesh.shape)
    sp, tp = shape[sp_axis], shape[tp_axis]
    assert sp_axis == 0 and tp_axis == 1, "sparse_mla host fallback assumes sp_axis=0 (outer), tp_axis=1"

    kv = kvpe_host  # [T, 576]
    idx_full = _to_host(indices).long()[0, 0]  # [S_global, k]
    s_global = idx_full.shape[0]
    local = s_global // sp

    q_shards = ttnn.get_device_tensors(q)  # row-major: shard(sp_i, tp_j) at sp_i*tp + tp_j
    h_local = q_shards[0].shape[1]
    out_full = torch.zeros(1, h_local * tp, s_global, 512, dtype=torch.float32)
    for sp_i in range(sp):
        idx_i = idx_full[sp_i * local : (sp_i + 1) * local]  # [local, k]
        q_pos = start_pos + sp_i * local + torch.arange(local)  # global query rows
        # topk's -inf sentinel (0xFFFFFFFF) marks causally-masked keys; together with any
        # future index it is dropped here (FlashMLA per-row contract). Clamp first so the
        # gather stays in-bounds — masked rows contribute nothing after the softmax.
        future = ((idx_i > q_pos.view(local, 1)) | (idx_i >= kv.shape[0])).unsqueeze(0)  # [1, local, k]
        idx_i = idx_i.clamp(max=kv.shape[0] - 1)
        sel = kv[idx_i]  # [local, k, 576]
        for tp_j in range(tp):
            q_t = ttnn.to_torch(q_shards[sp_i * tp + tp_j])[0].float()  # [H/tp, local, 576]
            scores = (torch.einsum("hsd,skd->hsk", q_t, sel.float()) * scale).masked_fill(future, float("-inf"))
            o = torch.einsum("hsk,skc->hsc", scores.softmax(-1), sel[..., :512].float())  # [H/tp, local, 512]
            out_full[0, tp_j * h_local : (tp_j + 1) * h_local, sp_i * local : (sp_i + 1) * local] = o

    dims = [None, None]
    dims[tp_axis] = 1  # heads
    dims[sp_axis] = 2  # sequence
    return ttnn.from_torch(
        out_full.to(torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
    )
