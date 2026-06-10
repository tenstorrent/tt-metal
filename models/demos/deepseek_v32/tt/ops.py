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
from loguru import logger

import ttnn


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    """First-shard readback for replicated mesh tensors (host fallbacks only)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def indexer_logits(q: ttnn.Tensor, k: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
    """
    Index scores per (query, key): logits[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t]).

    Fused reference: DeepGEMM fp8_mqa_logits (causal ks/ke windows; fp8 inputs).
    Workaround: composed ttnn ops, bf16, causal handled by the caller's mask add.

    Args:
        q: [1, H_idx, Sq, D_idx] index queries (non-interleaved RoPE preapplied)
        k: [1, 1, Skv, D_idx] index keys (shared across heads)
        w: [1, 1, Sq, H_idx] per-head weights (weights_proj output)
    Returns:
        logits [1, 1, Sq, Skv] bf16
    """
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # [1, H, Sq, Skv]
    scores = ttnn.relu(scores)
    # Weighted head reduce: [1,1,Sq,H] x [1,H,Sq,Skv] — bring H next to Skv and matmul.
    scores = ttnn.permute(scores, (0, 2, 1, 3))  # [1, Sq, H, Skv]
    w_rows = ttnn.permute(w, (0, 2, 1, 3))  # [1, Sq, 1, H]
    logits = ttnn.matmul(w_rows, scores)  # [1, Sq, 1, Skv]
    return ttnn.permute(logits, (0, 2, 1, 3))  # [1, 1, Sq, Skv]


def topk_indices(logits: ttnn.Tensor, k: int) -> ttnn.Tensor:
    """
    Top-k key indices per query row. Out [1, 1, Sq, k] uint32 (FlashMLA contract:
    indices replace the causal mask; caller pre-masks logits so future keys never win).
    Workaround: ttnn.topk needs row-major last-dim and k<=Skv; host fallback otherwise.
    """
    skv = logits.shape[-1]
    if k > skv:
        raise ValueError(f"topk k={k} exceeds Skv={skv}; pad upstream (status.md API 2)")
    try:
        # ttnn.topk asserts TILE layout (topk_device_operation.cpp) — agreement 4's
        # "row-major" note was wrong; documented in status.md.
        _, indices = ttnn.topk(logits, k, dim=-1)
        return indices
    except Exception as e:  # untested at k=2048 (status.md issue 2)
        logger.warning(f"ttnn.topk failed ({e}); host fallback")
        _, idx = torch.topk(_to_host(logits).float(), k, dim=-1)
        return ttnn.from_torch(
            idx.to(torch.int32),
            device=logits.device(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(logits.device()),
        )


def sparse_mla(q: ttnn.Tensor, kvpe: ttnn.Tensor, indices: ttnn.Tensor, scale: float) -> ttnn.Tensor:
    """
    Absorbed MQA over the top-k selected latents only (FlashMLA sparse contract:
    no causal mask — indices already encode it).

    Args:
        q: [1, H, Sq, 576] absorbed queries (nope·wkv_b ++ rope)
        kvpe: [1, 1, Skv, 576] latent cache (kv 512 ++ pe 64)
        indices: [1, 1, Sq, k] uint32 from topk_indices
        scale: softmax scale (with YaRN mscale)
    Returns:
        out [1, H, Sq, 512] bf16 latent attention output
    CPU FALLBACK (gather+SDPA on host): no ttnn gather over Skv; fused op is follow-up.
    """
    # q is head-sharded across TP; kvpe/indices replicated. Compute per shard so
    # each chip's output holds its own heads (chip i's out feeds chip i's o_proj).
    kvpe_t, idx = _to_host(kvpe), _to_host(indices).long()
    sel = kvpe_t[0, 0][idx[0, 0]]  # [Sq, k, 576]
    # Causality must be re-imposed here: rows with fewer than k causal keys get
    # arbitrary (future) indices from topk's -inf band, and scores are recomputed
    # from latents so the -inf does not carry over. FlashMLA handles this with
    # per-row topk_length; the fused op must too.
    sq = idx.shape[2]
    future = (idx[0, 0] > torch.arange(sq).view(sq, 1)).unsqueeze(0)  # [1, Sq, k]
    outs = []
    for q_shard in ttnn.get_device_tensors(q):
        q_t = ttnn.to_torch(q_shard)
        scores = (torch.einsum("hsd,skd->hsk", q_t[0].float(), sel.float()) * scale).masked_fill(future, float("-inf"))
        outs.append(torch.einsum("hsk,skc->hsc", scores.softmax(-1), sel[..., :512].float()).unsqueeze(0))
    return ttnn.from_torch(
        torch.cat(outs, dim=1).to(torch.bfloat16),
        device=q.device(),
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(q.device(), mesh_shape=tuple(q.device().shape), dims=(None, 1)),
    )
