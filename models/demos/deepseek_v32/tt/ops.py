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

    Distribution (matches v3's q/out layout): q is **SP-sharded on sequence (dim2)
    and TP-sharded on heads (dim1)**; kvpe is gathered full-T (replicated) by the
    caller; indices are full-T (replicated). Output is re-sharded the same way as q
    (heads on TP, sequence on SP).

    Args:
        q: [1, H/tp, S/sp, 576] absorbed queries (nope·wkv_b ++ rope)
        kvpe: [1, 1, T, 576] full latent prefix (kv 512 ++ pe 64), replicated
        indices: [1, 1, S_global, k] uint32 (global key positions), replicated
        scale: softmax scale (with YaRN mscale)
        start_pos: global position of the chunk's first query row
    Returns:
        out [1, H/tp, S/sp, 512] bf16 — heads TP-sharded, sequence SP-sharded
    CPU FALLBACK (gather+SDPA on host): no ttnn gather over Skv; fused op is follow-up.
    """
    mesh = q.device()
    shape = list(mesh.shape)
    sp, tp = shape[sp_axis], shape[tp_axis]
    assert sp_axis == 0 and tp_axis == 1, "sparse_mla host fallback assumes sp_axis=0 (outer), tp_axis=1"

    kv = _to_host(kvpe)[0, 0]  # [T, 576]
    idx_full = _to_host(indices).long()[0, 0]  # [S_global, k]
    s_global = idx_full.shape[0]
    local = s_global // sp

    q_shards = ttnn.get_device_tensors(q)  # row-major: shard(sp_i, tp_j) at sp_i*tp + tp_j
    h_local = q_shards[0].shape[1]
    out_full = torch.zeros(1, h_local * tp, s_global, 512, dtype=torch.float32)
    for sp_i in range(sp):
        idx_i = idx_full[sp_i * local : (sp_i + 1) * local]  # [local, k]
        sel = kv[idx_i]  # [local, k, 576]
        q_pos = start_pos + sp_i * local + torch.arange(local)  # global query rows
        future = (idx_i > q_pos.view(local, 1)).unsqueeze(0)  # [1, local, k]
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
