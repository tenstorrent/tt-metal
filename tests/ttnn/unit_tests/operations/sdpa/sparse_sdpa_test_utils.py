# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the sparse_sdpa tests (basic post-commit suite + nightly suite).

Correctness uses SMALL parametric shapes (the golden gathers sel[S,k,D]; the full production
640/2048/56320 shape is ~2.8 GiB and is only exercised by the perf-only test in the nightly suite).
"""

import torch

import ttnn

# Head dims (k_dim, v_dim) are supplied by each test, not baked in here. MASKED_INDEX is the op's sentinel.
MASKED_INDEX = 0xFFFFFFFF  # sentinel: a masked slot (scores -inf, contributes 0); a contiguous tail per row


def sparse_mla(q, kvpe, indices, scale, v_dim):
    """Torch reference for the sparse-MLA prefill op. Absorbed MQA over the top-k selected latents named by
    `indices` (one shared latent KV head); masking is baked into `indices` (index == MASKED_INDEX scores -inf).
    Dims are derived from the inputs; V is the leading `v_dim` cols of the K_DIM-wide kvpe.
        q [1,H,S,K_DIM], kvpe [T,K_DIM], indices [1,1,S,k] uint32  ->  out [1,H,S,v_dim]
    """
    B, H, S, Dk = q.shape
    k = indices.shape[-1]
    T = kvpe.shape[0]
    idx = indices.reshape(B, S, k)
    masked = idx == MASKED_INDEX
    idx_safe = torch.where(masked, torch.zeros_like(idx), idx).to(torch.int64)  # clamp sentinels in-bounds
    kv = kvpe.unsqueeze(0).expand(B, T, Dk)
    sel = torch.gather(  # gather the k selected KV rows (shared across heads): [B,T,Dk] -> [B,S,k,Dk]
        kv.unsqueeze(1).expand(B, S, T, Dk), 2, idx_safe.view(B, S, k, 1).expand(B, S, k, Dk)
    )
    scores = torch.einsum("bhsd,bsjd->bhsj", q, sel) * scale  # full-K_DIM scores [B,H,S,k]
    scores = scores.masked_fill(masked.view(B, 1, S, k), float("-inf"))
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.einsum("bhsj,bsjd->bhsd", probs, sel[..., :v_dim])  # weighted sum of V views [B,H,S,v_dim]


def make_inputs(H, S, T, TOPK, k_dim, n_valid_fn, seed=0):
    """Build (q, kv, indices) torch tensors matching the producer contract (tail-shaped sentinels)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, k_dim, generator=gen, dtype=torch.float32)
    kv = torch.randn(1, 1, T, k_dim, generator=gen, dtype=torch.float32)
    indices = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    for s in range(S):
        nv = max(1, min(TOPK, n_valid_fn(s)))
        perm = torch.randperm(T, generator=gen)[:nv]
        indices[0, 0, s, :nv] = perm
    return q, kv, indices


def golden(q, kv, indices, scale, v_dim):
    return sparse_mla(q, kv[0, 0], indices.to(torch.int64), scale, v_dim)  # [1,H,S,v_dim]


def to_dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def run_op(
    q,
    kv,
    indices,
    device,
    k_chunk_size,
    v_dim,
    compute_kernel_config=None,
    kv_dtype=ttnn.bfloat16,
    q_dtype=ttnn.bfloat16,
):
    q_host = q.to(torch.bfloat16) if q_dtype == ttnn.bfloat16 else q.to(torch.float32)
    tt_q = to_dev(q_host, device, q_dtype)  # ttnn quantizes float -> fp8 when q_dtype is fp8_e4m3
    kv_host = kv.to(torch.bfloat16) if kv_dtype == ttnn.bfloat16 else kv.to(torch.float32)
    tt_kv = to_dev(kv_host, device, kv_dtype)  # ttnn quantizes float -> fp8 when kv_dtype is fp8_e4m3
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    scale = q.shape[-1] ** -0.5  # 1/sqrt(k_dim), from the input width
    tt_out = ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_idx, v_dim, scale=scale, k_chunk_size=k_chunk_size, compute_kernel_config=compute_kernel_config
    )
    # Output dtype matches q. fp8 tensors can't be converted directly with to_torch, so typecast to bf16.
    if tt_out.dtype == ttnn.fp8_e4m3:
        tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
    return ttnn.to_torch(tt_out), scale


def pcc(out, golden_t):
    return torch.corrcoef(torch.stack([out.flatten().float(), golden_t.flatten().float()]))[0, 1].item()
