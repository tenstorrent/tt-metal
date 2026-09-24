# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the sparse_sdpa tests (basic post-commit suite + nightly suite).

Correctness uses SMALL parametric shapes (the golden gathers sel[S,k,D]; the full production
640/2048/56320 shape is ~2.8 GiB and is only exercised by the perf-only test in the nightly suite).
"""

import pytest
import torch

import ttnn
from ttnn.operations.transformer_golden import MASKED_INDEX, sparse_mla


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


def golden(q, kv, indices, scale, v_dim, attention_sink=None):
    return sparse_mla(q, kv[0, 0], indices.to(torch.int64), scale, v_dim, attention_sink)  # [1,H,S,v_dim]


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
    kv_format = (
        ttnn.transformer.SparseKVFormat.BF16 if kv_dtype == ttnn.bfloat16 else ttnn.transformer.SparseKVFormat.FP8_E4M3
    )
    tt_out = ttnn.transformer.sparse_sdpa(
        tt_q,
        tt_kv,
        tt_idx,
        v_dim,
        kv_format=kv_format,
        scale=scale,
        k_chunk_size=k_chunk_size,
        compute_kernel_config=compute_kernel_config,
    )
    # Output dtype matches q. fp8 tensors can't be converted directly with to_torch, so typecast to bf16.
    if tt_out.dtype == ttnn.fp8_e4m3:
        tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
    return ttnn.to_torch(tt_out), scale


def pcc(out, golden_t):
    return torch.corrcoef(torch.stack([out.flatten().float(), golden_t.flatten().float()]))[0, 1].item()


# DeepSeek-V4 CSA geometries, shared by correctness, determinism, and performance tests.
ATTENTION_SINK_SHAPES = [
    pytest.param(128, 4, 128, 64, 32, id="small-boundaries"),
    pytest.param(64, 640, 128 + 512, 512, 128, id="v4-flash-full-selection"),
    pytest.param(128, 640, 128 + 1024, 512, 128, id="v4-pro-full-selection"),
]


def make_attention_sink_inputs(H, S, TOPK, dim):
    # Keep the boundary smoke's partial chunks; CSA cases use every selected key.
    q, kv, indices = make_inputs(H, S, 2 * TOPK, TOPK, dim, lambda s: [1, 31, 65, TOPK][s] if S == 4 else TOPK)
    scale = dim**-0.5
    # Span negligible to dominant sinks in the scaled-logit domain for every head dimension.
    sink = (torch.linspace(-4, 8, H) / scale).reshape(1, 1, 1, H).to(torch.bfloat16)
    return q.to(torch.bfloat16), kv.to(torch.bfloat16), indices, sink, scale
