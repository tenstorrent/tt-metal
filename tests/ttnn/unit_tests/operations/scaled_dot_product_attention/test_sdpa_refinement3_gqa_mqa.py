# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — GQA / MQA head broadcasting tests.

Tests GQA (Grouped-Query Attention) and MQA (Multi-Query Attention)
head broadcasting directly. In GQA, H_q > H_kv > 1 with H_q % H_kv == 0.
In MQA, H_kv = 1. The K/V tensors have fewer heads than Q, and each KV
head is shared by H_q / H_kv Q heads (repeat_interleave broadcasting).
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _make_inputs(q_shape, k_shape, v_shape, device, dtype=ttnn.bfloat16):
    torch.manual_seed(42)
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    ttnn_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return Q, K, V, ttnn_Q, ttnn_K, ttnn_V


def _torch_ref(Q, K, V, attn_mask=None, scale=None):
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    H_q = Qf.shape[1]
    H_kv = Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)
    am = attn_mask.to(torch.float32) if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, scale=scale)


def _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, fp32_dest_acc_en=True, attn_mask=None, scale=None):
    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    return scaled_dot_product_attention(
        ttnn_Q,
        ttnn_K,
        ttnn_V,
        attn_mask=attn_mask,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )


# ── GQA self-attention shapes ──────────────────────────────────────────

GQA_SELF_SHAPES = [
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # 4:1
    ((1, 8, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64)),  # 4:1, longer
    ((1, 32, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)),  # Llama 3 8B
    ((1, 32, 512, 128), (1, 8, 512, 128), (1, 8, 512, 128)),  # Llama 3 long
    ((1, 12, 128, 64), (1, 4, 128, 64), (1, 4, 128, 64)),  # 3:1
    ((1, 16, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64)),  # 4:1 GPT
    ((2, 8, 128, 64), (2, 2, 128, 64), (2, 2, 128, 64)),  # 4:1 + batch
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape",
    GQA_SELF_SHAPES,
    ids=[f"GQA{q[1]}:{k[1]}_S{q[2]}_D{q[3]}" for q, k, v in GQA_SELF_SHAPES],
)
def test_gqa_self_attention(device, q_shape, k_shape, v_shape):
    """GQA self-attention: H_q > H_kv > 1, H_q % H_kv == 0."""
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── MQA self-attention shapes ──────────────────────────────────────────

MQA_SELF_SHAPES = [
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # basic
    ((1, 12, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # BERT-ish H
    ((1, 16, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),  # longer
    ((1, 32, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128)),  # large model
    ((2, 8, 128, 64), (2, 1, 128, 64), (2, 1, 128, 64)),  # + batch
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape", MQA_SELF_SHAPES, ids=[f"MQA{q[1]}:1_S{q[2]}_D{q[3]}" for q, k, v in MQA_SELF_SHAPES]
)
def test_mqa_self_attention(device, q_shape, k_shape, v_shape):
    """MQA self-attention: H_kv = 1, all Q heads share one KV head."""
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── MQA with H_q > num_cores (multi-work-unit per core) ────────────────


def test_mqa_hq_71_falcon7b(device):
    """MQA with H_q=71 (Falcon-7B config). 71 work units on 56 cores."""
    q_shape = (1, 71, 2048, 64)
    k_shape = (1, 1, 2048, 64)
    v_shape = (1, 1, 2048, 64)
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


def test_mqa_hq_64(device):
    """MQA with H_q=64, just over 56-core grid."""
    q_shape = (1, 64, 128, 64)
    k_shape = (1, 1, 128, 64)
    v_shape = (1, 1, 128, 64)
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── GQA/MQA cross-attention ────────────────────────────────────────────

GQA_MQA_CROSS_SHAPES = [
    ((1, 8, 64, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA + S_q < S_kv
    ((1, 8, 64, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA + S_q < S_kv
    ((1, 32, 128, 128), (1, 8, 512, 128), (1, 8, 512, 128)),  # GQA + S_q << S_kv
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape",
    GQA_MQA_CROSS_SHAPES,
    ids=[f"{'GQA' if k[1] > 1 else 'MQA'}_cross_S{q[2]}:{k[2]}" for q, k, v in GQA_MQA_CROSS_SHAPES],
)
def test_gqa_mqa_cross_attention(device, q_shape, k_shape, v_shape):
    """GQA/MQA cross-attention: S_q != S_kv + head broadcasting."""
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── GQA/MQA with mask ──────────────────────────────────────────────────

GQA_MASK_SHAPES = [
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA 4:1 + mask
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA + mask
    ((1, 8, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64)),  # GQA 4:1 long + mask
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape",
    GQA_MASK_SHAPES,
    ids=[f"{'GQA' if k[1] > 1 else 'MQA'}_mask_S{q[2]}" for q, k, v in GQA_MASK_SHAPES],
)
def test_gqa_mqa_with_mask(device, q_shape, k_shape, v_shape):
    """GQA/MQA with additive causal mask."""
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    B, _, S_q, _ = q_shape
    S_kv = k_shape[2]
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    ttnn_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expected = _torch_ref(Q, K, V, attn_mask=mask)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16, attn_mask=ttnn_mask)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── GQA/MQA with explicit scale ────────────────────────────────────────

GQA_SCALE_SHAPES = [
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA 4:1
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape",
    GQA_SCALE_SHAPES,
    ids=[f"{'GQA' if k[1] > 1 else 'MQA'}_scale" for q, k, v in GQA_SCALE_SHAPES],
)
def test_gqa_mqa_explicit_scale(device, q_shape, k_shape, v_shape):
    """GQA/MQA with explicit scale factor."""
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    scale = 0.125
    expected = _torch_ref(Q, K, V, scale=scale)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16, scale=scale)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── GQA/MQA long-context ───────────────────────────────────────────────


def test_gqa_long_context(device):
    """GQA with S=4096 (long context)."""
    q_shape = (1, 8, 4096, 128)
    k_shape = (1, 2, 4096, 128)
    v_shape = (1, 2, 4096, 128)
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


def test_mqa_long_context(device):
    """MQA with S=4096 (long context)."""
    q_shape = (1, 4, 4096, 64)
    k_shape = (1, 1, 4096, 64)
    v_shape = (1, 1, 4096, 64)
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)


# ── GQA/MQA with different dtypes ───────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_gqa_dtype(device, dtype):
    """GQA across all supported dtypes."""
    q_shape = (1, 8, 128, 64)
    k_shape = (1, 2, 128, 64)
    v_shape = (1, 2, 128, 64)
    fp32_acc = dtype == ttnn.float32  # fp32 requires fp32_dest_acc_en
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(q_shape, k_shape, v_shape, device, dtype)
    expected = _torch_ref(Q, K, V)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, fp32_dest_acc_en=fp32_acc)
    out = ttnn.to_torch(result)
    pcc_target = 0.999 if dtype == ttnn.float32 else 0.99
    assert_with_pcc(out.float(), expected.float(), pcc_target)


# ── Sequential GQA/MQA (no state leak) ─────────────────────────────────


def test_gqa_mqa_sequential(device):
    """Run GQA then MQA sequentially — verify no state leak between calls."""
    # First: GQA
    q1, k1, v1 = (1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)
    Q1, K1, V1, tQ1, tK1, tV1 = _make_inputs(q1, k1, v1, device)
    exp1 = _torch_ref(Q1, K1, V1)
    res1 = _run_sdpa(tQ1, tK1, tV1, device, ttnn.bfloat16)
    out1 = ttnn.to_torch(res1)
    assert_with_pcc(out1.float(), exp1.float(), 0.995)

    # Second: MQA
    q2, k2, v2 = (1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)
    Q2, K2, V2, tQ2, tK2, tV2 = _make_inputs(q2, k2, v2, device)
    exp2 = _torch_ref(Q2, K2, V2)
    res2 = _run_sdpa(tQ2, tK2, tV2, device, ttnn.bfloat16)
    out2 = ttnn.to_torch(res2)
    assert_with_pcc(out2.float(), exp2.float(), 0.995)


# ── Deterministic all-ones input ────────────────────────────────────────


def test_gqa_all_ones(device):
    """GQA with all-ones input: every intermediate is hand-calculable."""
    q_shape = (1, 8, 128, 64)
    k_shape = (1, 2, 128, 64)
    v_shape = (1, 2, 128, 64)
    Q = torch.ones(q_shape, dtype=torch.bfloat16)
    K = torch.ones(k_shape, dtype=torch.bfloat16)
    V = torch.ones(v_shape, dtype=torch.bfloat16)
    expected = _torch_ref(Q, K, V)
    ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, ttnn.bfloat16)
    out = ttnn.to_torch(result)
    assert_with_pcc(out.float(), expected.float(), 0.995)
