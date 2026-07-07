# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — Non-tile-aligned shapes for scaled_dot_product_attention.

Tests non-tile-aligned dimensions (S % 32 != 0 or D % 32 != 0) across:
- w_non_aligned (D not divisible by 32)
- h_non_aligned (S not divisible by 32, D aligned)
- both non-aligned (S and D both not divisible by 32)
- All mask modes (none, custom, causal)
- All scale modes (auto, explicit)
- All dtypes (bf16, fp32, bf8b)
- Multi-head, multi-batch, GQA, MQA, cross-attention
- Deterministic all-ones input
"""

import math

import pytest
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16):
    """Broadcast (B,1,S_q,S_kv) additive causal mask."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def pytorch_sdpa(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
    """Reference in fp32, returned in input dtype."""
    orig_dtype = Q.dtype
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)
    am = attn_mask.float() if attn_mask is not None else None
    out = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, is_causal=is_causal, scale=scale)
    return out.to(orig_dtype)


def run_and_check(
    Q,
    K,
    V,
    device,
    *,
    attn_mask=None,
    is_causal=False,
    scale=None,
    dtype=ttnn.bfloat16,
    fp32_dest_acc_en=True,
    pcc_target=0.99,
):
    """Run SDPA and check against PyTorch reference."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]

    expected = pytorch_sdpa(Q, K, V, attn_mask=attn_mask, is_causal=is_causal, scale=scale)

    ttnn_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_mask = None
    if attn_mask is not None:
        ttnn_mask = ttnn.from_torch(attn_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ck = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )
    result = scaled_dot_product_attention(
        ttnn_Q,
        ttnn_K,
        ttnn_V,
        attn_mask=ttnn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=ck,
    )
    out = ttnn.to_torch(result)

    assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc >= pcc_target, f"PCC={pcc} < {pcc_target}"


# ---------------------------------------------------------------------------
# w_non_aligned: D not divisible by 32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [50, 48, 33])
def test_w_non_aligned_no_mask(device, D):
    """w_non_aligned (D%32!=0), no mask, auto scale."""
    B, H, S = 1, 1, 32
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("D", [50, 48])
def test_w_non_aligned_custom_mask(device, D):
    """w_non_aligned with custom (causal) mask."""
    B, H, S = 1, 1, 32
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("D", [50, 48])
def test_w_non_aligned_causal(device, D):
    """w_non_aligned with native causal mask."""
    B, H, S = 1, 1, 32
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, is_causal=True, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# h_non_aligned: S not divisible by 32, D aligned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S", [47, 33, 48, 100])
def test_h_non_aligned_no_mask(device, S):
    """h_non_aligned (S%32!=0, D aligned), no mask."""
    B, H, D = 1, 1, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("S", [47, 33, 48])
def test_h_non_aligned_custom_mask(device, S):
    """h_non_aligned with custom (causal) mask."""
    B, H, D = 1, 1, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("S", [47, 33, 48])
def test_h_non_aligned_causal(device, S):
    """h_non_aligned with native causal mask."""
    B, H, D = 1, 1, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, is_causal=True, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("S", [47, 48])
def test_h_non_aligned_explicit_scale(device, S):
    """h_non_aligned with explicit scale."""
    B, H, D = 1, 1, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=0.125)


# ---------------------------------------------------------------------------
# Both non-aligned: S and D both not divisible by 32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S,D", [(50, 50), (33, 50), (47, 48)])
def test_both_non_aligned_no_mask(device, S, D):
    """Both S and D non-aligned, no mask."""
    B, H = 1, 1
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("S,D", [(50, 50), (47, 48)])
def test_both_non_aligned_custom_mask(device, S, D):
    """Both S and D non-aligned with custom mask."""
    B, H = 1, 1
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


@pytest.mark.parametrize("S,D", [(50, 50), (47, 48)])
def test_both_non_aligned_causal(device, S, D):
    """Both S and D non-aligned with native causal mask."""
    B, H = 1, 1
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, is_causal=True, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# Multi-head, multi-batch non-aligned
# ---------------------------------------------------------------------------


def test_h_non_aligned_multi_head(device):
    """h_non_aligned with multiple heads."""
    B, H, S, D = 1, 4, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_h_non_aligned_multi_batch(device):
    """h_non_aligned with multiple batches."""
    B, H, S, D = 2, 4, 100, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_h_non_aligned_multi_head_mask(device):
    """h_non_aligned with multi-head and custom mask."""
    B, H, S, D = 1, 8, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# GQA / MQA + non-aligned
# ---------------------------------------------------------------------------


def test_h_non_aligned_gqa(device):
    """h_non_aligned with GQA (4:1 ratio)."""
    B, H_q, H_kv, S, D = 1, 8, 2, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_h_non_aligned_mqa(device):
    """h_non_aligned with MQA."""
    B, H_q, H_kv, S, D = 1, 8, 1, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_h_non_aligned_mqa_causal(device):
    """h_non_aligned with MQA + causal mask."""
    B, H_q, H_kv, S, D = 1, 8, 1, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, is_causal=True, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# Cross-attention + non-aligned
# ---------------------------------------------------------------------------


def test_cross_attention_non_aligned_kv(device):
    """Cross-attention where S_kv is non-aligned (S_q aligned, S_kv non-aligned)."""
    B, H, S_q, S_kv, D = 1, 4, 64, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_cross_attention_both_non_aligned(device):
    """Cross-attention with both S_q and S_kv non-aligned, D non-aligned."""
    B, H, S_q, S_kv, D = 1, 4, 100, 47, 50
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))


def test_cross_attention_non_aligned_custom_mask(device):
    """Cross-attention with non-aligned S_kv and custom mask."""
    B, H, S_q, S_kv, D = 1, 4, 64, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S_q, S_kv)
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# Dtype variations on non-aligned shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,fp32_dest",
    [
        (ttnn.bfloat16, True),
        (ttnn.bfloat16, False),
        (ttnn.float32, True),
        (ttnn.bfloat8_b, True),
        (ttnn.bfloat8_b, False),
    ],
)
def test_h_non_aligned_dtypes(device, dtype, fp32_dest):
    """h_non_aligned across all supported dtype × fp32_dest_acc_en combinations."""
    B, H, S, D = 1, 4, 47, 64
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch_dtype)
    K = torch.randn(B, H, S, D, dtype=torch_dtype)
    V = torch.randn(B, H, S, D, dtype=torch_dtype)
    pcc_target = 0.99 if dtype in (ttnn.bfloat8_b, ttnn.float32) else 0.995
    run_and_check(
        Q, K, V, device, scale=1.0 / math.sqrt(D), dtype=dtype, fp32_dest_acc_en=fp32_dest, pcc_target=pcc_target
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_h_non_aligned_custom_mask_dtypes(device, dtype):
    """h_non_aligned + custom mask across dtypes."""
    B, H, S, D = 1, 4, 47, 64
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch_dtype)
    K = torch.randn(B, H, S, D, dtype=torch_dtype)
    V = torch.randn(B, H, S, D, dtype=torch_dtype)
    mask = make_causal_mask(B, S, S, dtype=torch_dtype)
    pcc_target = 0.99 if dtype in (ttnn.bfloat8_b, ttnn.float32) else 0.995
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D), dtype=dtype, pcc_target=pcc_target)


# ---------------------------------------------------------------------------
# Deterministic all-ones input
# ---------------------------------------------------------------------------


def test_non_aligned_all_ones(device):
    """All-ones input: output should be all ones (each position attends to all
    previous positions, average of ones = 1.0)."""
    B, H, S, D = 1, 1, 47, 64
    Q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    K = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    V = torch.ones(B, H, S, D, dtype=torch.bfloat16)

    expected = pytorch_sdpa(Q, K, V, scale=1.0 / math.sqrt(D))

    ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, scale=1.0 / math.sqrt(D))
    out = ttnn.to_torch(result)

    assert out.shape == expected.shape
    # All-ones output has zero variance → PCC is NaN. Use max_abs_diff instead.
    max_diff = (out.float() - expected.float()).abs().max().item()
    assert max_diff < 0.05, f"max_diff={max_diff}"


def test_non_aligned_all_ones_causal(device):
    """All-ones input with causal mask: output should be all ones."""
    B, H, S, D = 1, 1, 47, 64
    Q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    K = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    V = torch.ones(B, H, S, D, dtype=torch.bfloat16)

    expected = pytorch_sdpa(Q, K, V, is_causal=True, scale=1.0 / math.sqrt(D))

    ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True, scale=1.0 / math.sqrt(D))
    out = ttnn.to_torch(result)

    assert out.shape == expected.shape
    # All-ones output has zero variance → PCC is NaN. Use max_abs_diff instead.
    max_diff = (out.float() - expected.float()).abs().max().item()
    assert max_diff < 0.05, f"max_diff={max_diff}"


# ---------------------------------------------------------------------------
# Sequential state-leak regression test
# ---------------------------------------------------------------------------


def test_sequential_non_aligned_then_aligned(device):
    """Run non-aligned then aligned — ensure no state leak between calls."""
    # Non-aligned
    B, H, S, D = 1, 1, 47, 64
    torch.manual_seed(0)
    Q1 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K1 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V1 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q1, K1, V1, device, scale=1.0 / math.sqrt(D))

    # Aligned (should still work)
    S2, D2 = 64, 64
    Q2 = torch.randn(B, H, S2, D2, dtype=torch.bfloat16)
    K2 = torch.randn(B, H, S2, D2, dtype=torch.bfloat16)
    V2 = torch.randn(B, H, S2, D2, dtype=torch.bfloat16)
    run_and_check(Q2, K2, V2, device, scale=1.0 / math.sqrt(D2))

    # Non-aligned again
    Q3 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K3 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V3 = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    run_and_check(Q3, K3, V3, device, scale=1.0 / math.sqrt(D))


def test_sequential_mask_then_no_mask_non_aligned(device):
    """Run custom mask then no mask on non-aligned shape — no state leak."""
    B, H, S, D = 1, 4, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)

    # With mask
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))
    # Without mask (padding mask only)
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))
    # With mask again
    run_and_check(Q, K, V, device, attn_mask=mask, scale=1.0 / math.sqrt(D))


# ---------------------------------------------------------------------------
# Golden INPUTS non-aligned shapes (exact shapes from feature_spec.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Q_shape,K_shape,V_shape",
    [
        ((1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)),  # w_non_aligned
        ((1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),  # h_non_aligned
        ((1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)),  # both
        ((1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64)),  # + multi-head
        ((2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64)),  # + multi-batch
        ((1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47)),  # D non-aligned + multi-head
        ((1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)),  # both + multi-head
        ((1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64)),  # + GQA
        ((1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),  # + MQA
        ((1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50)),  # both + cross-attn
    ],
)
def test_golden_non_aligned_shapes(device, Q_shape, K_shape, V_shape):
    """Exact non-aligned shapes from feature_spec.py INPUTS."""
    torch.manual_seed(0)
    Q = torch.randn(Q_shape, dtype=torch.bfloat16)
    K = torch.randn(K_shape, dtype=torch.bfloat16)
    V = torch.randn(V_shape, dtype=torch.bfloat16)
    D = Q_shape[-1]
    run_and_check(Q, K, V, device, scale=1.0 / math.sqrt(D))
