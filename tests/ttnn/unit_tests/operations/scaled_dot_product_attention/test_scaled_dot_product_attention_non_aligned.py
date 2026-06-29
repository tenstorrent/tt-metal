# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for non-tile-aligned shape support (Refinement 2).

Tests all three alignment categories:
- w_non_aligned: D not divisible by 32
- h_non_aligned: D aligned, S_q not divisible by 32
- both non-aligned

Covers the golden test INPUTS non-aligned shapes plus additional edge cases.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c**2).sum()) * torch.sqrt((b_c**2).sum())
    return (num / den).item() if den > 0 else 1.0


def _run_sdpa(Q, K, V, attn_mask=None, scale=None, device=None):
    """Run SDPA and return torch output."""
    tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tM = None
    if attn_mask is not None:
        tM = ttnn.from_torch(attn_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tQ, tK, tV, attn_mask=tM, scale=scale)
    return ttnn.to_torch(out)


def _reference(Q, K, V, attn_mask=None, scale=None):
    """PyTorch reference in fp32. Handles GQA/MQA head broadcasting."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)
    D = Qf.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = Qf @ Kf.transpose(-1, -2) * scale
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return (weights @ Vf).to(torch.bfloat16)


# =============================================================================
# w_non_aligned: D not divisible by 32
# =============================================================================


@pytest.mark.parametrize(
    "q_shape",
    [
        pytest.param((1, 1, 32, 50), id="32x50"),
        pytest.param((1, 1, 64, 47), id="64x47"),
        pytest.param((1, 1, 128, 50), id="128x50"),
        pytest.param((1, 8, 64, 47), id="8h_64x47"),
    ],
)
def test_w_non_aligned(device, q_shape):
    """D not divisible by 32 — zero-padded D columns in K/V."""
    torch.manual_seed(42)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, : q_shape[2], : q_shape[3]], expected)
    assert pcc >= 0.995, f"w_non_aligned {q_shape}: PCC {pcc:.6f} < 0.995"
    assert not torch.isinf(result.float()).any(), "inf in output"
    assert not torch.isnan(result.float()).any(), "nan in output"


# =============================================================================
# h_non_aligned: S_q not divisible by 32, D aligned
# =============================================================================


@pytest.mark.parametrize(
    "q_shape",
    [
        pytest.param((1, 1, 47, 64), id="47x64"),
        pytest.param((1, 1, 33, 32), id="33x32"),
        pytest.param((1, 1, 100, 64), id="100x64"),
        pytest.param((1, 4, 47, 64), id="4h_47x64"),
        pytest.param((2, 4, 100, 64), id="2b4h_100x64"),
    ],
)
def test_h_non_aligned(device, q_shape):
    """S_q not divisible by 32 — padded Q rows stripped by to_torch."""
    torch.manual_seed(42)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, : q_shape[2], : q_shape[3]], expected)
    assert pcc >= 0.995, f"h_non_aligned {q_shape}: PCC {pcc:.6f} < 0.995"
    assert not torch.isinf(result.float()).any(), "inf in output"
    assert not torch.isnan(result.float()).any(), "nan in output"


# =============================================================================
# Both non-aligned
# =============================================================================


@pytest.mark.parametrize(
    "q_shape",
    [
        pytest.param((1, 1, 47, 50), id="47x50"),
        pytest.param((1, 1, 33, 50), id="33x50"),
        pytest.param((1, 12, 33, 50), id="12h_33x50"),
    ],
)
def test_both_non_aligned(device, q_shape):
    """Both S_q and D not divisible by 32."""
    torch.manual_seed(42)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, : q_shape[2], : q_shape[3]], expected)
    assert pcc >= 0.995, f"both_non_aligned {q_shape}: PCC {pcc:.6f} < 0.995"
    assert not torch.isinf(result.float()).any(), "inf in output"
    assert not torch.isnan(result.float()).any(), "nan in output"


# =============================================================================
# Non-aligned + mask
# =============================================================================


def test_h_non_aligned_with_mask(device):
    """S_q non-aligned + custom mask — padding mask combined with user mask."""
    torch.manual_seed(42)
    q_shape = (1, 1, 47, 64)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    # Causal mask
    mask = torch.zeros(1, 1, 47, 47, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(47, 47, dtype=torch.bool), diagonal=1), float("-inf"))
    result = _run_sdpa(Q, K, V, attn_mask=mask, device=device)
    expected = _reference(Q, K, V, attn_mask=mask)
    pcc = _pcc(result[:, :, :47, :64], expected)
    assert pcc >= 0.995, f"h_non_aligned + mask: PCC {pcc:.6f} < 0.995"
    assert not torch.isinf(result.float()).any(), "inf in output"


def test_both_non_aligned_with_mask(device):
    """Both non-aligned + custom mask."""
    torch.manual_seed(42)
    q_shape = (1, 1, 47, 50)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 47, 47, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(47, 47, dtype=torch.bool), diagonal=1), float("-inf"))
    result = _run_sdpa(Q, K, V, attn_mask=mask, device=device)
    expected = _reference(Q, K, V, attn_mask=mask)
    pcc = _pcc(result[:, :, :47, :50], expected)
    assert pcc >= 0.995, f"both_non_aligned + mask: PCC {pcc:.6f} < 0.995"
    assert not torch.isinf(result.float()).any(), "inf in output"


# =============================================================================
# Non-aligned + GQA/MQA
# =============================================================================


def test_h_non_aligned_gqa(device):
    """S_q non-aligned + GQA (4:1 ratio)."""
    torch.manual_seed(42)
    q_shape = (1, 8, 47, 64)
    k_shape = (1, 2, 47, 64)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, :47, :64], expected)
    assert pcc >= 0.995, f"h_non_aligned + GQA: PCC {pcc:.6f} < 0.995"


def test_h_non_aligned_mqa(device):
    """S_q non-aligned + MQA (H_kv=1)."""
    torch.manual_seed(42)
    q_shape = (1, 8, 47, 64)
    k_shape = (1, 1, 47, 64)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, :47, :64], expected)
    assert pcc >= 0.995, f"h_non_aligned + MQA: PCC {pcc:.6f} < 0.995"


# =============================================================================
# Non-aligned + cross-attention
# =============================================================================


def test_non_aligned_cross_attention(device):
    """Both non-aligned + cross-attention (S_q != S_kv)."""
    torch.manual_seed(42)
    q_shape = (1, 4, 100, 50)
    k_shape = (1, 4, 47, 50)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result[:, :, :100, :50], expected)
    assert pcc >= 0.995, f"non_aligned cross-attn: PCC {pcc:.6f} < 0.995"


# =============================================================================
# Non-aligned + explicit scale
# =============================================================================


def test_non_aligned_explicit_scale(device):
    """Both non-aligned + explicit scale."""
    torch.manual_seed(42)
    q_shape = (1, 1, 47, 50)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, scale=0.125, device=device)
    expected = _reference(Q, K, V, scale=0.125)
    pcc = _pcc(result[:, :, :47, :50], expected)
    assert pcc >= 0.995, f"non_aligned + explicit scale: PCC {pcc:.6f} < 0.995"


# =============================================================================
# Tile-aligned regression check (ensure non-aligned changes didn't break aligned)
# =============================================================================


def test_tile_aligned_regression(device):
    """Tile-aligned shape still works (regression check)."""
    torch.manual_seed(42)
    q_shape = (1, 1, 128, 64)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(q_shape, dtype=torch.bfloat16)
    V = torch.randn(q_shape, dtype=torch.bfloat16)
    result = _run_sdpa(Q, K, V, device=device)
    expected = _reference(Q, K, V)
    pcc = _pcc(result, expected)
    assert pcc >= 0.995, f"tile_aligned regression: PCC {pcc:.6f} < 0.995"
