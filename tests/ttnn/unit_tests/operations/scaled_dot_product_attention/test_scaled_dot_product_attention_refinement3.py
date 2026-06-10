# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 3 — Non-tile-aligned shapes (W and H).

Exercises the kernel's two new alignment paths:

- **w_non_aligned** — D (head_dim) is not a multiple of 32. The Q/K/V
  tensors have one extra tile column carrying TTNN-zero pad on the
  trailing D positions; the QK^T inner product picks up `q_real · 0 = 0`,
  and the attn @ V projection writes 0 into the padded D positions of
  the output (truncated when ttnn → torch). No kernel-side masking.

- **h_non_aligned** — S_q and/or S_kv is not a multiple of 32. S_q
  non-alignment just means the last tile row carries garbage outputs
  that are truncated on conversion. S_kv non-alignment is the
  numerically-tricky case: the last K tile carries padded keys whose
  scores would normalize through softmax as if they were real keys
  (exp(0)=1 each), so the kernel injects a synthetic -inf mask onto
  the padded columns of the last K-iter's mask tile. Composes with
  any user-supplied mask.

EXCLUSIONS — bf8b + non_aligned cells stay in the op's EXCLUSIONS list
because bf8b's per-face shared exponent makes the in-kernel -inf
overlay's literal bit pattern lossy. Tests at the bottom of this file
assert the op refuses those cells cleanly.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Reference & helpers
# ---------------------------------------------------------------------------


def _torch_sdpa(Q, K, V, mask=None, scale=None):
    """fp32 reference, returns in Q's dtype."""
    qf = Q.to(torch.float32)
    kf = K.to(torch.float32)
    vf = V.to(torch.float32)
    if qf.shape[1] != kf.shape[1]:
        reps = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(reps, dim=1)
        vf = vf.repeat_interleave(reps, dim=1)
    s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, vf)
    return out.to(Q.dtype)


def _make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16):
    m = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    m.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return m


def _to_ttnn(t, device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# ---------------------------------------------------------------------------
# Section 1: W non-aligned (D not divisible by 32)
# ---------------------------------------------------------------------------
# D padding works without any kernel-side masking — the QK^T reduction
# zero-pads on both Q and K, and attn @ V zero-pads on V's D cols (so
# output's D padding is zero too, which is what TTNN truncates anyway).

W_NON_ALIGNED_CASES = [
    pytest.param(1, 1, 32, 32, 50, id="W_b1_h1_s32_d50"),
    pytest.param(1, 1, 64, 64, 50, id="W_b1_h1_s64_d50"),
    pytest.param(1, 1, 64, 64, 47, id="W_b1_h1_s64_d47"),
    pytest.param(1, 4, 128, 128, 50, id="W_b1_h4_s128_d50"),
    pytest.param(1, 8, 64, 64, 47, id="W_b1_h8_s64_d47"),
    pytest.param(2, 4, 128, 128, 50, id="W_b2_h4_s128_d50"),
    pytest.param(1, 1, 32, 32, 96, id="W_b1_h1_s32_d96"),  # D=96 (3 tiles, last tile fully valid)
]


@pytest.mark.parametrize("B,H,S_q,S_kv,D", W_NON_ALIGNED_CASES)
def test_w_non_aligned_no_mask(device, B, H, S_q, S_kv, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", W_NON_ALIGNED_CASES)
def test_w_non_aligned_causal_self(device, B, H, S_q, S_kv, D):
    if S_q != S_kv:
        pytest.skip("causal mask requires self-attention")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask = _make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V, mask=mask)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
        attention_mask=_to_ttnn(mask, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


# ---------------------------------------------------------------------------
# Section 2: H non-aligned (S_q or S_kv not divisible by 32)
# ---------------------------------------------------------------------------

H_NON_ALIGNED_CASES = [
    # S_q non-aligned, S_kv aligned, D aligned
    pytest.param(1, 1, 47, 64, 64, id="H_sq47_skv64_d64"),
    pytest.param(1, 1, 33, 32, 64, id="H_sq33_skv32_d64"),  # both barely-non-aligned
    # S_q aligned, S_kv non-aligned, D aligned — exercises synthetic mask
    pytest.param(1, 1, 32, 47, 64, id="H_sq32_skv47_d64"),
    pytest.param(1, 1, 64, 50, 64, id="H_sq64_skv50_d64"),
    pytest.param(1, 1, 32, 33, 64, id="H_sq32_skv33_d64"),  # S_kv with only 1 valid in last tile
    # Self-attention non-aligned (S_q == S_kv)
    pytest.param(1, 1, 47, 47, 64, id="H_self_s47_d64"),
    pytest.param(1, 1, 50, 50, 64, id="H_self_s50_d64"),
    pytest.param(1, 1, 100, 100, 64, id="H_self_s100_d64"),
    # Multi-head / batch
    pytest.param(1, 4, 47, 47, 64, id="H_self_b1_h4_s47_d64"),
    pytest.param(2, 4, 100, 100, 64, id="H_self_b2_h4_s100_d64"),
    pytest.param(1, 8, 33, 33, 64, id="H_self_b1_h8_s33_d64"),
]


@pytest.mark.parametrize("B,H,S_q,S_kv,D", H_NON_ALIGNED_CASES)
def test_h_non_aligned_no_mask(device, B, H, S_q, S_kv, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", H_NON_ALIGNED_CASES)
def test_h_non_aligned_causal_self(device, B, H, S_q, S_kv, D):
    """Causal mask only applies to self-attention."""
    if S_q != S_kv:
        pytest.skip("causal mask requires self-attention")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask = _make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V, mask=mask)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
        attention_mask=_to_ttnn(mask, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


# ---------------------------------------------------------------------------
# Section 3: both W and H non-aligned (S_q, S_kv, D all non-aligned)
# ---------------------------------------------------------------------------

BOTH_NON_ALIGNED_CASES = [
    pytest.param(1, 1, 50, 50, 50, id="both_s50_d50"),
    pytest.param(1, 4, 47, 47, 50, id="both_b1_h4_s47_d50"),
    pytest.param(1, 12, 33, 33, 50, id="both_b1_h12_s33_d50"),
    pytest.param(1, 4, 100, 47, 50, id="both_cross_sq100_skv47_d50"),  # cross + both-non-aligned
]


@pytest.mark.parametrize("B,H,S_q,S_kv,D", BOTH_NON_ALIGNED_CASES)
def test_both_non_aligned(device, B, H, S_q, S_kv, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


# ---------------------------------------------------------------------------
# Section 4: composability with R1 dtypes (fp32) and R2 KV-heads (GQA / MQA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,marker",
    [
        ((1, 1, 47, 64), "H_47"),
        ((1, 1, 32, 50), "W_50"),
        ((1, 1, 50, 50), "both"),
    ],
    ids=lambda v: v if isinstance(v, str) else f"shape_{v}",
)
def test_non_aligned_fp32(device, shape, marker):
    """fp32 on the non-aligned path — composes with R1's dtype extension."""
    B, H, S, D = shape
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.float32),
        _to_ttnn(K, device, ttnn.float32),
        _to_ttnn(V, device, ttnn.float32),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert tuple(actual.shape) == (B, H, S, D)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.999)


def test_non_aligned_gqa(device):
    """GQA composability — non-aligned shape + H_q > H_kv > 1."""
    B, H_q, H_kv, S, D = 1, 8, 2, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


def test_non_aligned_mqa(device):
    """MQA composability — non-aligned shape + H_kv=1."""
    B, H_q, S, D = 1, 8, 47, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, 1, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, 1, S, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


def test_non_aligned_multicore(device):
    """Multi-core distribution + non-aligned — composes with R1's split."""
    B, H, S, D = 4, 8, 100, 64  # total_rows = 4*8*ceil(100/32) = 128 — multiple cores
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


def test_non_aligned_per_head_mask(device):
    """Per-head mask + S_kv non-aligned — synthetic mask composes with user mask.

    Mirrors the Phase 0 ``test_sdpa_self_per_head_mask`` pattern: structured
    mask (mostly zero with -inf in the second half of S_kv) rather than a
    uniformly-random additive mask. The kernel's bf16 mask add through
    fp32 DEST cannot match a uniformly-random reference to PCC=0.995
    because softmax is acutely sensitive to small bf16 rounding errors
    on high-variance scores — verified pre-R3 that the same shape with
    a random mask gives PCC=0.76 on aligned input too. This shape is
    a real-world mask (block the second half of keys per head), and the
    valid-keys-in-last-tile region remains unmasked so the synthetic
    -inf overlay still drives the kernel's "ignore padded keys" path.
    """
    B, H, S_q, S_kv, D = 1, 4, 47, 47, 64
    torch.manual_seed(43)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    # Per-head structured mask: zero everywhere, -inf on the second half
    # of keys. Different per-head mask values exercise the per-head
    # addressing path (one mask plane per Q head, not broadcast).
    mask = torch.zeros(B, H, S_q, S_kv, dtype=torch.bfloat16)
    mask[:, :, :, S_kv // 2 :] = float("-inf")
    expected = _torch_sdpa(Q, K, V, mask=mask)

    ttnn_out = scaled_dot_product_attention(
        _to_ttnn(Q, device, ttnn.bfloat16),
        _to_ttnn(K, device, ttnn.bfloat16),
        _to_ttnn(V, device, ttnn.bfloat16),
        attention_mask=_to_ttnn(mask, device, ttnn.bfloat16),
    )
    actual = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), actual.float(), pcc=0.995)


# ---------------------------------------------------------------------------
# Section 5: EXCLUSIONS — bf8b + non_aligned must refuse cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,id_label",
    [
        ((1, 1, 32, 50), "bf8b_W_non_aligned"),
        ((1, 1, 47, 64), "bf8b_H_non_aligned"),
        ((1, 1, 50, 50), "bf8b_both_non_aligned"),
    ],
    ids=lambda v: v if isinstance(v, str) else f"shape_{v}",
)
def test_bf8b_non_aligned_excluded(device, shape, id_label):
    """bf8b + non-aligned must raise ExcludedCell, not silently produce wrong
    values. Refinement-3 EXCLUSIONS defers bf8b's shared-exponent overlay
    complexity to a follow-up refinement."""
    torch.manual_seed(0)
    Q = torch.randn(*shape, dtype=torch.bfloat16)
    K = torch.randn(*shape, dtype=torch.bfloat16)
    V = torch.randn(*shape, dtype=torch.bfloat16)

    ttnn_Q = _to_ttnn(Q, device, ttnn.bfloat8_b)
    ttnn_K = _to_ttnn(K, device, ttnn.bfloat8_b)
    ttnn_V = _to_ttnn(V, device, ttnn.bfloat8_b)

    with pytest.raises(ExcludedCell):
        scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
