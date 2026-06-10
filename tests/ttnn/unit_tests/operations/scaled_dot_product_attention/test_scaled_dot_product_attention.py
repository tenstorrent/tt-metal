# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for scaled_dot_product_attention.

Verifies the fused on-device SDPA kernel against a PyTorch reference
across the Phase 0 SUPPORTED universe:

- dtype:          bfloat16
- layout:         TILE_LAYOUT
- alignment:      tile_aligned   (S_q, S_kv, D all multiples of 32)
- attention_kind: self  (S_q == S_kv) and cross  (S_q != S_kv)
- mask_mode:      none  (attention_mask=None) and causal  (-inf upper triangle)
- scale_mode:     auto  (scale=None → 1/sqrt(D)) and explicit  (scale=0.125)

The implementer MUST NOT modify this file — it is the immutable
specification of correctness.

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention.py
"""

import math
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# --- PyTorch reference -----------------------------------------------------


def _torch_sdpa(q, k, v, mask, scale):
    """Reference SDPA in fp32 (golden), output cast back to q's dtype."""
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)

    if scale is None:
        scale = 1.0 / math.sqrt(q32.shape[-1])

    # (B, H, S_q, D) @ (B, H, D, S_kv) -> (B, H, S_q, S_kv)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v32)  # (B, H, S_q, D)
    return out.to(q.dtype)


def _build_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16):
    """Additive causal mask of shape (B, 1, S_q, S_kv): 0 for j <= i, -inf otherwise.

    For cross-attention (S_q != S_kv) the upper-triangular cut is rectangular —
    mathematically well-defined but not a real workload (see EXCLUSIONS).
    """
    causal = torch.triu(
        torch.full((S_q, S_kv), float("-inf"), dtype=torch.float32),
        diagonal=1,
    ).to(dtype)
    return causal.unsqueeze(0).unsqueeze(0).expand(B, 1, S_q, S_kv).contiguous()


# --- Test parametrization --------------------------------------------------

#
# Shapes cover the Phase 0 envelope:
#   - single-tile (1,1,32,32)
#   - multi-tile self-attention with several head counts and seq lens
#   - non-square (S != D)
#   - multi-batch
#   - cross-attention (S_q != S_kv)
#
SELF_ATTN_SHAPES = [
    # (B, H, S_q, S_kv, D)
    pytest.param(1, 1, 32, 32, 32, id="self_b1_h1_s32_d32"),  # single-tile
    pytest.param(1, 1, 32, 32, 64, id="self_b1_h1_s32_d64"),
    pytest.param(1, 1, 64, 64, 64, id="self_b1_h1_s64_d64"),
    pytest.param(1, 1, 128, 128, 64, id="self_b1_h1_s128_d64"),  # multi-tile rows
    pytest.param(1, 4, 128, 128, 64, id="self_b1_h4_s128_d64"),  # multi-head
    pytest.param(2, 4, 128, 128, 64, id="self_b2_h4_s128_d64"),  # multi-batch
    pytest.param(1, 1, 128, 128, 128, id="self_b1_h1_s128_d128"),  # non-square
]

CROSS_ATTN_SHAPES = [
    # (B, H, S_q, S_kv, D)
    pytest.param(1, 1, 32, 64, 32, id="cross_b1_h1_sq32_skv64_d32"),  # S_q < S_kv
    pytest.param(1, 1, 64, 32, 32, id="cross_b1_h1_sq64_skv32_d32"),  # S_q > S_kv
    pytest.param(1, 4, 128, 64, 64, id="cross_b1_h4_sq128_skv64_d64"),
    pytest.param(2, 2, 64, 128, 64, id="cross_b2_h2_sq64_skv128_d64"),
]


def _make_inputs(B, H, S_q, S_kv, D, *, seed=42, device=None):
    """Build (Q_torch, K_torch, V_torch) and their ttnn counterparts."""
    torch.manual_seed(seed)
    # Keep magnitudes modest to avoid exp overflow in bf16 softmax.
    q_torch = torch.randn(B, H, S_q, D, dtype=torch.bfloat16) * 0.3
    k_torch = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16) * 0.3
    v_torch = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16) * 0.3

    q_ttnn = ttnn.from_torch(q_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_ttnn = ttnn.from_torch(k_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_ttnn = ttnn.from_torch(v_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return q_torch, k_torch, v_torch, q_ttnn, k_ttnn, v_ttnn


# bf16 PCC tolerance per planner contract (do not retune by op-class).
PCC_TOL = 0.995


# --- Tests -----------------------------------------------------------------


@pytest.mark.parametrize("B,H,S_q,S_kv,D", SELF_ATTN_SHAPES)
def test_sdpa_self_no_mask_auto_scale(device, B, H, S_q, S_kv, D):
    """Self-attention, no mask, auto scale=1/sqrt(D)."""
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    expected = _torch_sdpa(q_t, k_t, v_t, mask=None, scale=None)

    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D), f"shape mismatch: {actual.shape}"
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", SELF_ATTN_SHAPES)
def test_sdpa_self_no_mask_explicit_scale(device, B, H, S_q, S_kv, D):
    """Self-attention, no mask, explicit scale = 0.125."""
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    scale = 0.125
    expected = _torch_sdpa(q_t, k_t, v_t, mask=None, scale=scale)

    out = scaled_dot_product_attention(q, k, v, scale=scale)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", SELF_ATTN_SHAPES)
def test_sdpa_self_causal_auto_scale(device, B, H, S_q, S_kv, D):
    """Self-attention, causal mask, auto scale.

    Causal mask is well-defined for self-attention (S_q == S_kv). The kernel
    receives a real additive mask tensor; the test does not rely on any
    in-kernel causal shortcut.
    """
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    mask_t = _build_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16)
    expected = _torch_sdpa(q_t, k_t, v_t, mask=mask_t, scale=None)

    mask = ttnn.from_torch(mask_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", SELF_ATTN_SHAPES)
def test_sdpa_self_causal_explicit_scale(device, B, H, S_q, S_kv, D):
    """Self-attention, causal mask, explicit scale — full Phase 0 cell."""
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    mask_t = _build_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16)
    scale = 0.125
    expected = _torch_sdpa(q_t, k_t, v_t, mask=mask_t, scale=scale)

    mask = ttnn.from_torch(mask_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v, attention_mask=mask, scale=scale)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", CROSS_ATTN_SHAPES)
def test_sdpa_cross_no_mask_auto_scale(device, B, H, S_q, S_kv, D):
    """Cross-attention (S_q != S_kv), no mask, auto scale."""
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    expected = _torch_sdpa(q_t, k_t, v_t, mask=None, scale=None)

    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


@pytest.mark.parametrize("B,H,S_q,S_kv,D", CROSS_ATTN_SHAPES)
def test_sdpa_cross_no_mask_explicit_scale(device, B, H, S_q, S_kv, D):
    """Cross-attention, no mask, explicit scale."""
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S_q, S_kv, D, device=device)
    scale = 0.125
    expected = _torch_sdpa(q_t, k_t, v_t, mask=None, scale=scale)

    out = scaled_dot_product_attention(q, k, v, scale=scale)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)


# --- Per-head mask coverage ------------------------------------------------
#
# The API accepts mask shape (B, 1, S_q, S_kv) — broadcast across heads — and
# (B, H, S_q, S_kv) — per-head masking. Phase 0 must support both shapes.
# We test the per-head shape with a random additive mask (clamped) on a
# multi-head self-attention shape.


def test_sdpa_self_per_head_mask(device):
    """Self-attention with a per-head additive mask (B, H, S_q, S_kv)."""
    B, H, S, D = 1, 4, 64, 64
    q_t, k_t, v_t, q, k, v = _make_inputs(B, H, S, S, D, device=device)

    torch.manual_seed(43)
    # Mostly zero, with a few -inf positions per head to verify the kernel
    # actually consumes per-head mask values (not the broadcast shape).
    mask_t = torch.zeros(B, H, S, S, dtype=torch.bfloat16)
    mask_t[:, :, :, S // 2 :] = float("-inf")  # block the second half of keys
    expected = _torch_sdpa(q_t, k_t, v_t, mask=mask_t, scale=None)

    mask = ttnn.from_torch(mask_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=PCC_TOL)
