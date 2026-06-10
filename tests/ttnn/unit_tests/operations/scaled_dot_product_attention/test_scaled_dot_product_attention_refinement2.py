# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 2 — KV-head broadcast (MQA + GQA).

Exercises the behavior added in Refinement 2: the reader now computes
h_kv = h_q / (H_q / H_kv) and addresses the K/V cache with H_kv heads
instead of H_q. validate() accepts kv_heads_mode in {mha, mqa, gqa},
rejects non-divisible head ratios early.

What this file covers (above and beyond the golden suite):

  * MHA non-regression: H_q == H_kv must still produce the same
    output as before Refinement 2 (the broadcast collapses to identity).
  * GQA at the standard LLM ratios (4:1 → 32:8, 8:1 → 32:4, 3:1, 2:1).
  * MQA (H_kv == 1 with H_q ∈ {4, 8, 32}).
  * Cross-attention + GQA / MQA combo.
  * Multi-batch + GQA / MQA.
  * GQA / MQA with the (B, 1, S_q, S_kv) broadcast causal mask path.
  * GQA / MQA with a per-Q-head mask (mask is Q-head-indexed
    independently of the KV broadcast — important seam to lock in).
  * GQA / MQA with explicit scale.
  * dtype dimension (bf16 + fp32) — same shapes, both dtypes pass.
  * validate() rejection of H_q % H_kv != 0 (e.g. H_q=7, H_kv=2).

Run with:
    scripts/run_safe_pytest.sh \\
      tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_refinement2.py
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from ttnn.operations._op_contract import UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Reference + helpers
# ---------------------------------------------------------------------------


def _torch_sdpa(q, k, v, mask=None, scale=None):
    """Reference SDPA computed in fp32, returned in q's dtype. Handles
    GQA/MQA via head-broadcast (repeat_interleave on the K/V head axis)
    — exactly matches what the kernel does at the index level.
    """
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    H_q = q32.shape[1]
    H_kv = k32.shape[1]
    if H_q != H_kv:
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})"
        reps = H_q // H_kv
        k32 = k32.repeat_interleave(reps, dim=1)
        v32 = v32.repeat_interleave(reps, dim=1)
    if scale is None:
        scale = 1.0 / math.sqrt(q32.shape[-1])
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v32)
    return out.to(q.dtype)


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def _pcc_threshold(dtype):
    # Matches the golden-suite TOLERANCES table: bf16 ≥ 0.995, fp32 ≥ 0.999.
    return {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}[dtype]


def _make_qkv(B, H_q, H_kv, Sq, Skv, D, dtype, seed=42, scale=0.3):
    """Build (Q, K, V) with separate H_q and H_kv. K and V share H_kv."""
    torch.manual_seed(seed)
    torch_dtype = _TORCH_DTYPE[dtype]
    Q = (torch.randn(B, H_q, Sq, D) * scale).to(torch_dtype)
    K = (torch.randn(B, H_kv, Skv, D) * scale).to(torch_dtype)
    V = (torch.randn(B, H_kv, Skv, D) * scale).to(torch_dtype)
    return Q, K, V


def _to_ttnn(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _run_and_check(Q, K, V, *, dtype, device, mask=None, scale=None):
    expected = _torch_sdpa(Q, K, V, mask=mask, scale=scale)
    ttnn_Q = _to_ttnn(Q, device, dtype)
    ttnn_K = _to_ttnn(K, device, dtype)
    ttnn_V = _to_ttnn(V, device, dtype)
    ttnn_mask = _to_ttnn(mask, device, dtype) if mask is not None else None
    ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attention_mask=ttnn_mask, scale=scale)
    out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(expected.float(), out.float(), pcc=_pcc_threshold(dtype))
    return out


# ---------------------------------------------------------------------------
# Test 1 — MHA non-regression (kv_group_size == 1 path)
# ---------------------------------------------------------------------------


_MHA_REGRESSION_SHAPES = [
    # (B, H, S, D)
    pytest.param(1, 1, 32, 32, id="B1_H1_S32_D32"),
    pytest.param(1, 4, 128, 64, id="B1_H4_S128_D64"),
    pytest.param(2, 8, 128, 64, id="B2_H8_S128_D64"),
]


@pytest.mark.parametrize("B,H,S,D", _MHA_REGRESSION_SHAPES)
def test_mha_non_regression(device, B, H, S, D):
    """MHA (H_q == H_kv) must still produce correct output under the
    Refinement-2 reader rewrite. kv_group_size collapses to 1, so the
    h_kv = h_q / 1 indexing should be bit-for-bit equivalent to the
    pre-refinement (b * H + h) base."""
    Q, K, V = _make_qkv(B, H, H, S, S, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device)


# ---------------------------------------------------------------------------
# Test 2 — GQA at standard LLM ratios
# ---------------------------------------------------------------------------


_GQA_CONFIGS = [
    # (B, H_q, H_kv, S, D, label)
    pytest.param(1, 8, 2, 128, 64, id="gqa_4to1_base"),
    pytest.param(1, 8, 4, 128, 64, id="gqa_2to1"),
    pytest.param(1, 12, 4, 128, 64, id="gqa_3to1"),
    pytest.param(1, 12, 3, 128, 64, id="gqa_4to1_h12"),
    pytest.param(1, 32, 8, 128, 128, id="gqa_llama3_8B_70B"),
    pytest.param(1, 32, 4, 128, 128, id="gqa_8to1_h32"),
    pytest.param(2, 8, 2, 128, 64, id="gqa_4to1_batched"),
]


@pytest.mark.parametrize("B,H_q,H_kv,S,D", _GQA_CONFIGS)
def test_gqa_self_attention(device, B, H_q, H_kv, S, D):
    """GQA self-attention — H_q > H_kv > 1, H_q % H_kv == 0. The
    reader's h_kv = h_q / (H_q / H_kv) maps consecutive groups of Q
    heads to the same K/V head."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device)


# ---------------------------------------------------------------------------
# Test 3 — MQA (H_kv == 1)
# ---------------------------------------------------------------------------


_MQA_CONFIGS = [
    # (B, H_q, S, D, label) — H_kv is always 1
    pytest.param(1, 4, 128, 64, id="mqa_h4"),
    pytest.param(1, 8, 128, 64, id="mqa_h8"),
    pytest.param(1, 12, 128, 64, id="mqa_h12"),
    pytest.param(1, 16, 256, 64, id="mqa_h16_longer"),
    pytest.param(1, 32, 128, 128, id="mqa_h32_d128"),
    pytest.param(2, 8, 128, 64, id="mqa_h8_batched"),
]


@pytest.mark.parametrize("B,H_q,S,D", _MQA_CONFIGS)
def test_mqa_self_attention(device, B, H_q, S, D):
    """MQA self-attention — H_kv == 1, every Q head reads the same
    K/V tile-row (h_kv = h_q / H_q = 0 for all h_q)."""
    Q, K, V = _make_qkv(B, H_q, 1, S, S, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device)


# ---------------------------------------------------------------------------
# Test 4 — Cross-attention + GQA / MQA
# ---------------------------------------------------------------------------


_CROSS_GQA_MQA_CONFIGS = [
    # (B, H_q, H_kv, S_q, S_kv, D, label)
    pytest.param(1, 8, 2, 64, 128, 64, id="cross_gqa_4to1_short_long"),
    pytest.param(1, 32, 8, 128, 512, 128, id="cross_gqa_llama3_long_kv"),
    pytest.param(1, 8, 1, 64, 128, 64, id="cross_mqa_short_long"),
    pytest.param(1, 16, 1, 128, 256, 64, id="cross_mqa_longer_kv"),
]


@pytest.mark.parametrize("B,H_q,H_kv,S_q,S_kv,D", _CROSS_GQA_MQA_CONFIGS)
def test_cross_attention_gqa_mqa(device, B, H_q, H_kv, S_q, S_kv, D):
    """Cross-attention (S_q != S_kv) combined with GQA / MQA. KV cache
    has H_kv heads at S_kv tokens; queries have H_q heads at S_q
    tokens. The reader must use S_kv (Kt) for the K-loop range and
    h_kv for KV addressing — checked together here because they share
    the kv_base computation."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S_q, S_kv, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device)


# ---------------------------------------------------------------------------
# Test 5 — GQA / MQA + causal mask (broadcast (B, 1, S, S))
# ---------------------------------------------------------------------------


def _make_causal_mask(B, S, dtype):
    """Broadcast (B, 1, S, S) upper-triangular -inf mask."""
    torch_dtype = _TORCH_DTYPE[dtype]
    m = torch.zeros(B, 1, S, S, dtype=torch_dtype)
    m.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


_GQA_MQA_CAUSAL_CONFIGS = [
    pytest.param(1, 8, 2, 128, 64, id="gqa_4to1_causal"),
    pytest.param(1, 8, 1, 128, 64, id="mqa_h8_causal"),
    pytest.param(1, 32, 8, 128, 128, id="gqa_llama3_causal"),
]


@pytest.mark.parametrize("B,H_q,H_kv,S,D", _GQA_MQA_CAUSAL_CONFIGS)
def test_gqa_mqa_with_causal_mask(device, B, H_q, H_kv, S, D):
    """The broadcast (B, 1, S, S) causal mask path takes the
    mask_h_stride=0 branch in the reader — it has nothing to do with
    the KV broadcast; this test locks down that the two independent
    broadcast axes (H_q→H_kv on KV, H_q→1 on mask) compose correctly."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=ttnn.bfloat16)
    mask = _make_causal_mask(B, S, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device, mask=mask)


# ---------------------------------------------------------------------------
# Test 6 — GQA + per-Q-head mask (mask is H_q-indexed, NOT H_kv-indexed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H_q,H_kv,S,D",
    [
        pytest.param(1, 8, 2, 128, 64, id="gqa_4to1_per_head_mask"),
        pytest.param(1, 8, 1, 128, 64, id="mqa_h8_per_head_mask"),
    ],
)
def test_gqa_mqa_with_per_head_mask(device, B, H_q, H_kv, S, D):
    """Mask is per-Q-head (shape (B, H_q, S, S)) — different mask plane
    for every Q head. The reader's mask_h_stride = Qt * Kt and the
    h_q used for mask_base are the key seam: GQA/MQA must NOT route
    the mask through the KV broadcast (the mask has H_q planes, not
    H_kv). This test catches the bug where someone "consistently"
    uses h_kv everywhere."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=ttnn.bfloat16)
    torch.manual_seed(7)
    # Random additive mask with ~30% -inf positions per Q-head plane.
    mask = torch.zeros(B, H_q, S, S, dtype=torch.bfloat16)
    mask.masked_fill_(torch.rand(B, H_q, S, S) < 0.3, float("-inf"))
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device, mask=mask)


# ---------------------------------------------------------------------------
# Test 7 — GQA / MQA with explicit scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H_q,H_kv,S,D",
    [
        pytest.param(1, 8, 2, 128, 64, id="gqa_4to1_explicit_scale"),
        pytest.param(1, 8, 1, 128, 64, id="mqa_h8_explicit_scale"),
    ],
)
def test_gqa_mqa_explicit_scale(device, B, H_q, H_kv, S, D):
    """Explicit scale should compose with the KV broadcast — they
    flow through independent CT-args (scale via compute kernel, h_kv
    via reader)."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device, scale=0.125)


# ---------------------------------------------------------------------------
# Test 8 — dtype dimension (bf16 + fp32) with GQA / MQA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")],
)
@pytest.mark.parametrize(
    "B,H_q,H_kv,S,D",
    [
        pytest.param(1, 8, 2, 128, 64, id="gqa_4to1"),
        pytest.param(1, 8, 1, 128, 64, id="mqa_h8"),
    ],
)
def test_gqa_mqa_dtype_matrix(device, dtype, B, H_q, H_kv, S, D):
    """Refinement 2 must compose with Refinement 1's dtype set. Skip
    bf8b because Refinement-1's documented EXCLUSIONS path doesn't
    apply here (we're tile-aligned), but the bf8b axis is exercised
    by the golden suite — no need to re-cover it case-by-case."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=dtype)
    _run_and_check(Q, K, V, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Test 9 — validate() rejects non-divisible head ratios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "H_q,H_kv",
    [
        pytest.param(7, 2, id="7q_2kv"),
        pytest.param(5, 2, id="5q_2kv"),
        pytest.param(9, 4, id="9q_4kv"),
        pytest.param(3, 2, id="3q_2kv"),
    ],
)
def test_validate_rejects_non_divisible_head_ratio(device, H_q, H_kv):
    """H_q % H_kv != 0 has no consistent KV-broadcast semantics
    (some Q heads would map to one fewer/more KV head than others).
    validate() must reject these early with UnsupportedAxisValue.
    """
    Q, K, V = _make_qkv(1, H_q, H_kv, 64, 64, 64, dtype=ttnn.bfloat16)
    ttnn_Q = _to_ttnn(Q, device, ttnn.bfloat16)
    ttnn_K = _to_ttnn(K, device, ttnn.bfloat16)
    ttnn_V = _to_ttnn(V, device, ttnn.bfloat16)
    with pytest.raises(UnsupportedAxisValue, match=r"multiple of H_kv"):
        scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)


# ---------------------------------------------------------------------------
# Test 10 — GQA / MQA + multi-core distribution (force a multi-pass split)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H_q,H_kv,S,D",
    [
        # total_rows = B * H_q * Qt — pick combos that exceed a single
        # row per core on the smallest grid.
        pytest.param(2, 32, 8, 128, 128, id="gqa_llama3_multicore_256rows"),
        pytest.param(2, 32, 1, 128, 64, id="mqa_h32_multicore_256rows"),
        pytest.param(4, 8, 2, 256, 64, id="gqa_4to1_b4_multicore_256rows"),
    ],
)
def test_gqa_mqa_multicore(device, B, H_q, H_kv, S, D):
    """Refinement 2 must compose with Refinement 1's multi-core split.
    Each core gets a contiguous slice of (b, h_q, qt) — the per-row
    decode must produce the same h_kv it would have produced if every
    core were sequential."""
    Q, K, V = _make_qkv(B, H_q, H_kv, S, S, D, dtype=ttnn.bfloat16)
    _run_and_check(Q, K, V, dtype=ttnn.bfloat16, device=device)
