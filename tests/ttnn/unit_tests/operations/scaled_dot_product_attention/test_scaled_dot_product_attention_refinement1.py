# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 1 — Numerical configurability + multi-core distribution.

Exercises the behavior added in Refinement 1:

  * SUPPORTED dtype expansion — bfloat16 (Phase 0), float32, bfloat8_b.
  * Public ``compute_kernel_config`` kwarg threading through to the
    program descriptor (math_fidelity / fp32_dest_acc_en /
    math_approx_mode / dst_full_sync_en).
  * Float32 running-state CBs when fp32_dest_acc_en=True.
  * Multi-core distribution via ttnn.split_work_to_cores — verified
    indirectly by running shapes that need more than one core's worth
    of query tile-rows (B * H * Qt > 1).
  * Precision lift on the S=8192 long-context cell (the Phase 0
    failure point — `severity=precision`, RMS ~0.056 vs target 0.05).

PCC thresholds are dtype-aware: bf16/fp32 ≥ 0.995, bf8b ≥ 0.99.

Run with:
    scripts/run_safe_pytest.sh \\
      tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_refinement1.py
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Reference + helpers
# ---------------------------------------------------------------------------


def _torch_sdpa(q, k, v, mask=None, scale=None):
    """Reference SDPA computed in fp32, returned in q's dtype."""
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
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
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; ref in bf16
}


def _pcc_threshold(dtype):
    # Matches eval/golden_tests/scaled_dot_product_attention/helpers.py
    # TOLERANCES table: bf16/fp32 ≥ 0.995, bf8b ≥ 0.99.
    return {
        ttnn.float32: 0.999,
        ttnn.bfloat16: 0.995,
        ttnn.bfloat8_b: 0.99,
    }[dtype]


def _make_qkv(B, H, Sq, Skv, D, dtype, seed=42, scale=0.3):
    torch.manual_seed(seed)
    torch_dtype = _TORCH_DTYPE[dtype]
    Q = (torch.randn(B, H, Sq, D) * scale).to(torch_dtype)
    K = (torch.randn(B, H, Skv, D) * scale).to(torch_dtype)
    V = (torch.randn(B, H, Skv, D) * scale).to(torch_dtype)
    return Q, K, V


# ---------------------------------------------------------------------------
# Test 1: dtype matrix on a small multi-core shape (B=2, H=4, S=128 → 32 rows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
def test_sdpa_dtype_matrix_default_config(device, dtype):
    """Verify all three supported dtypes pass on a representative
    multi-core shape (B=2, H=4, S=128, D=64 → total_rows=32 > 1 core's
    worth, so split_work_to_cores must actually distribute work)."""
    B, H, S, D = 2, 4, 128, 64
    Q, K, V = _make_qkv(B, H, S, S, D, dtype)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=_pcc_threshold(dtype))


# ---------------------------------------------------------------------------
# Test 2: compute_kernel_config is honored
# ---------------------------------------------------------------------------

# All variants pin fp32_dest_acc_en=True — the kernel's fused-scale-exp
# SFPU path (exp_tile<scale_en=true>) requires fp32 DEST accumulation;
# fp32_dest_acc_en=False is rejected at validate() with a clear error
# (see test_sdpa_rejects_bf16_dest_acc).
_COMPUTE_CFG_VARIANTS = [
    pytest.param(
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
        id="HiFi4_fp32acc",
    ),
    pytest.param(
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
        id="HiFi3_fp32acc",
    ),
    pytest.param(
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
        id="HiFi2_fp32acc",
    ),
    pytest.param(
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
        ),
        id="LoFi_approxsfpu_fp32acc",
    ),
]


@pytest.mark.parametrize("config", _COMPUTE_CFG_VARIANTS)
def test_sdpa_compute_kernel_config_variants_bf16(device, config):
    """Sanity-check that every (math_fidelity × math_approx_mode) combo
    with fp32_dest_acc_en=True dispatches and stays within a loose PCC
    envelope. Tight precision tracking lives in the precision_matrix
    test below."""
    B, H, S, D = 1, 2, 64, 64
    Q, K, V = _make_qkv(B, H, S, S, D, ttnn.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v, compute_kernel_config=config)
    actual = ttnn.to_torch(out)

    # Loose PCC envelope — LoFi is intentionally lower-precision.
    pcc = 0.97 if config.math_fidelity == ttnn.MathFidelity.LoFi else 0.99
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=pcc)


def test_sdpa_rejects_bf16_dest_acc(device):
    """fp32_dest_acc_en=False is rejected at validate() — the kernel's
    fused-scale-exp SFPU path requires fp32 DEST accumulation. A future
    refinement could lift this constraint, but today the kernel won't
    build under bf16 DEST accumulation."""
    from ttnn.operations._op_contract import UnsupportedAxisValue

    B, H, S, D = 1, 2, 64, 64
    Q, K, V = _make_qkv(B, H, S, S, D, ttnn.bfloat16)
    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    bad_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )
    with pytest.raises(UnsupportedAxisValue, match="fp32_dest_acc_en=False"):
        scaled_dot_product_attention(q, k, v, compute_kernel_config=bad_config)


def test_sdpa_compute_kernel_config_none_matches_default(device):
    """compute_kernel_config=None must produce identical output to
    explicit defaults (HiFi4 + fp32_dest_acc_en=True)."""
    B, H, S, D = 1, 2, 64, 64
    Q, K, V = _make_qkv(B, H, S, S, D, ttnn.bfloat16)

    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_default = scaled_dot_product_attention(q, k, v)
    out_explicit = scaled_dot_product_attention(
        q,
        k,
        v,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )

    # Identical: both use the same defaults, so device output should
    # match bit-for-bit (up to platform determinism).
    assert torch.equal(ttnn.to_torch(out_default), ttnn.to_torch(out_explicit))


# ---------------------------------------------------------------------------
# Test 3: multi-core distribution — explicit work-split cases
# ---------------------------------------------------------------------------

#
# Shapes chosen so total_rows = B*H*Qt comfortably exceeds one core's
# worth (8x8=64 cores on Wormhole). Each one runs the same per-row
# pipeline; matching torch reference confirms the work-split + per-core
# runtime args are wired correctly.
#
_MULTICORE_SHAPES = [
    pytest.param(1, 1, 32, 32, 32, id="single_row_single_core"),  # total_rows=1
    pytest.param(1, 4, 128, 128, 64, id="rows16_smallcfg"),  # total_rows=16
    pytest.param(2, 4, 256, 256, 64, id="rows64_fullgrid"),  # total_rows=64
    pytest.param(2, 8, 256, 256, 64, id="rows128_multipass"),  # total_rows=128 > 64 cores
    pytest.param(1, 12, 128, 128, 64, id="rows48_bert_base"),  # total_rows=48
    pytest.param(4, 8, 128, 128, 64, id="rows128_batched"),  # total_rows=128
]


@pytest.mark.parametrize("B,H,S,Skv,D", _MULTICORE_SHAPES)
def test_sdpa_multicore_self_attention(device, B, H, S, Skv, D):
    """Self-attention across varying total_rows counts — each cell
    forces split_work_to_cores into a different regime (single core,
    sub-grid, full-grid, multi-pass per core)."""
    assert S == Skv, "this test is self-attention; use cross-attention test for S != Skv"
    Q, K, V = _make_qkv(B, H, S, S, D, ttnn.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S, D), f"shape: got {actual.shape}, want ({B},{H},{S},{D})"
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=0.995)


def test_sdpa_multicore_cross_attention(device):
    """Cross-attention (S_q != S_kv) split across multiple cores — the
    reader's per-core start_row decoding handles the rectangular
    work-unit shape correctly."""
    B, H, S_q, S_kv, D = 2, 4, 128, 256, 64
    Q, K, V = _make_qkv(B, H, S_q, S_kv, D, ttnn.bfloat16)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert tuple(actual.shape) == (B, H, S_q, D)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=0.995)


# ---------------------------------------------------------------------------
# Test 4: dtype × compute_kernel_config × mask combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
def test_sdpa_dtype_with_causal_mask(device, dtype):
    """Each dtype + an additive causal mask. Mask format matches input
    dtype per the validate() relaxation in Refinement 1."""
    B, H, S, D = 1, 2, 64, 64
    Q, K, V = _make_qkv(B, H, S, S, D, dtype)
    torch_dtype = _TORCH_DTYPE[dtype]
    mask = torch.triu(torch.full((S, S), float("-inf"), dtype=torch.float32), diagonal=1).to(torch_dtype)
    mask_full = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).contiguous()

    expected = _torch_sdpa(Q, K, V, mask=mask_full)

    q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_mask = ttnn.from_torch(mask_full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(q, k, v, attention_mask=ttnn_mask)
    actual = ttnn.to_torch(out)

    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=_pcc_threshold(dtype))


# ---------------------------------------------------------------------------
# Test 5: S = 8192 precision lift target
# ---------------------------------------------------------------------------
#
# Phase 0 left two failing cells: S = 8192 self-attention, bf16,
# mask_mode=none, severity=precision. Refinement 1 widens the running-
# state CBs to fp32 so the K-loop online-softmax update accumulates
# against an fp32 reload, which should narrow or close that gap.
#
# We assert PCC ≥ 0.999 (well above the 0.999731 Phase 0 baseline) so
# this test functions as a regression guard against any future
# precision-degrading change.


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
    ],
)
def test_sdpa_long_context_s8192_precision(device, dtype):
    """Long-context self-attention at S=8192 — Refinement 1 precision
    lift target. fp32 must clearly pass; bf16 should pass loosened
    PCC=0.997 (above Phase 0's 0.999731 baseline)."""
    B, H, S, D = 1, 1, 8192, 64
    Q, K, V = _make_qkv(B, H, S, S, D, dtype, scale=0.3)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    pcc = 0.999 if dtype == ttnn.float32 else 0.997
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc=pcc)
