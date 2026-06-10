# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 4 — fp32 long-context precision via two-pass output normalization.

Mirrors the failing-cell table from
``ttnn/ttnn/operations/scaled_dot_product_attention/op_requirements.md``:
six fp32 + S >= 4096 cells where R1's one-pass online-softmax cur_mm_out
cascade (``cur_mm_out = prev_mm_out * exp_max_diff + partial``)
accumulated rounding error that pushed RMS over the fp32 target of 0.02:

    | Cell                                | Pre-R4 PCC | Pre-R4 RMS |
    |-------------------------------------|------------|------------|
    | Q1x1x4096x64 fp32 self auto         |  0.999656  |  0.026267  |
    | Q1x1x4096x64 fp32 self explicit     |  0.999656  |  0.026267  |
    | Q1x4x4096x64 fp32 self auto         |  0.999662  |  0.026222  |
    | Q1x4x4096x64 fp32 self explicit     |  0.999662  |  0.026222  |
    | Q1x1x8192x64 fp32 self auto         |  0.998610  |  0.053144  |
    | Q1x1x8192x64 fp32 self explicit     |  0.998610  |  0.053144  |
    | Q1x4x4096x64_KV1x1x4096x64 mqa auto |  0.999696  |  0.024784  |
    | Q1x4x4096x64_KV1x1x4096x64 mqa exp. |  0.999696  |  0.024784  |
    | Q1x8x4096x128_KV1x2x4096x128 gqa a. |  0.999684  |  0.025178  |
    | Q1x8x4096x128_KV1x2x4096x128 gqa e. |  0.999649  |  0.026515  |

After R4 these should all pass the fp32 golden target (PCC >= 0.999,
RMS <= 0.02). Inputs use the same generator as
``eval/golden_tests/scaled_dot_product_attention/helpers.py`` —
``torch.randn(q_shape)`` with seed 0 — so the numeric values match the
golden suite exactly.

Plus non-regression coverage:
  * S=8192 bf16 stays passing (closed at R1).
  * Short-context fp32 + bf16 + bf8b unchanged (R1 path).
  * R2 GQA / MQA + R3 non-aligned still compose.

Run with:
    scripts/run_safe_pytest.sh \\
      tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_refinement4.py
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Reference + helpers (matches golden helpers.py exactly)
# ---------------------------------------------------------------------------


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native bf8b
}


def _torch_sdpa(Q, K, V, *, attention_mask=None, scale=None):
    """Reference SDPA in fp32, mirrors GQA/MQA broadcast of the golden helper."""
    original_dtype = Q.dtype
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.to(torch.float32)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, Vf).to(original_dtype)


def _make_qkv_golden_style(q_shape, k_shape, v_shape, dtype, *, seed=0):
    """Same input distribution as the golden suite — torch.randn with
    seed 0 and no scale multiplier. The pre-R4 PCC/RMS numbers in the
    requirements table were measured against this exact distribution."""
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(seed)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    return Q, K, V


def _rel_rms(actual, expected):
    """Relative RMS error (RMS of diff / RMS of expected). Matches the
    golden suite's RMS metric — see eval/metrics.py."""
    a = actual.to(torch.float32).flatten()
    e = expected.to(torch.float32).flatten()
    rms_diff = torch.sqrt(torch.mean((a - e) ** 2))
    rms_ref = torch.sqrt(torch.mean(e**2))
    return (rms_diff / rms_ref).item()


def _pcc(actual, expected):
    """Pearson correlation coefficient — matches assert_with_pcc's metric."""
    a = actual.to(torch.float32).flatten()
    e = expected.to(torch.float32).flatten()
    a = a - a.mean()
    e = e - e.mean()
    denom = torch.sqrt(torch.sum(a * a) * torch.sum(e * e))
    if denom == 0:
        return 1.0
    return ((a * e).sum() / denom).item()


# Fp32 golden target.
FP32_PCC_TARGET = 0.999
FP32_RMS_TARGET = 0.02

# Explicit scale used by the golden suite.
EXPLICIT_SCALE = 0.125


# ---------------------------------------------------------------------------
# 1. The named failing cells from op_requirements.md Refinement 4 table
# ---------------------------------------------------------------------------


_R4_FAILING_CELLS = [
    # MHA + S=4096, D=64
    pytest.param((1, 1, 4096, 64), (1, 1, 4096, 64), None, id="mha_s4096_d64_auto"),
    pytest.param((1, 1, 4096, 64), (1, 1, 4096, 64), EXPLICIT_SCALE, id="mha_s4096_d64_explicit"),
    pytest.param((1, 4, 4096, 64), (1, 4, 4096, 64), None, id="mha_s4096_h4_d64_auto"),
    pytest.param((1, 4, 4096, 64), (1, 4, 4096, 64), EXPLICIT_SCALE, id="mha_s4096_h4_d64_explicit"),
    # MHA + S=8192 — the worst pre-R4 cell (PCC=0.998610, RMS=0.053).
    pytest.param((1, 1, 8192, 64), (1, 1, 8192, 64), None, id="mha_s8192_d64_auto"),
    pytest.param((1, 1, 8192, 64), (1, 1, 8192, 64), EXPLICIT_SCALE, id="mha_s8192_d64_explicit"),
    # MQA: H_q=4, H_kv=1
    pytest.param((1, 4, 4096, 64), (1, 1, 4096, 64), None, id="mqa_h4to1_s4096_d64_auto"),
    pytest.param((1, 4, 4096, 64), (1, 1, 4096, 64), EXPLICIT_SCALE, id="mqa_h4to1_s4096_d64_explicit"),
    # GQA: H_q=8, H_kv=2 (4:1 ratio), D=128
    pytest.param((1, 8, 4096, 128), (1, 2, 4096, 128), None, id="gqa_h8to2_s4096_d128_auto"),
    pytest.param((1, 8, 4096, 128), (1, 2, 4096, 128), EXPLICIT_SCALE, id="gqa_h8to2_s4096_d128_explicit"),
]


@pytest.mark.parametrize("q_shape,k_shape,scale", _R4_FAILING_CELLS)
def test_r4_failing_cell_passes_fp32_target(device, q_shape, k_shape, scale):
    """Each of the 10 R4 named failing cells must pass the fp32 golden
    target (PCC >= 0.999, RMS <= 0.02) after two-pass output normalization."""
    Q, K, V = _make_qkv_golden_style(q_shape, k_shape, k_shape, ttnn.float32, seed=0)
    expected = _torch_sdpa(Q, K, V, scale=scale)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(q, k, v, scale=scale)
    actual = ttnn.to_torch(out)

    pcc = _pcc(actual, expected)
    rms = _rel_rms(actual, expected)
    assert pcc >= FP32_PCC_TARGET, (
        f"R4 cell q={q_shape} k={k_shape} scale={scale} — "
        f"PCC={pcc:.6f} below fp32 target {FP32_PCC_TARGET} "
        f"(was the same as / regressed from pre-R4)"
    )
    assert rms <= FP32_RMS_TARGET, (
        f"R4 cell q={q_shape} k={k_shape} scale={scale} — " f"RMS={rms:.6f} above fp32 target {FP32_RMS_TARGET}"
    )


# ---------------------------------------------------------------------------
# 2. Non-regression on shorter contexts (R1 fp32 path unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S,D",
    [
        pytest.param(1, 1, 64, 64, id="fp32_s64_d64"),
        pytest.param(1, 1, 128, 64, id="fp32_s128_d64"),
        pytest.param(1, 2, 256, 64, id="fp32_s256_h2_d64"),
        pytest.param(1, 1, 1024, 64, id="fp32_s1024_d64"),
    ],
)
def test_r4_fp32_short_context_no_regression(device, B, H, S, D):
    """Short-context fp32 cells were already passing pre-R4. R4 must not
    regress them — two passes of K instead of one is more DRAM traffic but
    must produce equivalent (or better) precision."""
    Q, K, V = _make_qkv_golden_style((B, H, S, D), (B, H, S, D), (B, H, S, D), ttnn.float32)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert _pcc(actual, expected) >= 0.999
    assert _rel_rms(actual, expected) <= 0.02


# ---------------------------------------------------------------------------
# 3. Non-regression on bf16 + bf8b paths (independent of R4's fp32 fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,pcc_target,rms_target",
    [
        pytest.param(ttnn.bfloat16, 0.995, 0.05, id="bf16"),
        pytest.param(ttnn.bfloat8_b, 0.99, 0.12, id="bf8b"),
    ],
)
def test_r4_non_fp32_dtype_non_regression(device, dtype, pcc_target, rms_target):
    """bf16 and bf8b were not the R4 target — they should still pass their
    pre-R4 tolerances (matching the golden TOLERANCES table). The two-pass
    rewrite touches the algorithm uniformly; this is the non-regression guard."""
    B, H, S, D = 1, 1, 128, 64
    Q, K, V = _make_qkv_golden_style((B, H, S, D), (B, H, S, D), (B, H, S, D), dtype)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    pcc = _pcc(actual, expected)
    rms = _rel_rms(actual, expected)
    assert pcc >= pcc_target, f"PCC={pcc:.4f} below {pcc_target}"
    assert rms <= rms_target, f"RMS={rms:.4f} above {rms_target}"


# ---------------------------------------------------------------------------
# 4. S=8192 fp32 — the worst pre-R4 cell, isolated guard
# ---------------------------------------------------------------------------


def test_r4_s8192_fp32_precision_lift(device):
    """The single worst pre-R4 cell — PCC=0.998610, RMS=0.053. This test
    asserts the FULL fp32 target (PCC>=0.999, RMS<=0.02) — the deepest
    precision win from two-pass normalization."""
    B, H, S, D = 1, 1, 8192, 64
    Q, K, V = _make_qkv_golden_style((B, H, S, D), (B, H, S, D), (B, H, S, D), ttnn.float32)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    pcc = _pcc(actual, expected)
    rms = _rel_rms(actual, expected)
    assert pcc >= 0.999, f"S=8192 fp32 PCC={pcc:.6f} below 0.999 (pre-R4: 0.998610)"
    assert rms <= 0.02, f"S=8192 fp32 RMS={rms:.6f} above 0.02 (pre-R4: 0.053144)"


# ---------------------------------------------------------------------------
# 5. R2/R3 composition still passes
# ---------------------------------------------------------------------------


def test_r4_composes_with_r2_gqa(device):
    """GQA + long context — R4's fixed-max pass-2 must work through the
    KV-head broadcast path."""
    B, Hq, Hkv, S, D = 1, 8, 2, 4096, 128
    Q, K, V = _make_qkv_golden_style((B, Hq, S, D), (B, Hkv, S, D), (B, Hkv, S, D), ttnn.float32)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    assert _pcc(actual, expected) >= 0.999
    assert _rel_rms(actual, expected) <= 0.02


def test_r4_composes_with_r3_non_aligned(device):
    """Non-aligned S_kv (synthetic alignment mask) + fp32 — R4's reader-side
    pass-2 must re-issue the synthetic mask correctly on the second pass."""
    # S_kv=47 → not tile-aligned. Last K tile has 15 valid keys; the
    # synthetic alignment mask must be re-stamped in pass 2 too.
    B, H, S_q, S_kv, D = 1, 1, 32, 47, 64
    Q, K, V = _make_qkv_golden_style((B, H, S_q, D), (B, H, S_kv, D), (B, H, S_kv, D), ttnn.float32)
    expected = _torch_sdpa(Q, K, V)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)

    # Looser tolerance — non-aligned shapes are a known precision-near-miss
    # condition (R3 changelog), and the goal here is just to confirm the
    # mask gets re-applied in pass 2 (not a NaN/garbage).
    assert _pcc(actual, expected) >= 0.99


def test_r4_composes_with_causal_mask(device):
    """Long-context fp32 + causal mask — exercises HAS_MASK + two-pass."""
    B, H, S, D = 1, 1, 4096, 64
    Q, K, V = _make_qkv_golden_style((B, H, S, D), (B, H, S, D), (B, H, S, D), ttnn.float32)
    mask = torch.zeros(B, 1, S, S, dtype=torch.float32)
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
    expected = _torch_sdpa(Q, K, V, attention_mask=mask)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    m = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v, attention_mask=m)
    actual = ttnn.to_torch(out)

    assert _pcc(actual, expected) >= 0.999
    assert _rel_rms(actual, expected) <= 0.02
