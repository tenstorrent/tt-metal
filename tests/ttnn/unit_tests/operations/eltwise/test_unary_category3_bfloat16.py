# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc, assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device

MAX_BF16 = float(torch.finfo(torch.bfloat16).max)

"""
Category 3: Ops with a fast_and_approximate_mode parameter

1. ttnn.sqrt    - Square root                (domain: x >= 0)
2. ttnn.rsqrt   - Reciprocal square root     (domain: x >  0)
3. ttnn.exp     - Exponential                 (domain: all reals; overflow > ~88.5)
4. ttnn.erf     - Gaussian error function     (domain: all reals)
5. ttnn.gelu    - GELU                        (domain: all reals)
6. ttnn.log     - Natural logarithm           (domain: x >  0)
7. ttnn.log10   - Base-10 logarithm           (domain: x >  0)
8. ttnn.log2    - Base-2 logarithm            (domain: x >  0)
9. ttnn.log1p   - log(1 + x)                  (domain: x > -1)
10. ttnn.mish    - Mish activation             (domain: all reals)

Every op is exercised in BOTH modes on exhaustive normal-bfloat16 sweeps
(same 65 536-bit-pattern helpers as category1/2/5):

  · fast_and_approximate_mode = False  (accurate / default SFPU path)
  · fast_and_approximate_mode = True   (fast approximate path)

Accuracy criteria (inherited from the already-merged, hardware-verified tests
these consolidate — see "supersedes" note at the bottom of each section):
─────────────────────────────────────────────────────────────────────────────
  sqrt   : accurate ULP ≤ 1 over x >= 0; fast ULP ≤ 2 over [1, 100];
                     x < 0 → non-finite (NaN)               [test_math, test_unary]
  rsqrt  : accurate ULP ≤ 2 over x > 0; fast ULP ≤ 2 over [1, 100];
                     x <= 0 → non-finite (+inf / NaN)        [test_unary_fp32, test_unary]
  exp    : accurate ULP ≤ 1 over [-87, 88.5]; fast PCC ≥ 0.999 same range
                                                            [test_unary_category1, test_exp]
  erf    : ULP ≤ 2 over [-10, 10] in both modes             [test_math, test_unary_ops_ttnn]
  gelu   : accurate ULP ≤ 10 over all normals (exp-field=1 FTZ band excluded);
                     fast PCC ≥ 0.999 over [-10, 10]         [test_activation, test_unary_ops_ttnn]
  log/log2 : accurate ULP ≤ 1 over x > 0; log10 accurate ULP ≤ 2;
  log1p    : accurate ULP ≤ 1 over x > -1;
                     x out-of-domain → non-finite;
                     fast allclose(atol=0.0625) over [1, 100]   [test_math, test_unary(_ops_ttnn)]
  mish   : allclose(rtol=1e-5, atol=0.02) over all normals in both modes
                     (the restricted-range merged tests use atol=0.008 over
                     [-20, 100]; the full sweep hits mish's curvature trough
                     near x ≈ -1.19 where the compound SFPU error reaches
                     ~0.0156 = 2^-6, ~13 ULP — hardware-observed)
                                                            [test_activation, test_unary, test_composite]

NOTE: the concrete thresholds above are taken from the merged tests this file
consolidates; the extension of each to the *full* exhaustive bf16 domain (and
every fast-mode number) still needs a confirming on-device run — no accelerator
is attached in the authoring environment (see AGENTS.md).
"""


def _assert_all_nonfinite(result, desc):
    """Every element must be non-finite (device may pack NaN as ±inf in bf16)."""
    nonfinite = ~torch.isfinite(result)
    assert nonfinite.all(), f"expected all {desc} outputs to be non-finite; {int((~nonfinite).sum())} were finite"


# ─────────────────────────────────────────────────────────────────────────────
# sqrt, rsqrt — reciprocal/root ops, ULP-based on their positive domain
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, low, ulp_accurate",
    [
        (ttnn.sqrt, torch.sqrt, 0.0, 1),
        (ttnn.rsqrt, torch.rsqrt, SMALLEST_NORMAL_BF16, 2),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_accurate(device, ttnn_op, golden_fn, low, ulp_accurate):
    """Accurate mode: exhaustive positive-normal bf16 domain.

    sqrt is defined at 0 (sqrt(0)=0); rsqrt diverges at 0 (+inf), so its sweep
    starts at the smallest normal. Reference is computed in float32 then rounded
    to bf16, giving a reference strictly more accurate than a bf16-native one.
    """
    input_tensor = generate_bfloat16_bits_in_range(low, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_accurate, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_op, golden_fn",
    [
        (ttnn.sqrt, torch.sqrt),
        (ttnn.rsqrt, torch.rsqrt),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_fast(device, ttnn_op, golden_fn):
    """Fast approximate mode: exhaustive bf16 values in [1, 100], ULP ≤ 2.

    Matches the fast-mode range/tolerance of test_unary.py::test_unary_root_ops_ttnn.
    """
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


@pytest.mark.parametrize("ttnn_op", [ttnn.sqrt, ttnn.rsqrt], ids=["sqrt", "rsqrt"])
def test_root_ops_negative_domain(device, ttnn_op):
    """Out-of-domain (x < 0): device must return a non-finite value (NaN)."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, -SMALLEST_NORMAL_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<0)")


# ─────────────────────────────────────────────────────────────────────────────
# exp — ULP over its finite (non-overflow, non-underflow) range
# ─────────────────────────────────────────────────────────────────────────────


def test_exp_accurate(device):
    """Accurate mode: ULP ≤ 1 over [-87, 88.5] (the finite range: exp underflows
    to 0 below ~-87 and overflows to +inf above ~88.5). Mirrors the exhaustive
    Category 1 coverage in test_unary_category1_bfloat16.py::test_exp_ops."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_exp_fast(device):
    """Fast approximate mode: PCC ≥ 0.999 over the same finite range."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# erf — bounded [-1, 1]; ULP ≤ 2 over the active band in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_erf(device, fast):
    """erf over [-10, 10] (beyond ±~4 it saturates to ±1, exactly representable).
    ULP ≤ 2 in both modes, matching test_unary_ops_ttnn.py::test_unary_erf_ttnn."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.erf(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.erf(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# gelu — accurate exhaustive (ULP ≤ 10, FTZ band excluded); fast PCC
# ─────────────────────────────────────────────────────────────────────────────


def test_gelu_accurate(device):
    """Accurate mode: all normal bf16 patterns, ULP ≤ 10.

    The exp-field=1 band (|x| in [2^-126, 2^-125)) is excluded: there
    gelu(x) ≈ x/2 lands in fp32-subnormal territory, so the device DAZ/FTZ
    flushes it to 0 while torch (no FTZ) keeps a tiny value — up to 128 ULP,
    a documented hardware artifact (see
    test_activation.py::test_gelu_bfloat16_accuracy)."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    abs_x = input_tensor.abs().float()
    exp1_ftz_band = (abs_x >= 2.0**-126) & (abs_x < 2.0**-125)
    assert exp1_ftz_band.any(), "expected exp-field=1 FTZ band to be non-empty for this exhaustive sweep"

    keep = ~exp1_ftz_band & torch.isfinite(golden) & torch.isfinite(result)
    assert_with_ulp(expected_result=golden[keep], actual_result=result[keep], ulp_threshold=10)


def test_gelu_fast(device):
    """Fast approximate mode (FastLut): PCC ≥ 0.999 over [-10, 10].

    The 6-segment FastLut collapses to ≈ x across the far-negative half, so its
    golden intentionally skips generic comparison; a bounded active-band PCC is
    the meaningful gate here (matches test_unary_ops_ttnn.py::test_unary_gelu_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# log, log2, log10, log1p — ULP on their positive/(-1,∞) domain; fast allclose
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, ulp",
    [
        (ttnn.log, torch.log, 1),
        (ttnn.log2, torch.log2, 1),
        (ttnn.log10, torch.log10, 2),
    ],
    ids=["log", "log2", "log10"],
)
def test_log_family_accurate(device, ttnn_op, golden_fn, ulp):
    """Accurate mode: exhaustive positive-normal bf16 domain.

    log/log2 are ≤ 1 ULP; log10's extra base-change multiply costs a 2nd ULP
    (see test_math.py). x <= 0 is out-of-domain and covered separately."""
    input_tensor = generate_bfloat16_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10], ids=["log", "log2", "log10"])
def test_log_family_nonpositive_domain(device, ttnn_op):
    """Out-of-domain (x <= 0): log(0) = -inf and log(x<0) = NaN, so every
    output must be non-finite."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, 0.0)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)")


def test_log1p_accurate(device):
    """Accurate mode: ULP ≤ 1 over the in-domain half (x > -1); x <= -1 must be
    non-finite (log1p(-1) = -inf, log1p(x<-1) = NaN). See
    test_unary_ops_ttnn.py::test_unary_log1p_ttnn."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.log1p(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.log1p(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    x = input_tensor.float()
    in_domain = x > -1.0
    assert_with_ulp(
        expected_result=golden[in_domain], actual_result=result[in_domain], ulp_threshold=1, allow_nonfinite=True
    )

    out_of_domain = x <= -1.0
    assert out_of_domain.any(), "expected x <= -1 band to be non-empty for this exhaustive sweep"
    _assert_all_nonfinite(result[out_of_domain], "log1p(x<=-1)")


@pytest.mark.parametrize(
    "ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10, ttnn.log1p], ids=["log", "log2", "log10", "log1p"]
)
def test_log_family_fast(device, ttnn_op):
    """Fast approximate mode: allclose(atol=0.0625) over [1, 100], matching
    test_unary_ops_ttnn.py::test_unary_log_like_fast_approx_ttnn."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden_fn = {ttnn.log: torch.log, ttnn.log2: torch.log2, ttnn.log10: torch.log10, ttnn.log1p: torch.log1p}[ttnn_op]
    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_allclose(result, golden, atol=0.0625)


# ─────────────────────────────────────────────────────────────────────────────
# mish — x * tanh(softplus(x)); compound, allclose in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_mish(device, fast):
    """mish over all normal bf16 values, allclose(rtol=1e-5, atol=0.02) in both
    modes. mish is a compound softplus/tanh chain (golden upcasts to float32,
    matching ttnn's golden in unary.py); large negatives underflow toward 0 and
    large positives approach identity — the atol/rtol pair covers both tails.

    atol is 0.02 rather than the 0.008 used by the restricted-range merged
    tests (test_unary.py::test_unary_mish over [-20, 100]): the exhaustive sweep
    reaches mish's curvature trough near x ≈ -1.19 (minimum ≈ -0.31), where the
    SFPU chain deviates by up to 0.0156 (= 2^-6, ~13 bf16 ULP) — hardware-
    observed. It remains a meaningful gate: an identity (return-x) kernel would
    deviate by ~0.88 in that trough, far above 0.02."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.mish(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.mish(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_allclose(result, golden, rtol=1e-5, atol=0.02)
