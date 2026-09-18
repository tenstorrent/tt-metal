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
    MAX_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 3: ops with a fast_and_approximate_mode parameter.
sqrt, rsqrt, exp, erf, gelu, log, log10, log2, log1p, mish.

Each op is swept in both modes (accurate=False, fast=True) over exhaustive
normal-bfloat16 input, mirroring thresholds from the merged tests they
consolidate (test_math, test_unary, test_activation, test_unary_ops_ttnn,
test_unary_category1). Out-of-domain inputs (x<0 for sqrt, x<=0 for log, etc.)
are checked separately and must be non-finite.

gelu's golden is computed in float64 (see test_gelu_accurate/test_gelu_fast):
fp32's 1 + erf(x/sqrt(2)) suffers cancellation for |x| around 5-6. This
initially produced a false ~171 ULP failure misattributed to a device kernel
defect (issue #56616 should be revisited/closed as a misattribution). Even
with float64, the far-negative tail's true result is so tiny that ULP stops
being meaningful (per assert_with_ulp's own docstring) — those elements are
checked via absolute tolerance instead (see GELU_TINY_GOLDEN_ABS).

assert_allclose's rtol is scaled by actual_result (device output), not
expected_result (golden), despite its docstring's stated formula — a
pre-existing quirk of the shared helper, not specific to this file.
"""


def _assert_all_nonfinite(result, desc):
    """Every element must be non-finite (device may pack NaN as ±inf in bf16)."""
    nonfinite = ~torch.isfinite(result)
    assert nonfinite.all(), f"expected all {desc} outputs to be non-finite; {int((~nonfinite).sum())} were finite"


def _assert_finite_matches_golden(golden, result, desc):
    """Finite-input elements must produce finite output (PCC alone can hide an
    isolated NaN/Inf regression since comparison_funcs zeroes both sides)."""
    unexpected_nonfinite = torch.isfinite(golden) & ~torch.isfinite(result)
    assert (
        not unexpected_nonfinite.any()
    ), f"{desc}: {int(unexpected_nonfinite.sum())} finite-input elements produced a non-finite device output"


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
    """Accurate mode over the positive-normal domain (rsqrt excludes 0)."""
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
    """Fast mode over [1, 100], ULP <= 2 (matches test_unary_root_ops_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


@pytest.mark.parametrize(
    "ttnn_op, high",
    [
        (ttnn.sqrt, -SMALLEST_NORMAL_BF16),  # sqrt(0) = 0 is in-domain; keep 0 out of this sweep
        (ttnn.rsqrt, 0.0),  # rsqrt(0) = +inf is out-of-domain; include 0 here
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_negative_domain(device, ttnn_op, high):
    """Out-of-domain must be non-finite (accurate mode only). Fast mode is
    excluded: hardware showed it doesn't validate its domain and returns
    finite garbage for negative inputs instead of NaN/inf."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, high)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)" if high == 0.0 else f"{ttnn_op.__name__}(x<0)")


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, low, high, ulp",
    [
        (ttnn.sqrt, torch.sqrt, 0.0, 100.0, 1),
        (ttnn.gelu, F.gelu, 1.0, 10.0, 10),
        (ttnn.log, torch.log, 1.0, 100.0, 1),
        (ttnn.log2, torch.log2, 1.0, 100.0, 1),
        (ttnn.log10, torch.log10, 1.0, 100.0, 2),
        (ttnn.log1p, torch.log1p, 1.0, 100.0, 1),
    ],
    ids=["sqrt", "gelu", "log", "log2", "log10", "log1p"],
)
def test_row_major_layout_smoke(device, ttnn_op, golden_fn, low, high, ulp):
    """ROW_MAJOR_LAYOUT dispatch-path smoke check (accurate mode); exhaustive
    sweeps above use TILE_LAYOUT. gelu's range stays positive to avoid the
    exp-field=1 FTZ band (see test_gelu_accurate)."""
    input_tensor = generate_bfloat16_bits_in_range(low, high)
    tt_in = to_tt_tensor(input_tensor, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp)


def test_non_tile_aligned_shape_smoke(device):
    """gelu smoke check on a non-tile-aligned width (120 isn't a multiple of
    32), accurate mode, ULP <= 10 over [1, 10]."""
    shape = (1, 2, 64, 120)
    num_elements = torch.prod(torch.tensor(shape)).item()
    input_tensor = torch.linspace(1.0, 10.0, num_elements, dtype=torch.bfloat16).reshape(shape)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=10)


# ─────────────────────────────────────────────────────────────────────────────
# exp — ULP over its finite (non-overflow, non-underflow) range
# ─────────────────────────────────────────────────────────────────────────────


def test_exp_accurate(device):
    """Accurate mode: ULP <= 1 over [-87, 88.5], the finite range."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_exp_fast(device):
    """Fast mode over the same finite range. PCC alone is near-vacuous here
    (exp spans ~76 decades, so it's dominated by the largest values), so an
    allclose bound is added as the real gate for the fast path. Below
    x~-13.8 (golden < atol=1e-6), atol dominates rtol and this bound stops
    being a meaningful gate — that tail's decay-to-zero behavior isn't
    characterized here (only test_exp_underflow's x<-87 boundary is)."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    _assert_finite_matches_golden(golden, result, "exp(fast)")
    assert_with_pcc(golden, result, pcc=0.999)
    assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=1e-6)


def test_exp_underflow(device):
    """Underflow tail (x < -87), accurate mode only: exp(x) rounds to exactly
    0. Fast mode is excluded — hardware showed it retains a slowly-decaying
    non-zero tail instead of hard-flushing; see test_exp_fast."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, -87.5)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=False))
    assert torch.all(result == 0.0), "expected exp underflow region (x < -87) to be exactly 0 in accurate mode"


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_exp_overflow(device, fast):
    """Overflow tail (x > 88.5): exp(x) saturates to +inf, in both modes."""
    input_tensor = generate_bfloat16_bits_in_range(89.0, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=fast))
    assert torch.all(torch.isposinf(result)), "expected exp overflow region (x > 88.5) to be +inf"


# ─────────────────────────────────────────────────────────────────────────────
# erf — bounded [-1, 1]; ULP ≤ 2 over the active band in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_erf(device, fast):
    """erf over [-10, 10] (saturates to +-1 beyond ~+-4), ULP <= 2 both modes."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.erf(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.erf(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# gelu — accurate exhaustive (ULP ≤ 10, FTZ band excluded); fast PCC+allclose
# ─────────────────────────────────────────────────────────────────────────────

# Below this golden magnitude, ULP is not a meaningful metric (per
# assert_with_ulp's own docstring): for x <~ -5, gelu(x) = x/2*(1+erf(x/sqrt2))
# is a near-total cancellation between two O(1) terms, so both torch (even in
# float64) and the device round the minuscule true result very differently at
# the bit level, while the absolute error stays negligible. These elements are
# checked against GELU_TINY_GOLDEN_ATOL instead.
GELU_TINY_GOLDEN_ABS = 1e-6
GELU_TINY_GOLDEN_ATOL = 1e-6


def test_gelu_accurate(device):
    """Accurate mode: ULP <= 10 over all normal bf16 patterns, with two
    exclusions:
    - exp-field=1 (|x| in [2^-126, 2^-125)): x/2 underflows to an fp32
      subnormal that hardware DAZ/FTZ flushes to 0.
    - |golden| < GELU_TINY_GOLDEN_ABS (the far-negative cancellation tail,
      e.g. x~=-6.6): checked via absolute tolerance instead of ULP (see
      GELU_TINY_GOLDEN_ABS above). Originally misdiagnosed as a device kernel
      defect at a specific element (issue #56616) — see PR #56338 review
      discussion; golden is computed in float64 to rule out torch's own fp32
      cancellation as a separate source of error."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.double()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    abs_x = input_tensor.abs().float()
    exp1_ftz_band = (abs_x >= 2.0**-126) & (abs_x < 2.0**-125)
    assert exp1_ftz_band.any(), "expected exp-field=1 FTZ band to be non-empty for this exhaustive sweep"

    finite_golden = torch.isfinite(golden)
    unexpected_nonfinite = finite_golden & ~exp1_ftz_band & ~torch.isfinite(result)
    assert not unexpected_nonfinite.any(), (
        f"gelu(accurate): {int(unexpected_nonfinite.sum())} finite-input elements outside the documented "
        "FTZ band produced a non-finite device output"
    )

    tiny_golden = golden.abs() < GELU_TINY_GOLDEN_ABS
    ulp_keep = ~exp1_ftz_band & finite_golden & ~tiny_golden
    tiny_keep = ~exp1_ftz_band & finite_golden & tiny_golden
    assert tiny_keep.any(), "expected the far-negative cancellation tail to be non-empty"

    assert_with_ulp(expected_result=golden[ulp_keep], actual_result=result[ulp_keep], ulp_threshold=10)
    assert_allclose(
        expected_result=golden[tiny_keep], actual_result=result[tiny_keep], rtol=0, atol=GELU_TINY_GOLDEN_ATOL
    )


def test_gelu_fast(device):
    """Fast mode (FastLut): PCC >= 0.999 over [-10, 10]. PCC alone is weak here
    (dominated by the large-|x| near-identity tail), so an allclose bound
    backs it up, same rationale as test_exp_fast. Golden uses float64 for the
    same cancellation reason as test_gelu_accurate."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.double()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    _assert_finite_matches_golden(golden, result, "gelu(fast)")
    assert_with_pcc(golden, result, pcc=0.999)
    assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=0.05)


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
    """Accurate mode over the positive-normal domain; x <= 0 is covered
    separately. log10 gets an extra ULP for its base-change multiply."""
    input_tensor = generate_bfloat16_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10], ids=["log", "log2", "log10"])
@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_log_family_nonpositive_domain(device, ttnn_op, fast):
    """Out-of-domain (x <= 0): log(0)=-inf, log(x<0)=NaN, so output must be
    non-finite in both modes."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, 0.0)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=fast))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)")


def test_log1p_accurate(device):
    """Accurate mode: ULP <= 1 for x > -1; x <= -1 must be non-finite
    (log1p(-1)=-inf, log1p(x<-1)=NaN)."""
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
    """Fast mode: allclose(atol=0.0625) over [1, 100]."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden_fn = {ttnn.log: torch.log, ttnn.log2: torch.log2, ttnn.log10: torch.log10, ttnn.log1p: torch.log1p}[ttnn_op]
    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_allclose(expected_result=golden, actual_result=result, atol=0.0625)


# ─────────────────────────────────────────────────────────────────────────────
# mish — x * tanh(softplus(x)); compound, allclose in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_mish(device, fast):
    """mish over all normal bf16 values, allclose(rtol=1e-5, atol=0.02) in both
    modes. atol is looser than the merged tests' 0.008 because the exhaustive
    sweep hits mish's curvature trough near x~-1.19."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.mish(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.mish(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_allclose(expected_result=golden, actual_result=result, rtol=1e-5, atol=0.02)
