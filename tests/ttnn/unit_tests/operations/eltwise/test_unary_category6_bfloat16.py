# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
    bf16_bits_to_float,
    bf16_quantize_rne,
    float_to_bf16_bits,
    SMALLEST_NORMAL_BF16,
    MAX_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 6: Ops with multiple/special parameters

89. ttnn.softplus         - beta (default 1.0), threshold (default 20.0)
90. ttnn.xielu            - alpha_p (default 0.8), alpha_n (default 0.8)
91. ttnn.tanh             - fast_and_approximate_mode (default False)
92. ttnn.sigmoid_accurate - fast_and_approximate_mode (default False) [deprecated]
93. ttnn.sigmoid          - vector_mode (default 4), mode (SigmoidMode enum, default Accurate)
94. ttnn.unary_chain      - ops_chain (list of UnaryWithParam)
95. ttnn.clamp            - min, max (float/int scalar, or Tensor)
96. ttnn.clip             - min, max (float scalar, or Tensor)
97. ttnn.selu             - scale (default 1.0507), alpha (default 1.67326)
98. ttnn.hardtanh         - min_val (default -1.0), max_val (default 1.0)
99. ttnn.threshold        - threshold, value
100. ttnn.tril            - diagonal (default 0)
101. ttnn.triu            - diagonal (default 0)
102. ttnn.round           - decimals (default 0, supported range -6..7)
103. ttnn.polygamma       - k (int, supported range 1..10)
104. ttnn.logit           - eps (optional; eps > 0.5 takes a manual-clamp golden branch)
105. ttnn.rdiv            - value, rounding_mode (None | "floor" | "trunc")
106. ttnn.bitcast         - dtype (bfloat16 <-> uint16 is the only same-bit-width pair)

Accuracy criteria
─────────────────
  Exact (bit-for-bit): hardtanh, clamp, clip, tril, triu, bitcast, threshold.
      Select/compare/reinterpret only. Their goldens model the device's
      scalar-parameter conversion (see quirks below), which is what keeps
      them exact even for bounds that aren't representable in bf16.
  ULP-gated: round (<=1), selu (<=1), sigmoid Accurate (<=2),
      rdiv None mode (<=3), unary_chain([SQUARE, SQRT]) (<=2).
  PCC-gated: tanh, sigmoid_accurate, xielu, softplus, logit, polygamma,
      rdiv floor/trunc, sigmoid AccurateWithFastExp/FastApproximate. The
      approximate modes and the integer-valued floor/trunc results can move
      by more than a tight ULP bound for one bf16 ULP of input, so PCC is the
      honest gate there.
      A PCC over the whole grid is dominated by the largest-magnitude decade,
      so tanh, sigmoid_accurate and polygamma additionally get a per-element
      ULP assertion over the well-conditioned sub-range their replaced tests
      covered.

Coverage that deliberately lives elsewhere (recorded here so an unrelated
refactor of those files doesn't silently delete coverage this one relies on):
  - sigmoid vector_mode=2 (C): test_sigmoid_vector_modes.py, which uses the
    narrow shapes it is meant for (see the vector_mode quirk below).
  - sigmoid_accurate over [-87, 88.5] at ULP<=3:
    test_sigmoid_accurate_21f.py::test_sigmoid_accurate_arange.
  - round on bfloat8_b: test_round.py::test_round_new. Only that file's
    bfloat16 row was removed in favour of the sweep here.

Golden-function quirks (see ttnn/ttnn/operations/unary.py):
  - _golden_function_selu ignores its scale/alpha kwargs and always uses the
    paper defaults, but the kernel does honour them, so this file computes
    its own reference (_selu_reference) to actually exercise them.
  - bitcast only supports same-bit-width dtype pairs; bfloat16 -> uint16 is
    the only valid target.

Real-hardware quirks. Every item below was observed running this file against
a live wormhole_b0 chip -- each started as a failure whose exclusion band was
read off the failing elements. Blackhole has not been exercised at all.
  - hardtanh/clamp/clip bounds and threshold's `value` take the fp32->bf16
    *truncation* path (same as `fill`), not round-to-nearest-even, so their
    goldens are built from _to_device_bf16_scalar.
  - the two scalars of one `threshold` call do not share a conversion:
    `value` truncates, but `threshold`, which is compared against, rounds to
    nearest even (_to_device_bf16_comparand). Truncating both leaves exactly
    one element of the grid wrong.
  - a -0.0 scalar parameter comes back as +0.0: threshold(value=-0.0) and
    clamp(min=-0.0) canonicalize the sign away, consistent with the DAZ model
    elsewhere. torch keeps it, hence _canonicalize_negative_zero and the
    bit-pattern compares -- torch.equal reports +0.0 == -0.0 and cannot see
    this at all.
  - sigmoid's vector_mode is only valid as C (2) or RC (4); R (1) raises a
    TT_FATAL. C is a sub-tile optimization for narrow inputs, and on a full
    32x32 tile the unprocessed columns hold stale DEST content rather than a
    sigmoid output, so only vector_mode=4 is swept here.
  - the fast exp approximation stops saturating from x ~= 172 (0.98 at 172,
    0.26 at 177, 0.05 at 179) instead of holding 1.0. This hits both entry
    points -- SigmoidMode.AccurateWithFastExp and
    sigmoid_accurate(fast_and_approximate_mode=True), whose full-grid PCC it
    drags to 0.07; excluding x >= 174 gives 0.99996. tanh's fast mode does
    not use this path and needs no exclusion.
  - rdiv(rounding_mode=None) is value * reciprocal(x), so once |x| >= 1/tiny
    the reciprocal underflows and is flushed to zero before the multiply,
    zeroing the product regardless of value (mirrors softsign in
    test_unary_category4_bfloat16.py).
  - selu/sigmoid/tanh hit a flush-to-zero boundary near the smallest normal
    where the device and a float64 reference legitimately round to opposite
    sides. The mask keys off the *expected* side only (see _ftz_boundary_mask).
  - unary_chain([SQUARE, SQRT]) underflows for |x| < sqrt(tiny) (~3.4e-20),
    since x^2 underflows before the sqrt runs.
  - unary_chain([RELU, TYPECAST(bf16->fp32)]) is not bit-exact for the lowest
    ~11 bf16 exponents (|x| < 2^-115), where the widened result differs from
    a zero-extension by a small relative factor.
  - selu needs no cancellation band. The removed test_selu_arange excluded
    [-0.30859375, tiny] and earlier revisions of this file inherited it,
    masking a quarter of the grid; measured against _selu_reference the
    device is bit-exact (max ULP delta 0.0) over all 65,536 patterns for
    every scale/alpha, so it was dropped. Its companion test_selu_atol had
    never asserted the band either -- it called torch.allclose and discarded
    the result.
"""


MAX_BF16_VAL = torch.finfo(torch.bfloat16).max
TINY_BF16 = torch.finfo(torch.bfloat16).tiny

# Where the fast exp approximation stops saturating. Shared by both entry
# points into that path; characterized but excluded from the gates.
_SIGMOID_FAST_EXP_UNSAFE_X = 174.0


def _exhaustive_bf16_4d():
    """All 65,536 bfloat16 bit-patterns (finite normals only), shape (1, 1, 256, 256)."""
    return generate_bfloat16_bits(include_spl_values=False).unsqueeze(0).unsqueeze(0)


def _to_device_bf16_scalar(value):
    """Round-trip a scalar the kernel *writes into the output* through the
    device's fp32->bf16 truncation (0.66 -> 0.65625, where RNE gives
    0.66015625). Ints and None pass through. Compare _to_device_bf16_comparand.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, float):
        return value
    return bf16_bits_to_float(float_to_bf16_bits(value))


def _to_device_bf16_comparand(value):
    """Round-trip a scalar the kernel *compares against*, which is RNE rather
    than truncating. The difference is one element of the exhaustive grid --
    the single bf16 value falling between the two roundings.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, float):
        return value
    return bf16_quantize_rne(value)


def _canonicalize_negative_zero(tensor):
    """Map -0.0 to +0.0, matching what the device does to scalar parameters.

    Applied to the golden so the bit-exact comparisons pin the device's actual
    contract; if the hardware ever preserved the sign, they would fail here.
    """
    return torch.where(tensor == 0, torch.zeros_like(tensor), tensor)


def _ftz_boundary_mask(golden):
    """Elements whose *expected* value sits on the flush-to-zero boundary.

    Keyed off the expected side alone, deliberately: OR-ing in the device
    result would also mask an element where the device returned ~0 but the
    true result is large, which is the defect this should surface, not hide.
    """
    return golden.abs() <= 2 * TINY_BF16


def _assert_excluded_region(mask, label, max_fraction):
    """Bound how much of a sweep an exclusion mask may remove.

    Several bands here legitimately cover a quarter to a half of the grid
    (bf16 is exponent-dense near zero), so each caller states what it expects
    to lose and a mask that silently widened can't gut the sweep unnoticed.
    """
    excluded = int(mask.sum().item())
    fraction = excluded / mask.numel()
    assert fraction <= max_fraction, (
        f"{label}: exclusion mask removed {excluded} of {mask.numel()} elements "
        f"({fraction:.1%}), more than the {max_fraction:.0%} this test expects -- "
        f"the sweep is no longer covering what its docstring claims"
    )


def _assert_ftz_region_is_zeroish(result, ftz_mask, label):
    """FTZ-masked elements are dropped from the ULP gate, not left unchecked:
    the device must still land at or below the boundary there, which rules out
    garbage or a stale-DEST read."""
    if not ftz_mask.any():
        return
    off = result[ftz_mask].abs() > 2 * TINY_BF16
    assert not off.any(), (
        f"{label}: {int(off.sum().item())} of {int(ftz_mask.sum().item())} elements in the "
        f"flush-to-zero region came back above the boundary (max {result[ftz_mask].abs().max().item():g})"
    )


def _assert_nonfinite_agreement(golden, result, label):
    """Assert both sides agree on *where* and *which* non-finites occur.

    The finite-both-sides mask used by the PCC tests drops non-finites
    silently, so without this a device returning NaN where the reference is
    finite is indistinguishable from one that gets it right.

    Elements where one side is non-finite and the other is finite but within a
    binade of the bf16 max are exempt: the golden is evaluated in higher
    precision, so the two can legitimately straddle the overflow boundary.
    """
    straddles_overflow = (torch.isfinite(golden) & (golden.abs() >= MAX_BF16_VAL / 2)) | (
        torch.isfinite(result) & (result.abs() >= MAX_BF16_VAL / 2)
    )

    nan_mismatch = (torch.isnan(golden) != torch.isnan(result)) & ~straddles_overflow
    assert not nan_mismatch.any(), f"{label}: {int(nan_mismatch.sum().item())} elements where exactly one side is NaN"

    both_inf = torch.isinf(golden) & torch.isinf(result)
    inf_mismatch = (
        (torch.isinf(golden) != torch.isinf(result)) | (both_inf & (torch.signbit(golden) != torch.signbit(result)))
    ) & ~straddles_overflow
    assert not inf_mismatch.any(), (
        f"{label}: {int(inf_mismatch.sum().item())} elements where the infinities disagree in " f"position or sign"
    )


def _assert_bitwise_equal(result, golden, label):
    """Compare two bf16 tensors by bit pattern rather than by value.

    torch.equal reports +0.0 == -0.0, so an op handed a -0.0 scalar could
    canonicalize the sign away and still pass an exact-equality check. For the
    select/clamp ops the output is either the input or the scalar itself, so
    the sign of zero is part of the contract.
    """
    result_bits = result.view(torch.int16)
    golden_bits = golden.view(torch.int16)
    assert torch.equal(result_bits, golden_bits), (
        f"{label} diverged for {int((result_bits != golden_bits).sum().item())} of "
        f"{result_bits.numel()} elements (compared as bf16 bit patterns; "
        f"{int((result != golden).sum().item())} differ by value, the rest are signed-zero mismatches)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# hardtanh, clamp, clip, tril, triu, bitcast — exact (bit-for-bit)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("min_val", [0.25, 0.5, 0.66, -1.0])
@pytest.mark.parametrize("max_val", [1.0, 2.5, 3.0, 6.6])
def test_hardtanh_op(device, min_val, max_val):
    """output_i = clamp(x_i, min_val, max_val). Exhaustive normal bf16 sweep.

    The golden is built from the truncated bf16 bounds, since the device
    truncates rather than rounding them: min_val=0.66 would otherwise be
    0.66015625 on the golden side against 0.65625 on the device, diverging on
    every clamped element.
    """
    input_tensor = _exhaustive_bf16_4d()
    device_min_val = _to_device_bf16_scalar(min_val)
    device_max_val = _to_device_bf16_scalar(max_val)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden = golden_function(input_tensor, min_val=device_min_val, max_val=device_max_val, device=device)

    tt_result = ttnn.hardtanh(tt_in, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"hardtanh(min_val={min_val}, max_val={max_val}) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


def test_hardtanh_default_bounds(device):
    """ttnn.hardtanh(x) with no bounds, i.e. the (-1, 1) defaults.

    Worth its own case because a bounds argument that fails to arrive falls
    back to those defaults *silently* rather than raising (the trap noted in
    test_scalar_param_ops_rank_coverage), and every other call in this file
    passes bounds explicitly.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden = golden_function(input_tensor, device=device)

    result = ttnn.to_torch(ttnn.hardtanh(tt_in))

    assert golden.abs().max().item() == 1.0, "expected the (-1, 1) defaults to actually clamp"
    _assert_bitwise_equal(result, golden, "hardtanh() with default bounds")


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (0.0, 1.0),
        (None, 1.0),
        (-1.0, None),
        (-5, 5),  # ints
        (2.0, 1.0),  # degenerate: min > max collapses to a constant
        (-0.1, 0.66),  # neither bound is representable in bf16
        (-0.1, None),
        (-0.0, 1.0),  # signed-zero bound
        (-MAX_BF16_VAL, MAX_BF16_VAL),  # widest finite bounds: a no-op clamp
        (-TINY_BF16, TINY_BF16),  # narrowest: collapses onto the normal boundary
    ],
    ids=[
        "both",
        "min_only",
        "max_only",
        "int_bounds",
        "degenerate",
        "inexact_bounds",
        "inexact_min_only",
        "neg_zero_min",
        "max_bounds",
        "tiny_bounds",
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_scalar_ops(device, ttnn_op, min_val, max_val):
    """output_i = clamp(x_i, min_val, max_val) with scalar float/int bounds.

    The inexact_* cases are the only ones that exercise the fp32->bf16
    truncation (-0.1 truncates to -0.09960938, rounds to -0.10009766).
    max_bounds/tiny_bounds cover the conversion boundary classes rather than
    round numbers. neg_zero_min pins the signed-zero canonicalization, which
    is why the comparison is by bit pattern.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(
        input_tensor, _to_device_bf16_scalar(min_val), _to_device_bf16_scalar(max_val), device=device
    )

    tt_result = ttnn_op(tt_in, min_val, max_val)
    result = ttnn.to_torch(tt_result)

    golden = _canonicalize_negative_zero(golden)
    _assert_bitwise_equal(result, golden, f"{ttnn_op.__name__}(min={min_val}, max={max_val})")


@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_tensor_bounds(device, ttnn_op):
    """output_i = clamp(x_i, min_i, max_i) with per-element Tensor bounds.

    The bounds must be decorrelated from the input: min=-|x|, max=|x| holds by
    construction, making the clamp an identity that an op returning its input
    would pass. Opposing rolls give each element unrelated bounds, and the
    assertion below pins that a substantial fraction really are clamped.

    Rolls rather than arithmetic: an earlier `max = |rolled| * 0.5` walked the
    lowest normal binade into subnormals, which the device flushes on load but
    the golden clamps against, so 64 elements disagreed for reasons unrelated
    to clamp. A roll only permutes exact grid values.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    input_tensor = B.unsqueeze(0).unsqueeze(0)
    min_tensor = (-B.roll(64, dims=1).abs()).unsqueeze(0).unsqueeze(0)
    max_tensor = B.roll(-64, dims=1).abs().unsqueeze(0).unsqueeze(0)

    for name, bound in (("min", min_tensor), ("max", max_tensor)):
        subnormal = (bound.abs() > 0) & (bound.abs() < TINY_BF16)
        assert not subnormal.any(), (
            f"{int(subnormal.sum().item())} {name} bounds are bf16 subnormals; the device "
            f"flushes those on load but the golden does not, so the comparison below would "
            f"fail for reasons unrelated to {ttnn_op.__name__}"
        )

    tt_in = to_tt_tensor(input_tensor, device)
    tt_min = to_tt_tensor(min_tensor, device)
    tt_max = to_tt_tensor(max_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_tensor, max_tensor, device=device)

    clamped = int((golden != input_tensor).sum().item())
    assert clamped > golden.numel() // 10, (
        f"Tensor bounds are too loose to be a meaningful test: only {clamped} of "
        f"{golden.numel()} elements are actually clamped"
    )

    tt_result = ttnn_op(tt_in, tt_min, tt_max)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(Tensor min/max) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


def _tril_triu_input():
    """A (1, 1, 96, 96) tensor (3x3 tiles) sampling the full bf16 range.

    The grid is in ascending bit-pattern order, so a leading slice is entirely
    non-negative and below 2.8e-17. tril/triu are index masking so values are
    mostly incidental, but a sign-handling bug would be invisible against an
    all-positive input; striding by a coprime number walks the whole range.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256) -- 65,536 unique values
    strided = B.flatten()[::7][: 96 * 96]
    assert (strided < 0).any() and (strided.abs() > 1.0).any(), "tril/triu input is not full-range"
    return strided.view(1, 1, 96, 96)


@pytest.mark.parametrize("diagonal", [-100, -50, -10, -1, 0, 1, 10, 50, 100])
@pytest.mark.parametrize("ttnn_op", [ttnn.tril, ttnn.triu])
def test_tril_triu_ops(device, ttnn_op, diagonal):
    """Index-dependent masking, not a per-value sweep. diagonal spans from
    below the matrix to above it, including exactly at its bounds (+-96).

    NOTE: the generic golden wrapper forwards only *args to the underlying
    torch function, silently dropping keyword arguments, so `diagonal` must be
    passed positionally or the golden falls back to diagonal=0.
    """
    input_tensor = _tril_triu_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, diagonal, device=device)

    tt_result = ttnn_op(tt_in, diagonal=diagonal)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(diagonal={diagonal}) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


def test_bitcast_op(device):
    """bitcast reinterprets bits without conversion; bfloat16 -> uint16 is the
    only same-bit-width target.

    This is the one op here that is *about* bit patterns, so it is the one
    that needs include_spl_values=True: the default rewrites -0, both
    infinities and all 254 NaN encodings to +0. Subnormals are flushed to +0
    on both branches, so they never reach the device as subnormals and need no
    mask.
    """
    input_tensor = generate_bfloat16_bits(include_spl_values=True).unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)

    tt_result = ttnn.bitcast(tt_in, ttnn.uint16)
    result = ttnn.to_torch(tt_result)

    # Bounce through int16: torch has limited uint16 support on some builds.
    golden_bits = input_tensor.view(torch.int16)
    result_bits = result.to(torch.int16) if result.dtype != torch.int16 else result

    assert torch.equal(result_bits, golden_bits), (
        f"bitcast(bfloat16 -> uint16) diverged for "
        f"{int((result_bits != golden_bits).sum().item())} of {result_bits.numel()} elements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# threshold, round, selu, rdiv — ULP-gated
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("threshold_val", [-100.0, -1.0, 0.0, 0.5, 3.0, -0.1])
@pytest.mark.parametrize("value", [-100.0, 0.0, -0.0, 1.0, 100.0, 0.66])
def test_threshold_op(device, threshold_val, value):
    """output_i = x_i if x_i > threshold_val else value.

    A pure select, so bit-exactness is the contract -- the same one the
    removed test_unary_threshold_ttnn asserted over 24 pairs; all 36
    combinations are crossed here. The attached golden RNE-quantizes `value`
    and relaxes itself to ULP<=1, but feeding it the bf16 scalar the device
    actually uses makes that a no-op and recovers exactness.

    The two scalars take different conversion paths (`value` truncates,
    `threshold` rounds), and `value=-0.0` comes back as +0.0; both are quirks
    only an exhaustive sweep and a bit-pattern compare can see.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.threshold)
    golden = golden_function(
        input_tensor, _to_device_bf16_comparand(threshold_val), _to_device_bf16_scalar(value), device=device
    )

    tt_result = ttnn.threshold(tt_in, threshold_val, value)
    result = ttnn.to_torch(tt_result)

    golden = _canonicalize_negative_zero(golden)
    _assert_bitwise_equal(result, golden, f"threshold(threshold={threshold_val}, value={value})")


@pytest.mark.parametrize("decimals", [None, 0, -1, -3, 6, -6, 7])
def test_round_op(device, decimals):
    """Exhaustive normal bf16 sweep. Both endpoints of the supported [-6, 7]
    range are included, since that is where an off-by-one in the range check
    or the 10**decimals scaling would show up.

    Large positive `decimals` overflows the top of the bf16 range (7.8% of the
    grid at 6, 9.1% at 7); those are classified rather than compared."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.round)
    if decimals is None:
        golden = golden_function(input_tensor, device=device)
        tt_result = ttnn.round(tt_in)
    else:
        golden = golden_function(input_tensor, decimals=decimals, device=device)
        tt_result = ttnn.round(tt_in, decimals=decimals)
    result = ttnn.to_torch(tt_result)

    _assert_nonfinite_agreement(golden, result, f"round(decimals={decimals})")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, f"round(decimals={decimals}) overflow", max_fraction=0.15)
    assert_with_ulp(expected_result=golden[finite], actual_result=result[finite], ulp_threshold=1)


# selu(x) = scale * (x if x > 0 else alpha * (exp(x) - 1)). The attached golden
# ignores scale/alpha, so a scale/alpha-aware reference is computed here.
def _selu_reference(x, scale, alpha):
    x64 = x.to(torch.float64)
    pos = x64
    neg = alpha * (torch.expm1(x64))
    return (scale * torch.where(x64 > 0, pos, neg)).to(torch.bfloat16)


@pytest.mark.parametrize(
    "scale, alpha",
    [(1.0507, 1.67326), (1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
    ids=["default", "identity", "scale2_alpha_half", "scale_half_alpha2"],
)
def test_selu_op(device, scale, alpha):
    """Exhaustive normal bf16 sweep at ULP<=1, excluding only the FTZ boundary.

    The expm1 cancellation band earlier revisions inherited from the removed
    test_selu_arange is gone: measured against the reference above, selu is
    bit-exact over all 65,536 patterns for every scale/alpha here, so the
    quarter of the grid it masked is now inside the strict gate.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = _selu_reference(input_tensor, scale, alpha)

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    ftz = _ftz_boundary_mask(golden)
    _assert_excluded_region(ftz, "selu FTZ boundary", max_fraction=0.05)
    _assert_ftz_region_is_zeroish(result, ftz, "selu")

    keep = ~ftz
    assert_with_ulp(expected_result=golden[keep], actual_result=result[keep], ulp_threshold=1, allow_nonfinite=True)


# Above this |x|, reciprocal(x) underflows and is flushed to zero before the
# multiply, zeroing value * reciprocal(x) independent of value.
_RDIV_RECIP_FTZ_THRESHOLD = 1.0 / torch.finfo(torch.bfloat16).tiny


def _rdiv_safe_mask(input_tensor):
    """Elements where rdiv's accuracy is well-defined. Two exclusions, both
    needed on every rdiv test:
      - |x| >= 1/tiny: the reciprocal-FTZ band above (512 elements).
      - x == 0: value/0 is +-inf, and the comparison helpers substitute
        non-finites per-side rather than comparing them, so these would be
        uncheckable rather than checked (512 elements). The division-by-zero
        contract is asserted explicitly by _assert_rdiv_div_by_zero instead.
    """
    return (input_tensor.abs() < _RDIV_RECIP_FTZ_THRESHOLD) & (input_tensor != 0)


def _assert_rdiv_div_by_zero(input_tensor, result, value):
    """value/0 must be the infinity carrying sign(value), not NaN or a finite.
    Checked here rather than left to allow_nonfinite/PCC, neither of which
    actually compares non-finite elements."""
    zeros = input_tensor == 0
    if not zeros.any() or value == 0.0:
        return
    expected = math.inf if value > 0 else -math.inf
    at_zero = result[zeros]
    assert torch.equal(at_zero, torch.full_like(at_zero, expected)), (
        f"rdiv({value}) / 0 should be {expected} for all {int(zeros.sum().item())} zero elements, "
        f"got {int((at_zero != expected).sum().item())} mismatches"
    )


@pytest.mark.parametrize("value", [1.0, -1.0, 2.5, 100.0])
def test_rdiv_op_none_mode(device, value):
    """output_i = value / x_i (no rounding). Matches test_unary_rdiv_ttnn's
    ULP<=3 convention over the well-defined domain (see _rdiv_safe_mask), with
    the division-by-zero result asserted separately."""
    B = generate_bfloat16_bits(include_spl_values=False)
    input_tensor = B.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden = golden_function(input_tensor, value, rounding_mode=None, device=device)

    tt_result = ttnn.rdiv(tt_in, value, rounding_mode=None)
    result = ttnn.to_torch(tt_result)

    _assert_rdiv_div_by_zero(input_tensor, result, value)

    safe = _rdiv_safe_mask(input_tensor)
    _assert_excluded_region(~safe, f"rdiv({value}) domain", max_fraction=0.05)
    assert_with_ulp(expected_result=golden[safe], actual_result=result[safe], ulp_threshold=3, allow_nonfinite=True)


@pytest.mark.parametrize("value", [1.0, -1.0, 2.5, 100.0])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_rdiv_op_rounded_modes(device, value, rounding_mode):
    """output_i = floor|trunc(value / x_i). PCC-gated rather than ULP-gated: a
    single bf16-ULP difference in the division can flip the result by a whole
    integer near a boundary (matches test_unary_rdiv_ttnn's convention).

    Uses the same domain mask as test_rdiv_op_none_mode -- without it the 512
    infinities and the reciprocal-FTZ band reach assert_with_pcc, which zeroes
    non-finites independently on each side, so a wrong result in exactly the
    region this test nominally covers could not fail it."""
    B = generate_bfloat16_bits(include_spl_values=False)
    input_tensor = B.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden = golden_function(input_tensor, value, rounding_mode=rounding_mode, device=device)

    tt_result = ttnn.rdiv(tt_in, value, rounding_mode=rounding_mode)
    result = ttnn.to_torch(tt_result)

    _assert_rdiv_div_by_zero(input_tensor, result, value)

    safe = _rdiv_safe_mask(input_tensor)
    _assert_nonfinite_agreement(golden[safe], result[safe], f"rdiv({value}, {rounding_mode})")
    keep = safe & torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~keep, f"rdiv({value}, {rounding_mode}) domain", max_fraction=0.05)
    assert_with_pcc(golden[keep], result[keep], pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# tanh, sigmoid_accurate, xielu, softplus, logit, polygamma — PCC-gated
# ─────────────────────────────────────────────────────────────────────────────


# tanh and sigmoid have both saturated well before |x| = 10 in bf16 and have no
# underflowing sub-computation, so this is the well-conditioned window where a
# per-element ULP bound is meaningful. It also contains the randn range the
# removed test_activation.py::test_sigmoid/test_sigmoid_accurate used.
_SATURATING_ULP_WINDOW = 10.0


@pytest.mark.parametrize(
    "ttnn_op, golden_fn",
    [(ttnn.tanh, torch.tanh), (ttnn.sigmoid_accurate, torch.sigmoid)],
    ids=["tanh", "sigmoid_accurate"],
)
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_tanh_sigmoid_accurate_ops(device, ttnn_op, golden_fn, fast_and_approximate_mode):
    """Full-domain dual-mode sweep. sigmoid_accurate is deprecated but still
    exercises the same parameter.

    Accurate mode gets both gates: PCC over the whole grid plus ULP<=2 over
    |x| <= 10. PCC alone would loosen what the removed test_activation.py
    tests asserted, and one correlation number over the full dynamic range
    cannot see a regression confined to the small-|x| region where these are
    actually used.

    Fast mode is gated at a deliberately loose PCC>=0.99 -- leaving it
    unasserted means paying for a 65k-element device run that stays green no
    matter what comes back. Doing so turned up that sigmoid_accurate's fast
    path shares the fast-exp overflow documented for AccurateWithFastExp
    (full-grid PCC 0.07; 0.99996 once x >= 174 is excluded). tanh's fast mode
    is unaffected, so the exclusion is applied only where that path is in play.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)

    tt_result = ttnn_op(tt_in, fast_and_approximate_mode=fast_and_approximate_mode)
    result = ttnn.to_torch(tt_result)

    label = f"{ttnn_op.__name__}(fast={fast_and_approximate_mode})"
    _assert_nonfinite_agreement(golden, result, label)
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, f"{label} non-finite", max_fraction=0.05)

    if fast_and_approximate_mode:
        keep = finite
        if ttnn_op is ttnn.sigmoid_accurate:
            keep = keep & (input_tensor < _SIGMOID_FAST_EXP_UNSAFE_X)
            _assert_excluded_region(~keep, f"{label} fast-exp overflow", max_fraction=0.30)
        assert_with_pcc(golden[keep], result[keep], pcc=0.99)
        return

    assert_with_pcc(golden[finite], result[finite], pcc=0.999)

    # tanh(x) ~= x near 0, so the small-|x| end of the window runs into the same
    # FTZ boundary as selu/sigmoid and is excluded the same way.
    window = finite & (input_tensor.abs() <= _SATURATING_ULP_WINDOW) & ~_ftz_boundary_mask(golden)
    assert_with_ulp(expected_result=golden[window], actual_result=result[window], ulp_threshold=2)


@pytest.mark.parametrize(
    "alpha_p, alpha_n", [(0.8, 0.8), (1.0, 1.0), (0.5, 1.5), (2.0, 0.1)], ids=["default", "one", "mix1", "mix2"]
)
def test_xielu_op(device, alpha_p, alpha_n):
    """xIELU: x>0 -> alpha_p*x^2 + 0.5*x ; x<=0 -> alpha_n*(expm1(min(x,eps))) -
    alpha_n*x + 0.5*x, with beta=0.5, eps=-1e-6 fixed (see _golden_function_xielu)."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.xielu)
    golden = golden_function(input_tensor, alpha_p=alpha_p, alpha_n=alpha_n, device=device)

    tt_result = ttnn.xielu(tt_in, alpha_p=alpha_p, alpha_n=alpha_n)
    result = ttnn.to_torch(tt_result)

    # alpha_p * x^2 overflows across roughly the top eighth of the grid.
    _assert_nonfinite_agreement(golden, result, f"xielu(alpha_p={alpha_p}, alpha_n={alpha_n})")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, "xielu non-finite", max_fraction=0.25)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


@pytest.mark.parametrize(
    "beta, threshold_val",
    [(1.0, 20.0), (0.5, 20.0), (2.0, 10.0), (1.0, 5.0)],
    ids=["default", "beta_half", "beta2", "low_threshold"],
)
def test_softplus_op(device, beta, threshold_val):
    """softplus(x) = (1/beta) * log(1 + exp(beta*x)), replaced by the linear
    identity x for beta*x > threshold (numerical-stability branch). A low
    threshold pushes more of the sweep through that branch."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softplus)
    golden = golden_function(input_tensor, beta=beta, threshold=threshold_val, device=device)

    tt_result = ttnn.softplus(tt_in, beta=beta, threshold=threshold_val)
    result = ttnn.to_torch(tt_result)

    _assert_nonfinite_agreement(golden, result, f"softplus(beta={beta}, threshold={threshold_val})")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, "softplus non-finite", max_fraction=0.05)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


@pytest.mark.parametrize("eps", [None, 1e-6, 0.1, 0.5, 0.9], ids=["none", "tiny", "small", "half", "gt_half"])
def test_logit_op(device, eps):
    """logit(x) = log(x / (1-x)), domain (0, 1). eps clamps x into [eps, 1-eps]
    first (eps > 0.5 uses a manual-clamp golden branch to avoid UB). When eps
    is None, out-of-[0,1] inputs are excluded via a domain mask."""
    input_tensor = generate_bfloat16_bits_in_range(-2.0, 2.0)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    kwargs = {} if eps is None else {"eps": eps}
    golden = golden_function(input_tensor, device=device, **kwargs)

    tt_result = ttnn.logit(tt_in, **kwargs)
    result = ttnn.to_torch(tt_result)

    if eps is None:
        in_domain = (input_tensor > 0.0) & (input_tensor < 1.0)
    else:
        in_domain = torch.ones_like(input_tensor, dtype=torch.bool)

    # Only classify non-finites in-domain: outside it the golden is
    # legitimately NaN/inf and the device need not match.
    _assert_nonfinite_agreement(golden[in_domain], result[in_domain], f"logit(eps={eps})")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    keep = in_domain & finite
    _assert_excluded_region(in_domain & ~finite, "logit in-domain non-finite", max_fraction=0.05)
    assert_with_pcc(golden[keep], result[keep], pcc=0.999)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 10])
def test_polygamma_op(device, k):
    """polygamma(k, x), domain x > 0, supported k range 1..10. Exhaustive
    positive-normal bf16 sweep, PCC-gated."""
    input_tensor = generate_bfloat16_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden = golden_function(input_tensor, k, device=device)

    tt_result = ttnn.polygamma(tt_in, k)
    result = ttnn.to_torch(tt_result)

    # polygamma(k, x) ~ k!/x^(k+1) as x -> 0, so it overflows across 25% of the
    # sweep at k=1 rising to 46% at k=10 -- inherent, but it leaves what
    # remains weighted towards the large-|x| tail, which is most of the reason
    # test_polygamma_op_ulp_window exists.
    _assert_nonfinite_agreement(golden, result, f"polygamma(k={k})")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, f"polygamma(k={k}) non-finite", max_fraction=0.55)
    assert_with_pcc(golden[finite], result[finite], pcc=0.99)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 10])
def test_polygamma_op_ulp_window(device, k):
    """The [1, 10] window at ULP<=1, which is what the removed
    test_math.py::test_polygamma asserted. Kept alongside the exhaustive PCC
    sweep rather than folded into it: polygamma is ~1e38 as x approaches 0, so
    a correlation over the whole positive range is dominated by those elements
    and a regression confined to [1, 10] would not move it."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 10.0).unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden = golden_function(input_tensor, k, device=device)

    tt_result = ttnn.polygamma(tt_in, k)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# sigmoid — vector_mode x SigmoidMode (structural: own dedicated tests)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode, pcc",
    [
        (ttnn.SigmoidMode.Accurate, None),  # ULP-gated instead (ulp<=2)
        (ttnn.SigmoidMode.AccurateWithFastExp, 0.999),
        (ttnn.SigmoidMode.FastApproximate, 0.99),
    ],
    ids=["accurate", "accurate_fast_exp", "fast_approximate"],
)
def test_sigmoid_op(device, mode, pcc):
    """Exhaustive normal bf16 sweep for every SigmoidMode, at the default
    vector_mode=4 (RC, full tile processed) -- vector_mode is not swept here
    for the reason given in the module docstring.

    Accurate is ULP-gated (<=2); the other two are PCC-gated since both trade
    accuracy for speed by design. All three exclude the FTZ boundary, and
    AccurateWithFastExp additionally excludes x >= 174.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = torch.sigmoid(input_tensor.float()).to(torch.bfloat16)

    tt_result = ttnn.sigmoid(tt_in, vector_mode=4, mode=mode)
    result = ttnn.to_torch(tt_result)

    ftz = _ftz_boundary_mask(golden)
    _assert_ftz_region_is_zeroish(result, ftz, f"sigmoid({mode})")

    keep = ~ftz
    if mode == ttnn.SigmoidMode.AccurateWithFastExp:
        keep = keep & (input_tensor < _SIGMOID_FAST_EXP_UNSAFE_X)

    # sigmoid underflows to 0 below x ~= -88, so the FTZ band alone is ~24% of
    # the grid and the fast-exp overflow band another ~24%.
    _assert_excluded_region(~keep, f"sigmoid({mode}) excluded region", max_fraction=0.55)

    golden_keep = golden[keep]
    result_keep = result[keep]

    if pcc is not None:
        assert_with_pcc(golden_keep, result_keep, pcc=pcc)
    else:
        assert_with_ulp(expected_result=golden_keep, actual_result=result_keep, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# unary_chain — composition of individual op goldens (structural)
# ─────────────────────────────────────────────────────────────────────────────


def test_unary_chain_relu_exp_square(device):
    """[RELU, EXP(accurate), POWER(2)] == square(exp(relu(x))). PCC-gated since
    EXP is transcendental; overflow is excluded via a finite-both-sides mask."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    x64 = input_tensor.to(torch.float64)
    golden = torch.relu(x64).exp().pow(2).to(torch.bfloat16)

    # exp(relu(x)) overflows above x ~= 88, so half the positive half of the
    # grid goes non-finite.
    _assert_nonfinite_agreement(golden, result, "unary_chain([RELU, EXP, POWER(2)])")
    finite = torch.isfinite(golden) & torch.isfinite(result)
    _assert_excluded_region(~finite, "unary_chain([RELU, EXP, POWER(2)]) non-finite", max_fraction=0.55)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


def test_unary_chain_square_sqrt(device):
    """[SQUARE, SQRT] == |x| (up to bf16 rounding). ULP-gated; excludes both the
    band where squaring overflows to inf and the band where it underflows to
    zero -- rounded to bf16 at each stage, x^2 underflows well before |x| would."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    absx = input_tensor.abs()
    safe = (absx >= math.sqrt(TINY_BF16)) & (absx < math.sqrt(MAX_BF16_VAL))
    golden = input_tensor.abs().to(torch.bfloat16)

    # |x| < sqrt(tiny) is half the grid by bit-pattern count, since bf16 is
    # exponent-dense near zero. Expected here, but bounded so it can't widen.
    _assert_excluded_region(~safe, "unary_chain([SQUARE, SQRT]) under/overflow", max_fraction=0.55)

    assert_with_ulp(expected_result=golden[safe], actual_result=result[safe], ulp_threshold=2, allow_nonfinite=True)


def test_unary_chain_relu_typecast(device):
    """[RELU, TYPECAST(bfloat16 -> float32)] exercises unary_chain's
    dtype-changing last-op path (output dtype comes from the chain's final
    TYPECAST/BITCAST op). The widening should be exact, so the comparison is
    bit-for-bit, except for the near-subnormal band noted in the module
    docstring."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    assert (
        result.dtype == torch.float32
    ), f"expected float32 output from a TYPECAST-terminated chain, got {result.dtype}"

    golden = torch.relu(input_tensor).to(torch.float32)

    safe = input_tensor.abs() >= 2.0**-115
    _assert_excluded_region(~safe, "unary_chain([RELU, TYPECAST]) near-subnormal", max_fraction=0.10)
    golden_safe = golden[safe]
    result_safe = result[safe]
    assert torch.equal(result_safe, golden_safe), (
        f"[RELU, TYPECAST(bf16->fp32)] diverged for {int((result_safe != golden_safe).sum().item())} "
        f"of {result_safe.numel()} elements (|x| >= 2**-115 only)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# rank / shape coverage
# ─────────────────────────────────────────────────────────────────────────────
#
# Exhaustive *value* coverage and *shape* coverage are separate axes, and the
# sweeps above only ever run the single shape that makes the bf16 grid
# tile-aligned. The tests this file replaces did carry rank variety
# (test_hardtanh.py ran 2D (64, 64) and 5D (2, 2, 3, 256, 256);
# test_unary_threshold_ttnn ran (1, 3, 320, 384)), so it is reproduced here
# rather than dropped: same ops, same shapes, one small deterministic input each.

_RANK_SHAPES = [
    (32, 32),
    (64, 64),
    (1, 1, 32, 32),
    (1, 3, 320, 384),
    (2, 2, 3, 256, 256),
]


@pytest.mark.parametrize("shape", _RANK_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_scalar_param_ops_rank_coverage(device, shape):
    """hardtanh/clamp/clip/threshold on ranks 2 through 5, bit-for-bit.

    Each golden is fed the way *it* accepts parameters, which is not uniform:
    hardtanh's drops positional bounds and silently falls back to (-1, 1), so
    it needs keywords, while tril/triu are the exact opposite and
    clamp/clip/threshold take theirs positionally. Getting hardtanh wrong here
    does not raise -- it quietly compares against the wrong bounds.
    """
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16) * 4.0

    tt_in = to_tt_tensor(input_tensor, device)

    min_val, max_val = _to_device_bf16_scalar(-1.0), _to_device_bf16_scalar(0.66)
    for ttnn_op, args, kwargs, golden_args, golden_kwargs in (
        (ttnn.hardtanh, (), {"min_val": -1.0, "max_val": 0.66}, (), {"min_val": min_val, "max_val": max_val}),
        (ttnn.clamp, (-1.0, 0.66), {}, (min_val, max_val), {}),
        (ttnn.clip, (-1.0, 0.66), {}, (min_val, max_val), {}),
        (ttnn.threshold, (0.5, 0.66), {}, (_to_device_bf16_comparand(0.5), max_val), {}),
    ):
        golden_function = ttnn.get_golden_function(ttnn_op)
        golden = golden_function(input_tensor, *golden_args, device=device, **golden_kwargs)
        result = ttnn.to_torch(ttnn_op(tt_in, *args, **kwargs))

        assert torch.equal(result, golden), (
            f"{ttnn_op.__name__} on shape {tuple(shape)} diverged for "
            f"{int((result != golden).sum().item())} of {result.numel()} elements"
        )


@pytest.mark.parametrize("shape", _RANK_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("decimals", [None, -1, 6])
def test_round_rank_coverage(device, shape, decimals):
    """round on ranks 2 through 5, reproducing the multi-dim shapes of the
    removed test_round.py row. Separate from the loop above because round is
    ULP-gated rather than exact."""
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16) * 4.0

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.round)
    if decimals is None:
        golden = golden_function(input_tensor, device=device)
        result = ttnn.to_torch(ttnn.round(tt_in))
    else:
        golden = golden_function(input_tensor, decimals=decimals, device=device)
        result = ttnn.to_torch(ttnn.round(tt_in, decimals=decimals))

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize("shape", _RANK_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("diagonal", [-1, 0, 1])
@pytest.mark.parametrize("ttnn_op", [ttnn.tril, ttnn.triu])
def test_tril_triu_rank_coverage(device, ttnn_op, shape, diagonal):
    """tril/triu mask the trailing two dims, so rank and batch extents matter
    more for them than for the elementwise ops. `diagonal` is passed
    positionally to the golden for the reason noted in test_tril_triu_ops."""
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, diagonal, device=device)

    result = ttnn.to_torch(ttnn_op(tt_in, diagonal=diagonal))

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(diagonal={diagonal}) on shape {tuple(shape)} diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )
