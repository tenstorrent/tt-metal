# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import pytest
import ttnn
from ttnn.comparison import ulp
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    assert_finite_matches_golden,
    assert_ftz_band,
    assert_ftz_then_ulp,
    generate_float32_bits,
    generate_float32_bits_in_range,
    to_float32_tt_tensor,
    SMALLEST_NORMAL_BF16,
    MAX_BF16,
)

pytestmark = pytest.mark.use_module_device


"""
Category 6: ops with multiple or special parameters

| #  | Op                    | Parameters                                          |
|----|-----------------------|-----------------------------------------------------|
| 1  | ttnn.softplus         | beta (1.0), threshold (20.0)                        |
| 2  | ttnn.xielu            | alpha_p (0.8), alpha_n (0.8)                        |
| 3  | ttnn.tanh             | fast_and_approximate_mode (False)                   |
| 4  | ttnn.sigmoid_accurate | fast_and_approximate_mode (False), deprecated       |
| 5  | ttnn.sigmoid          | vector_mode (4), mode (SigmoidMode.Accurate)        |
| 6  | ttnn.unary_chain      | ops_chain (list of UnaryWithParam)                  |
| 7  | ttnn.clamp            | min, max (scalar or Tensor)                         |
| 8  | ttnn.clip             | min, max (scalar or Tensor)                         |
| 9  | ttnn.selu             | scale (1.0507), alpha (1.67326)                     |
| 10 | ttnn.hardtanh         | min_val (-1.0), max_val (1.0)                       |
| 11 | ttnn.threshold        | threshold, value                                    |
| 12 | ttnn.tril             | diagonal (0)                                        |
| 13 | ttnn.triu             | diagonal (0)                                        |
| 14 | ttnn.round            | decimals (0; supported -6..7)                       |
| 15 | ttnn.polygamma        | k (1..10)                                           |
| 16 | ttnn.logit            | eps (None)                                          |
| 17 | ttnn.rdiv             | value, rounding_mode (None, "floor", "trunc")       |
| 18 | ttnn.bitcast          | dtype (float32 -> uint32 here)                      |

Accuracy criteria (FTZ = flush-to-zero, SNB = smallest normal bfloat16)

| Op                       | Input range / mode             | Check                          |
|--------------------------|--------------------------------|--------------------------------|
| hardtanh, clamp, clip,   | all                            | exact                          |
| threshold, tril, triu,   |                                |                                |
| bitcast                  |                                |                                |
| round                    | decimals None, 0               | exact                          |
|                          | decimals 6, 7                  | ULP ≤ 1                        |
|                          | decimals < 0                   | abs ≤ 10^-decimals + 2 ULP     |
| selu                     | golden < SNB                   | exactly 0                      |
|                          | default scale/alpha            | ULP ≤ 3                        |
|                          | scale * alpha = 1              | ULP ≤ 1                        |
| rdiv                     | x = 0                          | ±inf                           |
|                          | |x| > 1/SNB                    | exactly 0                      |
|                          | |x| = 1/SNB                    | 0 or golden                    |
|                          | rounding_mode None             | ULP ≤ 1; BH 2 for value ≠ ±1   |
|                          | floor, trunc                   | abs ≤ max(1, 2 ULP) vs float64 |
| tanh, sigmoid_accurate,  | golden < SNB                   | exactly 0                      |
| sigmoid Accurate         | accurate                       | ULP ≤ 2                        |
| tanh                     | fast                           | abs ≤ 0.019                    |
| sigmoid_accurate,        | fast / AccurateWithFastExp     | rtol 0.031                     |
| sigmoid                  |                                |                                |
| sigmoid                  | FastApproximate                | abs ≤ 0.12                     |
| xielu                    | x > 0                          | ULP ≤ 2 vs float64             |
|                          | x <= 0                         | rtol 1e-5, atol 1e-6           |
| softplus                 | beta * x <= -2^23              | exactly 0                      |
|                          | beta * x <= -87.5, golden < SNB| exactly 0                      |
|                          | beta * x > threshold           | exact                          |
|                          | elsewhere                      | rtol 5.1e-4                    |
| logit                    | |golden| < 1                   | abs ≤ 2^-23                    |
|                          | |golden| >= 1                  | ULP ≤ 1                        |
| polygamma                | x > 0                          | rtol 1e-6 (k ≤ 4) to 6e-3      |
|                          |                                | (k = 10) vs float64            |
|                          | x = 2^126 (k = 1)              | 0 or golden                    |
|                          | [1, 10]                        | WH ULP ≤ 2 (k = 1) to 11       |
|                          |                                | (k = 10); BH 3 to 20           |
| unary_chain              | [RELU, EXP, POWER(2)]          | ULP ≤ 3                        |
|                          | [SQUARE, SQRT]                 | exact for 2^-63 <= |x| < 2^64  |
|                          | [RELU, TYPECAST(bfloat16)]     | exact                          |

Scalar parameters reach the float32 kernel at full float32 precision, so
goldens are given the same Python floats. A -0.0 scalar comes back as +0.0
from threshold and clamp; the select ops are compared by bit pattern so the
sign of zero is part of the check.
"""


FLT_MAX = torch.finfo(torch.float32).max


def _canonicalize_negative_zero(tensor):
    """Map -0.0 to +0.0, which is what the device does to a -0.0 scalar parameter."""
    return torch.where(tensor == 0, torch.zeros_like(tensor), tensor)


def _assert_bitwise_equal(result, golden, label):
    """Compare float32 tensors by bit pattern; torch.equal treats +0.0 and -0.0 as equal."""
    result_bits = result.view(torch.int32)
    golden_bits = golden.view(torch.int32)
    assert torch.equal(result_bits, golden_bits), (
        f"{label} diverged for {int((result_bits != golden_bits).sum().item())} of "
        f"{result_bits.numel()} elements (compared as float32 bit patterns; "
        f"{int((result != golden).sum().item())} differ by value, the rest are signed-zero mismatches)"
    )


def _assert_inf_matches(golden, result, label):
    """Infinities must sit at the same positions with the same sign on both sides."""
    assert torch.equal(torch.isinf(golden), torch.isinf(result)), f"{label}: infinities at different positions"
    both_inf = torch.isinf(golden)
    assert torch.equal(
        torch.signbit(golden[both_inf]), torch.signbit(result[both_inf])
    ), f"{label}: infinities with different signs"


def _assert_abs_within(golden, result, bound, label):
    """No NaN, infinities match the golden, and |result - golden| <= bound on the finite rest."""
    assert not torch.isnan(golden).any(), f"{label}: golden contains NaN"
    assert_finite_matches_golden(golden, result, label)
    _assert_inf_matches(golden, result, label)
    finite = torch.isfinite(golden)
    golden, result, bound = golden[finite], result[finite], bound[finite]
    err = (result.double() - golden.double()).abs()
    over = ~(err <= bound.double())
    assert not over.any(), (
        f"{label}: {int(over.sum().item())} elements exceed the bound; worst |error| {err.max().item():g} "
        f"against a bound of {bound.double()[err.argmax()].item():g}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# hardtanh, clamp, clip, threshold, tril, triu, bitcast — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("min_val", [0.25, 0.5, 0.66, -1.0])
@pytest.mark.parametrize("max_val", [1.0, 2.5, 3.0, 6.6])
def test_hardtanh(device, min_val, max_val):
    """hardtanh(x, min_val, max_val) = clamp(x, min_val, max_val).

    | Input range | Check | Reason                                         |
    |-------------|-------|------------------------------------------------|
    | all         | exact | every output is x, min_val or max_val          |

    0.66 and 6.6 are not bfloat16 values; float32 keeps them exactly.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden = golden_function(input_tensor, min_val=min_val, max_val=max_val, device=device)

    result = ttnn.to_torch(ttnn.hardtanh(tt_in, min_val=min_val, max_val=max_val))
    assert_equal(golden, result)


def test_hardtanh_default_bounds(device):
    """hardtanh(x) with no bounds, i.e. (-1, 1).

    | Input range | Check               | Reason                                 |
    |-------------|---------------------|----------------------------------------|
    | all         | exact (bit pattern) | every other call passes bounds, and a  |
    |             |                     | lost bound falls back to the defaults  |
    |             |                     | silently                               |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
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
        (-5, 5),
        (2.0, 1.0),
        (-0.1, 0.66),
        (-0.1, None),
        (-0.0, 1.0),
        (-FLT_MAX, FLT_MAX),
        (-SMALLEST_NORMAL_BF16, SMALLEST_NORMAL_BF16),
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
def test_clamp_clip_scalar(device, ttnn_op, min_val, max_val):
    """clamp/clip(x, min_val, max_val) with scalar bounds.

    | Input range | Check               | Reason                                    |
    |-------------|---------------------|-------------------------------------------|
    | all         | exact (bit pattern) | every output is x or a bound; a -0.0      |
    |             |                     | bound comes back as +0.0                  |

    degenerate (min > max) collapses to a constant. inexact_* bounds are not
    bfloat16 values. max_bounds and tiny_bounds are the ends of the normal range.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_val, max_val, device=device)

    result = ttnn.to_torch(ttnn_op(tt_in, min_val, max_val))

    _assert_bitwise_equal(result, _canonicalize_negative_zero(golden), f"{ttnn_op.__name__}({min_val}, {max_val})")


@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_tensor_bounds(device, ttnn_op):
    """clamp/clip(x, min, max) with per-element Tensor bounds.

    | Input range | Check | Reason                                |
    |-------------|-------|---------------------------------------|
    | all         | exact | every output is x, min_i or max_i     |

    The bounds are the input rolled by ±64 columns, so each element gets
    bounds unrelated to its own value; min = -|x|, max = |x| would make the
    clamp an identity. More than 10% of the elements must actually be clamped.
    """
    input_tensor = generate_float32_bits()
    min_tensor = -input_tensor.roll(64, dims=1).abs()
    max_tensor = input_tensor.roll(-64, dims=1).abs()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    tt_min = to_float32_tt_tensor(min_tensor, device)
    tt_max = to_float32_tt_tensor(max_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_tensor, max_tensor, device=device)

    clamped = int((golden != input_tensor).sum().item())
    assert clamped > golden.numel() // 10, f"only {clamped} of {golden.numel()} elements are clamped"

    result = ttnn.to_torch(ttnn_op(tt_in, tt_min, tt_max))
    assert_equal(golden, result)


@pytest.mark.parametrize("threshold_val", [-100.0, -1.0, 0.0, 0.5, 3.0, -0.1])
@pytest.mark.parametrize("value", [-100.0, 0.0, -0.0, 1.0, 100.0, 0.66])
def test_threshold(device, threshold_val, value):
    """threshold(x, threshold, value) = x if x > threshold else value.

    | Input range | Check               | Reason                                   |
    |-------------|---------------------|------------------------------------------|
    | all         | exact (bit pattern) | every output is x or value; value = -0.0 |
    |             |                     | comes back as +0.0                       |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.threshold)
    golden = golden_function(input_tensor, threshold_val, value, device=device)

    result = ttnn.to_torch(ttnn.threshold(tt_in, threshold_val, value))

    _assert_bitwise_equal(result, _canonicalize_negative_zero(golden), f"threshold({threshold_val}, {value})")


def _tril_triu_input():
    """A 96 x 96 tensor drawn from the whole bfloat16 range.

    Every 7th lattice value, so the input has both signs and both small and
    large magnitudes; the leading slice of the lattice alone is all tiny and
    positive.
    """
    B = generate_float32_bits()
    strided = B.flatten()[::7][: 96 * 96]
    assert (strided < 0).any() and (strided.abs() > 1.0).any(), "tril/triu input is not full-range"
    return strided.view(96, 96)


@pytest.mark.parametrize("diagonal", [-100, -50, -10, -1, 0, 1, 10, 50, 100])
@pytest.mark.parametrize("ttnn_op", [ttnn.tril, ttnn.triu])
def test_tril_triu(device, ttnn_op, diagonal):
    """tril/triu(x, diagonal) on a 96 x 96 matrix.

    | diagonal         | Check | Reason                                   |
    |------------------|-------|------------------------------------------|
    | -100 .. 100      | exact | index masking; covers past both edges    |

    The golden wrapper drops keyword arguments, so diagonal is passed
    positionally.
    """
    input_tensor = _tril_triu_input()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, diagonal, device=device)

    result = ttnn.to_torch(ttnn_op(tt_in, diagonal=diagonal))
    assert_equal(golden, result)


def test_bitcast(device):
    """bitcast(x, uint32) reinterprets float32 bits.

    | Input                     | Check               | Reason                         |
    |---------------------------|---------------------|--------------------------------|
    | all 65,536 bfloat16 bit   | exact (bit pattern) | no conversion; -0, ±inf and    |
    | patterns, incl. specials  |                     | every NaN payload pass through |

    Subnormals are flushed to +0 before upload.
    """
    input_tensor = generate_float32_bits(include_spl_values=True)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    result = ttnn.to_torch(ttnn.bitcast(tt_in, ttnn.uint32))

    golden_bits = input_tensor.view(torch.int32)
    result_bits = result.view(torch.int32)
    assert torch.equal(result_bits, golden_bits), (
        f"bitcast(float32 -> uint32) diverged for "
        f"{int((result_bits != golden_bits).sum().item())} of {result_bits.numel()} elements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# round
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("decimals", [None, 0, -1, -3, -6, 6, 7])
def test_round(device, decimals):
    """round(x, decimals), including both ends of the supported -6..7 range.

    | decimals | Input range | Check                     | Reason                                 |
    |----------|-------------|---------------------------|----------------------------------------|
    | None, 0  | all         | exact                     | round to integer                       |
    | 6, 7     | overflow    | ±inf                      | x * 10^decimals overflows (7.8% of the |
    |          |             |                           | lattice at 6, 9.1% at 7)               |
    |          | elsewhere   | ULP ≤ 1                   | one rounding of the rescale            |
    | < 0      | all         | abs ≤ 10^-decimals        | near a tie, the device's or the        |
    |          |             | + 2 ULP                   | golden's float32 quotient can round to |
    |          |             |                           | the next step (62 ULP at decimals=-3,  |
    |          |             |                           | x = 1.447e8)                           |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.round)
    if decimals is None:
        golden = golden_function(input_tensor, device=device)
        result = ttnn.to_torch(ttnn.round(tt_in))
    else:
        golden = golden_function(input_tensor, decimals=decimals, device=device)
        result = ttnn.to_torch(ttnn.round(tt_in, decimals=decimals))

    if not decimals:
        assert_equal(golden, result)
        return

    label = f"round(decimals={decimals})"
    _assert_inf_matches(golden, result, label)
    finite = torch.isfinite(golden)
    golden, result = golden[finite], result[finite]
    if decimals > 0:
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)
    else:
        bound = 10.0**-decimals + 2 * ulp(golden)
        _assert_abs_within(golden, result, bound, label)


# ─────────────────────────────────────────────────────────────────────────────
# selu
# ─────────────────────────────────────────────────────────────────────────────


def _selu_reference(x, scale, alpha):
    """scale * (x if x > 0 else alpha * (exp(x) - 1)) in float64, rounded to float32."""
    x64 = x.double()
    return (scale * torch.where(x64 > 0, x64, alpha * torch.expm1(x64))).float()


@pytest.mark.parametrize(
    "scale, alpha, ulp_threshold",
    [(1.0507, 1.67326, 3), (1.0, 1.0, 1), (2.0, 0.5, 1), (0.5, 2.0, 1)],
    ids=["default", "identity", "scale2_alpha_half", "scale_half_alpha2"],
)
def test_selu(device, scale, alpha, ulp_threshold):
    """selu(x, scale, alpha) = scale * (x if x > 0 else alpha * (exp(x) - 1)).

    | scale, alpha       | Input range    | Check     | Reason                                 |
    |--------------------|----------------|-----------|----------------------------------------|
    | all                | reference < SNB| exactly 0 | FTZ                                    |
    | default            | elsewhere      | ULP ≤ 3   | scale and alpha each add a rounding    |
    |                    |                |           | (3 ULP at x = -0.332, -6.3e-8)         |
    | scale * alpha = 1  | elsewhere      | ULP ≤ 1   | the scalings cancel exactly            |

    The reference is float64 rather than the golden function, so this does
    not depend on how the golden handles scale and alpha.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden = _selu_reference(input_tensor, scale, alpha)

    result = ttnn.to_torch(ttnn.selu(tt_in, scale=scale, alpha=alpha))

    assert_ftz_then_ulp(golden, result, ulp_threshold, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# rdiv — value / x
# ─────────────────────────────────────────────────────────────────────────────

# Above this |x| the reciprocal is subnormal and flushed before the multiply.
RDIV_RECIP_FTZ = 1.0 / SMALLEST_NORMAL_BF16


def _assert_zero_or_golden(golden, result, mask, label):
    """Where 1/x is exactly SNB (|x| = 2^126), Blackhole flushes it to 0 and Wormhole keeps it."""
    ok = (result[mask] == 0) | (result[mask] == golden[mask])
    assert ok.all(), f"{label}: |x| = 2^126 returned neither 0 nor the golden for {int((~ok).sum())} elements"


def _assert_rdiv_edges(input_tensor, result, value):
    """value / 0 is the infinity with the sign of value; |x| > 1/SNB gives exactly 0."""
    zeros = input_tensor == 0
    expected = math.inf if value > 0 else -math.inf
    assert torch.equal(result[zeros], torch.full_like(result[zeros], expected)), f"rdiv({value}) / 0 is not {expected}"
    assert_ftz_band(result, input_tensor.abs() > RDIV_RECIP_FTZ, "|x| > 1/SNB")


# value -> (Wormhole, Blackhole) ULP threshold
RDIV_ULP = {1.0: (1, 1), -1.0: (1, 1), 2.5: (1, 2), 100.0: (1, 2)}


@pytest.mark.parametrize("value", list(RDIV_ULP))
def test_rdiv(device, value):
    """rdiv(x, value) = value / x, no rounding.

    | value | Input range   | Check        | Reason                                     |
    |-------|---------------|--------------|--------------------------------------------|
    | all   | x = 0         | ±inf         | sign of value                              |
    |       | |x| > 1/SNB   | exactly 0    | value * reciprocal(x); the reciprocal is   |
    |       |               |              | FTZ'd                                      |
    |       | |x| = 1/SNB   | 0 or golden  | the reciprocal is exactly SNB: Blackhole   |
    |       |               |              | flushes it, Wormhole keeps it              |
    | ±1    | elsewhere     | ULP ≤ 1      | only the reciprocal rounds                 |
    | other | elsewhere     | WH ULP ≤ 1   | reciprocal and multiply each round; on BH  |
    |       |               | BH ULP ≤ 2   | 1512 (2.5) and 496 (100) elements reach 2  |

    value / x overflows to ±inf on both sides for tiny x.
    """
    wh_ulp, bh_ulp = RDIV_ULP[value]
    ulp_threshold = bh_ulp if is_blackhole() else wh_ulp
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden = golden_function(input_tensor, value, rounding_mode=None, device=device)

    result = ttnn.to_torch(ttnn.rdiv(tt_in, value, rounding_mode=None))

    _assert_rdiv_edges(input_tensor, result, value)
    recip_snb = input_tensor.abs() == RDIV_RECIP_FTZ
    _assert_zero_or_golden(golden, result, recip_snb, f"rdiv({value})")
    keep = (input_tensor != 0) & (input_tensor.abs() < RDIV_RECIP_FTZ)
    assert_with_ulp(
        expected_result=golden[keep], actual_result=result[keep], ulp_threshold=ulp_threshold, allow_nonfinite=True
    )


@pytest.mark.parametrize("value", [1.0, -1.0, 2.5, 100.0])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_rdiv_rounded(device, value, rounding_mode):
    """rdiv(x, value, rounding_mode) = floor or trunc of value / x.

    | Input range   | Check                     | Reason                                    |
    |---------------|---------------------------|-------------------------------------------|
    | x = 0         | ±inf                      | sign of value                             |
    | |x| > 1/SNB   | exactly 0                 | reciprocal FTZ, so floor gives 0 where    |
    |               |                           | the true result is -1                     |
    | elsewhere     | abs ≤ max(1, 2 ULP)       | a 1 ULP quotient error can cross an       |
    |               | vs float64                | integer (off by one at ~150 elements)     |

    The reference is float64: torch's float32 floor division is itself off
    by up to 2 for quotients near 2^23.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    quotient = value / input_tensor.double()
    golden = (torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)).float()

    result = ttnn.to_torch(ttnn.rdiv(tt_in, value, rounding_mode=rounding_mode))

    _assert_rdiv_edges(input_tensor, result, value)
    keep = (input_tensor != 0) & (input_tensor.abs() <= RDIV_RECIP_FTZ)
    golden, result = golden[keep], result[keep]
    bound = torch.clamp(2 * ulp(golden), min=1.0)
    _assert_abs_within(golden, result, bound, f"rdiv({value}, {rounding_mode})")


# ─────────────────────────────────────────────────────────────────────────────
# tanh, sigmoid_accurate, sigmoid
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn",
    [(ttnn.tanh, torch.tanh), (ttnn.sigmoid_accurate, torch.sigmoid)],
    ids=["tanh", "sigmoid_accurate"],
)
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_tanh_sigmoid_accurate(device, ttnn_op, golden_fn, fast_and_approximate_mode):
    """tanh and sigmoid_accurate (deprecated), both modes.

    | Op               | Mode     | Input range  | Check       | Reason                             |
    |------------------|----------|--------------|-------------|------------------------------------|
    | both             | both     | golden < SNB | exactly 0   | FTZ                                |
    |                  | accurate | elsewhere    | ULP ≤ 2     | measured max 2                     |
    | tanh             | fast     | elsewhere    | abs ≤ 0.019 | max 0.0188 at |x| = 0.5            |
    | sigmoid_accurate | fast     | elsewhere    | rtol 0.031  | fast exp; max 0.0306 at x = -4.19  |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden = golden_fn(input_tensor)

    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=fast_and_approximate_mode))

    ftz = golden.abs() < SMALLEST_NORMAL_BF16
    assert_ftz_band(result, ftz, "golden < SNB")
    golden, result = golden[~ftz], result[~ftz]

    if not fast_and_approximate_mode:
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)
    elif ttnn_op is ttnn.tanh:
        assert_allclose(expected_result=golden, actual_result=result, rtol=0, atol=0.019)
    else:
        assert_allclose(expected_result=golden, actual_result=result, rtol=0.031, atol=0)


@pytest.mark.parametrize(
    "vector_mode, mode",
    [
        (4, ttnn.SigmoidMode.Accurate),
        (4, ttnn.SigmoidMode.AccurateWithFastExp),
        (4, ttnn.SigmoidMode.FastApproximate),
        (2, ttnn.SigmoidMode.Accurate),
        (2, ttnn.SigmoidMode.FastApproximate),
    ],
    ids=["vm4-accurate", "vm4-accurate_fast_exp", "vm4-fast_approximate", "vm2-accurate", "vm2-fast_approximate"],
)
def test_sigmoid(device, vector_mode, mode):
    """sigmoid(x, vector_mode, mode).

    | Mode                | Input range  | Check       | Reason                                   |
    |---------------------|--------------|-------------|------------------------------------------|
    | all                 | golden < SNB | exactly 0   | FTZ                                      |
    | Accurate            | elsewhere    | ULP ≤ 2     | measured max 2                           |
    | AccurateWithFastExp | elsewhere    | rtol 0.031  | fast exp; max 0.0306 at x = -4.19        |
    | FastApproximate     | elsewhere    | abs ≤ 0.12  | saturates to 0 and 1 from |x| = 2        |
    |                     |              |             | (abs error 0.1192 there)                 |

    vector_mode 4 (RC) processes whole tiles. vector_mode 2 (C) processes
    only columns 0-15 of each tile, so its input is reshaped to 16 columns.
    """
    input_tensor = generate_float32_bits()
    if vector_mode == 2:
        input_tensor = input_tensor.reshape(4096, 16)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden = torch.sigmoid(input_tensor)

    result = ttnn.to_torch(ttnn.sigmoid(tt_in, vector_mode=vector_mode, mode=mode))

    ftz = golden.abs() < SMALLEST_NORMAL_BF16
    assert_ftz_band(result, ftz, "golden < SNB")
    golden, result = golden[~ftz], result[~ftz]

    if mode == ttnn.SigmoidMode.Accurate:
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)
    elif mode == ttnn.SigmoidMode.AccurateWithFastExp:
        assert_allclose(expected_result=golden, actual_result=result, rtol=0.031, atol=0)
    else:
        assert_allclose(expected_result=golden, actual_result=result, rtol=0, atol=0.12)


# ─────────────────────────────────────────────────────────────────────────────
# xielu
# ─────────────────────────────────────────────────────────────────────────────


def _xielu_reference(x, alpha_p, alpha_n):
    """xielu in float64 with beta = 0.5 and eps = -1e-6, rounded to float32."""
    x64 = x.double()
    pos = alpha_p * x64 * x64 + 0.5 * x64
    neg = alpha_n * torch.expm1(torch.clamp(x64, max=-1e-6)) - alpha_n * x64 + 0.5 * x64
    return torch.where(x64 > 0, pos, neg).float()


@pytest.mark.parametrize(
    "alpha_p, alpha_n", [(0.8, 0.8), (1.0, 1.0), (0.5, 1.5), (2.0, 0.1)], ids=["default", "one", "mix1", "mix2"]
)
def test_xielu(device, alpha_p, alpha_n):
    """xielu(x) = alpha_p * x^2 + 0.5 * x for x > 0, else
    alpha_n * expm1(min(x, -1e-6)) - alpha_n * x + 0.5 * x.

    | Input range       | Check                | Reason                                    |
    |-------------------|----------------------|-------------------------------------------|
    | x > 0, ref < SNB  | exactly 0            | FTZ                                       |
    | x > 0             | ULP ≤ 2 vs float64   | multiply-add; alpha_p * x^2 overflows to  |
    |                   |                      | +inf on both sides near the top           |
    | x <= 0            | rtol 1e-5, atol 1e-6 | the terms cancel where the output crosses |
    |                   | vs float64           | 0 (2816 ULP at x = -1.59); up to 9.8e-6   |
    |                   |                      | relative for |x| >= 1e4                   |

    The reference is float64: the float32 golden overflows -alpha_n * x for
    x < -2.27e38 where the true result is finite, and the device's is.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden = _xielu_reference(input_tensor, alpha_p, alpha_n)

    result = ttnn.to_torch(ttnn.xielu(tt_in, alpha_p=alpha_p, alpha_n=alpha_n))

    positive = input_tensor > 0
    ftz = positive & (golden.abs() < SMALLEST_NORMAL_BF16)
    assert_ftz_band(result, ftz, "x > 0, reference < SNB")
    keep = positive & ~ftz
    assert_with_ulp(expected_result=golden[keep], actual_result=result[keep], ulp_threshold=2, allow_nonfinite=True)

    non_positive = input_tensor <= 0
    assert_allclose(expected_result=golden[non_positive], actual_result=result[non_positive], rtol=1e-5, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# softplus
# ─────────────────────────────────────────────────────────────────────────────

# exp(t) underflows to 0 for t <= -2^23. The fp32 tail used to round t / ln2 to
# int32 without a clamp and return garbage there (#55798); that range is checked
# on its own in test_softplus_tail_breakdown.
SOFTPLUS_TAIL_BREAKDOWN = -(2.0**23)


@pytest.mark.parametrize(
    "beta, threshold_val",
    [(1.0, 20.0), (0.5, 20.0), (2.0, 10.0), (1.0, 5.0), (-1.0, 20.0), (1.0, -20.0)],
    ids=["default", "beta_half", "beta2", "low_threshold", "beta_neg", "threshold_neg"],
)
def test_softplus(device, beta, threshold_val):
    """softplus(x) = log(1 + exp(beta * x)) / beta, or x where beta * x > threshold.

    | Input range (t = beta * x) | Check      | Reason                                    |
    |----------------------------|------------|-------------------------------------------|
    | -2^23 < t <= -87.5         | exactly 0  | exp(t) < SNB is FTZ'd, even where the     |
    |                            |            | golden / beta is normal                   |
    | golden < SNB               | exactly 0  | FTZ                                       |
    | t > threshold              | exact      | returns x                                 |
    | elsewhere                  | rtol       | degree-8 polynomial for abs(t) <= 5; max  |
    |                            | 5.1e-4     | 5.07e-4 at t = -5                         |

    t <= -2^23 is test_softplus_tail_breakdown.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softplus)
    golden = golden_function(input_tensor, beta=beta, threshold=threshold_val, device=device)

    result = ttnn.to_torch(ttnn.softplus(tt_in, beta=beta, threshold=threshold_val))

    t = beta * input_tensor.double()
    in_range = t > SOFTPLUS_TAIL_BREAKDOWN
    linear = t > threshold_val
    ftz = in_range & ~linear & ((t <= -87.5) | (golden.abs() < SMALLEST_NORMAL_BF16))
    curve = in_range & ~ftz & ~linear

    assert_ftz_band(result, ftz, "-2^23 < beta * x <= -87.5 or golden < SNB")
    assert_equal(golden[linear], result[linear])
    assert_allclose(expected_result=golden[curve], actual_result=result[curve], rtol=5.1e-4, atol=0)


def test_softplus_tail_breakdown(device):
    """softplus(x) for beta * x <= -2^23, default beta and threshold.

    | Input range  | Check     | Reason                                            |
    |--------------|-----------|---------------------------------------------------|
    | x <= -2^23   | exactly 0 | exp(x) underflows; the unclamped int32 round of   |
    |              |           | x / ln2 used to return non-zero garbage (#55798)  |
    """
    input_tensor = generate_float32_bits()
    input_tensor = input_tensor[input_tensor <= SOFTPLUS_TAIL_BREAKDOWN].reshape(1, -1)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    result = ttnn.to_torch(ttnn.softplus(tt_in, beta=1.0, threshold=20.0))

    assert_equal(torch.zeros_like(result), result)


def test_softplus_beta_zero(device):
    """softplus with beta = 0 is log(2) / 0 = +inf everywhere.

    | Input range | Check | Reason                    |
    |-------------|-------|---------------------------|
    | all         | +inf  | division by beta = 0      |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softplus)
    golden = golden_function(input_tensor, beta=0.0, threshold=20.0, device=device)

    result = ttnn.to_torch(ttnn.softplus(tt_in, beta=0.0, threshold=20.0))

    assert torch.isposinf(golden).all(), "golden must be all +inf for beta = 0"
    assert torch.isposinf(result).all(), "device must return +inf everywhere for beta = 0"


# ─────────────────────────────────────────────────────────────────────────────
# logit
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("eps", [None, 1e-6, 0.1, 0.5, 0.9], ids=["none", "tiny", "small", "half", "gt_half"])
def test_logit(device, eps):
    """logit(x, eps) = log(x / (1 - x)), with x clamped to [eps, 1 - eps] when eps is given.

    | Input range           | Check       | Reason                                     |
    |-----------------------|-------------|--------------------------------------------|
    | eps = None, x ∉ (0,1) | not checked | outside the domain                         |
    | |golden| < 1          | abs ≤ 2^-23 | log cancels near x = 0.5 (15 ULP at        |
    |                       |             | x = 0.5156)                                |
    | |golden| >= 1         | ULP ≤ 1     |                                            |

    Inputs are the bfloat16 values in [-2, 2]. eps > 0.5 takes the golden's
    manual-clamp branch.
    """
    input_tensor = generate_float32_bits_in_range(-2.0, 2.0)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    kwargs = {} if eps is None else {"eps": eps}
    golden = golden_function(input_tensor, device=device, **kwargs)

    result = ttnn.to_torch(ttnn.logit(tt_in, **kwargs))

    if eps is None:
        in_domain = (input_tensor > 0) & (input_tensor < 1)
        golden, result = golden[in_domain], result[in_domain]

    small = golden.abs() < 1
    assert_allclose(expected_result=golden[small], actual_result=result[small], rtol=0, atol=2**-23)
    assert_with_ulp(expected_result=golden[~small], actual_result=result[~small], ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# polygamma
# ─────────────────────────────────────────────────────────────────────────────


# k -> (Wormhole, Blackhole) ULP threshold on [1, 10]
POLYGAMMA_WINDOW_ULP = {1: (2, 3), 2: (4, 6), 3: (5, 7), 4: (5, 7), 5: (7, 7), 10: (11, 20)}


@pytest.mark.parametrize(
    "k, rtol",
    [(1, 1e-6), (2, 1e-6), (3, 1e-6), (4, 1e-6), (5, 1e-5), (10, 6e-3)],
)
def test_polygamma(device, k, rtol):
    """polygamma(k, x) for x > 0.

    | k   | Input range        | Check       | Reason                                   |
    |-----|--------------------|-------------|------------------------------------------|
    | all | overflow           | +inf        | ~k!/x^(k+1) as x -> 0 (25% of the sweep  |
    |     |                    |             | at k = 1, 46% at k = 10)                 |
    |     | reference < SNB    | exactly 0   | FTZ                                      |
    | 1   | x = 2^126          | 0 or golden | trigamma ≈ 1/x is exactly SNB: Blackhole |
    |     |                    |             | flushes it, Wormhole keeps it            |
    | 1-4 | elsewhere          | rtol 1e-6   | max 8.5e-7 (k = 4, BH)                   |
    | 5   | elsewhere          | rtol 1e-5   | 6.3e-6 for x >= 1e6                      |
    | 10  | elsewhere          | rtol 6e-3   | 5.4e-3 for x in [1e3, 1e6)               |
    | all | [1, 10]            | WH ULP ≤ 2, | k = 1, 2, 3, 4, 5, 10                    |
    |     |                    | 4, 5, 5, 7, |                                          |
    |     |                    | 11          |                                          |
    |     |                    | BH ULP ≤ 3, |                                          |
    |     |                    | 6, 7, 7, 7, |                                          |
    |     |                    | 20          |                                          |

    The reference is float64: torch's float32 polygamma is 18% off at k = 10,
    x = 22272. The [1, 10] window keeps a tight bound where a correlation over
    the whole range, dominated by the huge values near 0, would not notice.
    """
    wh_ulp, bh_ulp = POLYGAMMA_WINDOW_ULP[k]
    window_ulp = bh_ulp if is_blackhole() else wh_ulp
    input_tensor = generate_float32_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden = torch.special.polygamma(k, input_tensor.double()).float()

    result = ttnn.to_torch(ttnn.polygamma(tt_in, k))

    _assert_inf_matches(golden, result, f"polygamma(k={k})")
    ftz = golden.abs() < SMALLEST_NORMAL_BF16
    assert_ftz_band(result, ftz, "reference < SNB")
    recip_snb = (input_tensor == RDIV_RECIP_FTZ) & ~ftz
    _assert_zero_or_golden(golden, result, recip_snb, f"polygamma(k={k})")
    keep = torch.isfinite(golden) & ~ftz & ~recip_snb
    assert_allclose(expected_result=golden[keep], actual_result=result[keep], rtol=rtol, atol=0)

    window = (input_tensor >= 1.0) & (input_tensor <= 10.0)
    assert_with_ulp(expected_result=golden[window], actual_result=result[window], ulp_threshold=window_ulp)


# ─────────────────────────────────────────────────────────────────────────────
# unary_chain
# ─────────────────────────────────────────────────────────────────────────────


def test_unary_chain_relu_exp_square(device):
    """unary_chain([RELU, EXP(accurate), POWER(2)]) = exp(relu(x))^2.

    | Input range | Check   | Reason                                         |
    |-------------|---------|------------------------------------------------|
    | x >= 44.5   | +inf    | exp(x)^2 overflows (24% of the sweep)          |
    | elsewhere   | ULP ≤ 3 | exp and square each round (3 ULP at x = 0.299) |
    """
    input_tensor = generate_float32_bits()
    tt_in = to_float32_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
    ]
    result = ttnn.to_torch(ttnn.unary_chain(tt_in, ops_chain))

    golden = torch.relu(input_tensor.double()).exp().pow(2).float()
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=3, allow_nonfinite=True)


def test_unary_chain_square_sqrt(device):
    """unary_chain([SQUARE, SQRT]) = |x|.

    | Input range           | Check     | Reason                                    |
    |-----------------------|-----------|-------------------------------------------|
    | |x| < 2^-63           | exactly 0 | x^2 is subnormal and FTZ'd                |
    | |x| >= 2^64           | +inf      | x^2 overflows                             |
    | elsewhere             | exact     | x^2 of a bfloat16 value is exact in fp32, |
    |                       |           | and so is its sqrt                        |
    """
    input_tensor = generate_float32_bits()
    tt_in = to_float32_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
    ]
    result = ttnn.to_torch(ttnn.unary_chain(tt_in, ops_chain))

    absx = input_tensor.abs()
    underflow = absx < 2.0**-63
    overflow = absx >= 2.0**64
    assert_ftz_band(result, underflow, "|x| < 2^-63")
    assert torch.isposinf(result[overflow]).all(), "expected +inf where x^2 overflows"
    middle = ~underflow & ~overflow
    assert_equal(absx[middle], result[middle])


def test_unary_chain_relu_typecast(device):
    """unary_chain([RELU, TYPECAST(float32 -> bfloat16)]), a chain that changes dtype.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | inputs are bfloat16 values, so narrowing relu(x)|
    |             |       | is exact                                        |
    """
    input_tensor = generate_float32_bits()
    tt_in = to_float32_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value),
    ]
    result = ttnn.to_torch(ttnn.unary_chain(tt_in, ops_chain))

    assert result.dtype == torch.bfloat16, f"expected bfloat16 output, got {result.dtype}"
    assert_equal(torch.relu(input_tensor).to(torch.bfloat16), result)
