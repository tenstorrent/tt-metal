# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_allclose,
    assert_equal,
    assert_with_ulp,
    flush_subnormal_values_to_zero,
)
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    binary_grid_values,
    pairwise_from_values,
    run_binary,
    to_float32_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 4: composite binary math

| #  | Op                            | Computes               | Check (float32)                                |
|----|-------------------------------|------------------------|------------------------------------------------|
| 1  | ttnn.ldexp / ldexp_           | a * 2**b               | ULP ≤ 1 (grid), ULP ≤ 2 (full mantissa)        |
| 2  | ttnn.logaddexp / logaddexp_   | log(exp(a) + exp(b))   | ULP ≤ 1 for abs(golden) ≥ 1, else abs ≤ 2^-23  |
| 3  | ttnn.logaddexp2 / logaddexp2_ | log2(2**a + 2**b)      | ULP ≤ 1 for abs(golden) ≥ 1, else abs ≤ 2^-22  |
| 4  | ttnn.squared_difference / _   | (a - b)**2             | exact vs torch float32                         |
| 5  | ttnn.xlogy                    | a * log(b)             | abs ≤ abs(a) * 5e-4 (bfloat16-grade log)       |
| 6  | ttnn.hypot                    | sqrt(a**2 + b**2)      | ULP ≤ 1 where neither square under/overflows   |
| 7  | ttnn.bias_gelu / bias_gelu_   | gelu(a + b)            | x = a + b: ULP ≤ 3 (x ≥ 0), ULP ≤ 5            |
|    |                               |                        | (-1 < x < 0), abs ≤ 1e-6 (x ≤ -1); and bitwise |
|    |                               |                        | gelu(add(a, b))                                |
| 8  | ttnn.prelu                    | a if a >= 0 else a * w | exact                                          |

Working range of the accuracy sweeps. "Full grid" is the stratified binary
grid at float32: finite bfloat16 values and +0, no -0, inf or NaN. In-place
ops share the kernel, the range and the thresholds.

| Op                 | a                    | b or weight            | Outside the range, pinned by           |
|--------------------|----------------------|------------------------|----------------------------------------|
| ldexp              | full grid            | [-126, 127]            | test_ldexp_exponent_underflow/overflow |
| logaddexp(2)       | full grid            | full grid              | —                                      |
| squared_difference | full grid            | full grid              | —                                      |
| xlogy              | [-1e30, 1e30]        | positive normals       | test_xlogy_nonpositive                 |
| hypot              | ±[2^-62, 2^62], or 0 | same as a              | test_hypot_outside_square_range        |
| bias_gelu          | [-100, 100]          | [-100, 100]            | —                                      |
| prelu              | full grid            | full grid, per channel | —                                      |

The grid only carries bfloat16 mantissas, so the chains with a
transcendental step are also swept on random full-mantissa inputs
(test_full_mantissa).

Goldens are float64 rounded to float32 unless noted. The device flushes
subnormal results to zero, so those elements are expected to be exactly 0
wherever the float64 result is below 2^-126.

Most ops lower to a pre-op / binary / post-op chain (OpConfig in
binary_ng_utils.cpp) that runs on the SFPU with fp32 dest accumulation, so
the intermediate stays float32 and the bfloat16 file's extra ULP for a
rounded intermediate does not apply. The chains still inherit their pre-op's
domain: EXP2 overflows for b ≥ 128, SQUARE flushes or overflows outside
2^±63.

Tensor-scalar overloads — every op except hypot, which has none — are covered
out-of-place by test_tensor_scalar at the same thresholds.
"""

FLOAT32_TINY = float(torch.finfo(torch.float32).tiny)

LDEXP_MIN_EXPONENT = -126.0
LDEXP_MAX_EXPONENT = 127.0

# The xlogy kernel evaluates log(b) with the bfloat16-grade log polynomial for
# every dtype, not the float32 log ttnn.log uses. Its error is absolute and
# nearly constant over b: 4.4e-4 at log(1), at most 4.51e-4 measured. It
# enters the product scaled by abs(a).
XLOGY_LOG_ATOL = 5e-4

HYPOT_MIN_OPERAND = 2.0**-62
HYPOT_MAX_OPERAND = 2.0**62

# Accurate gelu on x = a + b, banded as in the unary category 3 float32 test.
# That test sweeps bfloat16 inputs only; full-mantissa sums cost one more ULP
# and roughly double the absolute error.
GELU_NONNEGATIVE_ULP = 3
GELU_SMALL_NEGATIVE_ULP = 5
GELU_LARGE_NEGATIVE_ATOL = 1e-6


def _golden_f64(fn, input_a, input_b):
    """float64 golden for float32 inputs: (rounded to float32, float64)."""
    reference = fn(input_a.double(), input_b.double())
    return reference.to(torch.float32), reference


def _ldexp_f64(a, b):
    return a * torch.exp2(b)


def _bias_gelu_golden(input_a, input_b):
    """gelu of the float32 sum, computed in float64. Returns (x, golden).

    The device adds in float32 before the GELU post-op, so the golden starts
    from the same rounded sum. The gelu itself is float64, because float32's
    1 + erf(x / sqrt(2)) cancels for x around -5.
    """
    x = input_a + input_b
    return x, F.gelu(x.double()).to(torch.float32)


def _float32_result(result, desc):
    """comp_ulp sizes the ULP from the result's dtype, so a bfloat16 output
    would be measured in bfloat16 ULPs and pass float32 thresholds."""
    assert result.dtype == torch.float32, f"{desc}: expected float32 output, got {result.dtype}"
    return result


def _run_f64(device, ttnn_op, fn, input_a, input_b):
    """run_binary with a float64 golden in place of the torch float32 one."""
    _, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    golden, reference = _golden_f64(fn, input_a, input_b)
    return golden, reference, _float32_result(result, ttnn_op.__name__)


def _assert_ldexp(golden, reference, result, input_a, input_b):
    """ULP ≤ 1, with the flush band keyed on the float64 product.

    Below 2^-126 the device flushes, but the product can still land on
    ±2^-126 in two ways. So the band accepts 0, or exactly ±2^-126 where
    the golden or the float32 chain rounds there; any other nonzero value,
    such as an unflushed 2^-127 from ldexp(2^-126, -1), fails:

    | a       | b         | float32 2**b         | product                      | device               |
    |---------|-----------|----------------------|------------------------------|----------------------|
    | 2^-126  | -1.1e-16  | 1.0                  | golden itself rounds to      | 2^-126 or 0          |
    |         |           |                      | 2^-126                       |                      |
    | 2^-126  | ~ -2^-23  | 1 - 2^-24            | 2^-126 - 2^-150, the exact   | WH: 0 (flushed)      |
    |         |           | (0x3F7FFFFF)         | halfway point; RNE gives     | BH: 2^-126 (rounded) |
    |         |           |                      | 2^-126, golden stays 0x7FFFFF|                      |

    Both architectures return the same float32 2**b; they differ only in
    whether the multiply flushes the halfway product before or after rounding.
    """
    band = reference.abs() < FLOAT32_TINY
    chained = input_a * torch.exp2(input_b.double()).to(torch.float32)
    band_result = result[band]
    rounded_up = (band_result.abs() == FLOAT32_TINY) & ((band_result == golden[band]) | (band_result == chained[band]))
    band_ok = (band_result == 0) | rounded_up
    assert band_ok.all(), (
        f"ldexp: {int((~band_ok).sum())} results below 2^-126 are neither 0 nor a ±2^-126 that the golden "
        "or a * float32(2**b) rounds to"
    )
    assert_with_ulp(expected_result=golden[~band], actual_result=result[~band], ulp_threshold=1, allow_nonfinite=True)


def _assert_logaddexp_bands(golden, result, ulp_threshold, small_atol):
    large = golden.abs() >= 1.0
    if large.any():
        assert_with_ulp(expected_result=golden[large], actual_result=result[large], ulp_threshold=ulp_threshold)
    if (~large).any():
        assert_allclose(expected_result=golden[~large], actual_result=result[~large], rtol=0, atol=small_atol)


def _assert_xlogy_within_log_error(input_a, reference, result, desc):
    """abs(device - reference) ≤ abs(a) * XLOGY_LOG_ATOL + 2^-126.

    The 2^-126 term is the product's own flush to zero.
    """
    error = (reference - result.double()).abs()
    tolerance = XLOGY_LOG_ATOL * input_a.double().abs() + FLOAT32_TINY
    failed = error > tolerance
    if failed.any():
        worst = int(torch.argmax(error - tolerance))
        raise AssertionError(
            f"{desc}: {int(failed.sum())} elements exceed abs(a) * {XLOGY_LOG_ATOL}; worst is "
            f"a={input_a.reshape(-1)[worst].item()}, golden={reference.reshape(-1)[worst].item()}, "
            f"actual={result.reshape(-1)[worst].item()}"
        )


def _assert_bias_gelu(x, golden, result):
    """Accurate gelu contract on x = fl32(a + b).

    | Input x         | Check      | Measured: grid / full mantissa                  |
    |-----------------|------------|-------------------------------------------------|
    | abs(x) < 2^-125 | exactly 0  | x/2 is subnormal and flushed; a subnormal sum   |
    |                 |            | is already flushed by the add                   |
    | x ≥ 0           | ULP ≤ 3    | 2 / 3                                           |
    | -1 < x < 0      | ULP ≤ 5    | 4 / 5                                           |
    | x ≤ -1          | abs ≤ 1e-6 | 4.66e-7 / 8.66e-7; the device returns 0 once    |
    |                 |            | abs(gelu) < ~5.2e-8 (x ≈ -5.6 down to -8.4)     |
    """
    _float32_result(result, "bias_gelu")
    flush = x.abs() < 2.0**-125
    assert_equal(torch.zeros_like(result[flush]), result[flush])
    non_negative = ~flush & (x >= 0)
    small_negative = ~flush & (x < 0) & (x > -1)
    large_negative = ~flush & (x <= -1)
    if non_negative.any():
        assert_with_ulp(
            expected_result=golden[non_negative],
            actual_result=result[non_negative],
            ulp_threshold=GELU_NONNEGATIVE_ULP,
        )
    if small_negative.any():
        assert_with_ulp(
            expected_result=golden[small_negative],
            actual_result=result[small_negative],
            ulp_threshold=GELU_SMALL_NEGATIVE_ULP,
        )
    if large_negative.any():
        assert_allclose(
            expected_result=golden[large_negative],
            actual_result=result[large_negative],
            rtol=0,
            atol=GELU_LARGE_NEGATIVE_ATOL,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ldexp — a * 2**b, lowered to EXP2(b) followed by a multiply
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.ldexp, ttnn.ldexp_])
def test_ldexp(device, ttnn_op):
    """a over the full grid, b over the exponents EXP2 can represent.

    | Case                     | float32 behavior                              | Check        |
    |--------------------------|-----------------------------------------------|--------------|
    | b in [-126, 127]         | EXP2 and the multiply stay float32            | ULP ≤ 1      |
    | float64 result < 2^-126  | flushed to 0, or 2^-126 where float32         | 0 or golden  |
    |                          | rounding lifts it there                       |              |
    | overflow                 | ±inf on both sides                            | sign matches |

    Unlike bfloat16 there is no overflow-to-zero or overflow-to-inf mask:
    both saturations follow IEEE.
    """
    # The grid's b values stop at ±68, so the EXP2 endpoints are appended.
    exponents = torch.cat(
        [
            binary_grid_values(LDEXP_MIN_EXPONENT, LDEXP_MAX_EXPONENT, dtype=torch.float32),
            torch.tensor([LDEXP_MIN_EXPONENT, LDEXP_MAX_EXPONENT]),
        ]
    )
    input_a, input_b = pairwise_from_values(binary_grid_values(dtype=torch.float32), exponents)
    golden, reference, result = _run_f64(device, ttnn_op, _ldexp_f64, input_a, input_b)

    assert ((reference != 0) & (reference.abs() < FLOAT32_TINY)).any(), "expected the flush band in this sweep"
    assert torch.isinf(golden).any(), "expected overflowing products in this sweep"
    _assert_ldexp(golden, reference, result, input_a, input_b)


def test_ldexp_exponent_underflow(device):
    """b < -126: EXP2 underflows, the multiply sees 0, and ldexp returns +0
    for every a — including the 4784 grid pairs whose true result is a normal
    value as large as 1.41 (a = ±max bf16 with b = -127.5)."""
    input_a, input_b = pairwise_from_values(
        binary_grid_values(dtype=torch.float32),
        binary_grid_values(high=LDEXP_MIN_EXPONENT - 0.5, include_zero=False, dtype=torch.float32),
    )
    result = ttnn.to_torch(ttnn.ldexp(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))

    golden, _ = _golden_f64(_ldexp_f64, input_a, input_b)
    assert (golden != 0).any(), "expected some representable results in the EXP2 underflow region"
    assert_equal(torch.zeros_like(result), result)


def test_ldexp_exponent_overflow(device):
    """b > 127, the half of EXP2's range test_ldexp does not cover.

    | b       | Device                                      | Check                |
    |---------|---------------------------------------------|----------------------|
    | 127.5   | real 2^b                                    | ULP ≤ 1 vs float64   |
    | ≥ 128   | EXP2 is +inf: a > 0 → +inf, a < 0 → -inf,   | exact class          |
    |         | a == 0 → NaN (0 * inf)                      |                      |

    Unlike bfloat16, whose EXP2 clamps at 2^128 and returns ldexp(a, 128),
    the float32 EXP2 overflows to inf, so even 2^-126 * 2^128 comes back as
    +inf rather than 4.
    """
    input_a, input_b = pairwise_from_values(
        binary_grid_values(dtype=torch.float32),
        binary_grid_values(low=LDEXP_MAX_EXPONENT + 0.5, include_zero=False, dtype=torch.float32),
    )
    result = ttnn.to_torch(ttnn.ldexp(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))
    result = _float32_result(result, "ldexp exponent overflow")

    saturated = input_b >= 128
    assert saturated.any() and (~saturated).any(), "expected both the saturated and the 127.5 scales"
    assert (saturated & (input_a == 0)).any()

    golden, _ = _golden_f64(_ldexp_f64, input_a, input_b)
    expected = torch.where(saturated, input_a * float("inf"), golden)
    assert_with_ulp(
        expected_result=expected[~saturated], actual_result=result[~saturated], ulp_threshold=1, allow_nonfinite=True
    )
    assert torch.isnan(result[saturated & (input_a == 0)]).all(), "ldexp(0, b >= 128) is 0 * inf"
    nonzero = saturated & (input_a != 0)
    assert_equal(expected[nonzero], result[nonzero])


# ─────────────────────────────────────────────────────────────────────────────
# logaddexp / logaddexp2 — EXP/EXP2 on both operands, add, then LOG/LOG2
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, small_atol",
    [
        (ttnn.logaddexp, torch.logaddexp, 2.0**-23),
        (ttnn.logaddexp_, torch.logaddexp, 2.0**-23),
        (ttnn.logaddexp2, torch.logaddexp2, 2.0**-22),
        (ttnn.logaddexp2_, torch.logaddexp2, 2.0**-22),
    ],
    ids=["logaddexp", "logaddexp_", "logaddexp2", "logaddexp2_"],
)
def test_logaddexp_ops(device, ttnn_op, golden_fn, small_atol):
    """Full grid; unlike bfloat16 no range restriction is needed.

    | Golden          | Check                  | Measured                                   |
    |-----------------|------------------------|--------------------------------------------|
    | abs ≥ 1         | ULP ≤ 1                | 1                                          |
    | abs < 1         | abs err ≤ small_atol   | logaddexp 7.6e-8, logaddexp2 1.12e-7       |

    The error near the zero crossing is absolute: the LOG post-op's input
    carries the add's relative rounding, which d(log s) = ds/s turns into a
    fixed absolute error.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values(dtype=torch.float32))
    golden, _, result = _run_f64(device, ttnn_op, golden_fn, input_a, input_b)

    large = golden.abs() >= 1.0
    assert large.any() and (~large).any(), "expected both the ULP and the absolute band to be non-empty"
    _assert_logaddexp_bands(golden, result, ulp_threshold=1, small_atol=small_atol)


# ─────────────────────────────────────────────────────────────────────────────
# squared_difference — subtract, then a SQUARE post-op
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.squared_difference, ttnn.squared_difference_])
def test_squared_difference(device, ttnn_op):
    """Full grid against torch's float32 (a - b)**2.

    | Case             | float32 behavior                                  | Check           |
    |------------------|---------------------------------------------------|-----------------|
    | finite           | float32 subtract then square, both RNE            | exact           |
    | subnormal result | flushed to 0 on device                            | flush both      |
    | overflow         | +inf on both sides, including the 16 pairs whose  | allow_nonfinite |
    |                  | float64 result rounds to float32 max              |                 |

    The golden is torch float32 rather than float64, because the device
    rounds a - b to float32 before squaring it; against float64 the sweep is
    within 2 ULP.

    A square is never negative, and on this grid the device returns +0
    for every zero result.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values(dtype=torch.float32))
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)

    golden = flush_subnormal_values_to_zero(golden)
    result = flush_subnormal_values_to_zero(_float32_result(result, ttnn_op.__name__))
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# xlogy — a * log(b), a single SFPU instruction with a bfloat16-grade log
# ─────────────────────────────────────────────────────────────────────────────


def test_xlogy(device):
    """b over the positive normals, a capped at 1e30 so a * log(b) (abs(log b)
    ≤ 87.4) stays finite. b ≤ 0 is test_xlogy_nonpositive.

    | Case              | float32 behavior                                   | Check                    |
    |-------------------|----------------------------------------------------|--------------------------|
    | all b > 0         | log(b) carries ~4.5e-4 absolute error at every b   | abs ≤ abs(a) * 5e-4      |
    | b = 1             | log(1) = 4.4e-4: xlogy(6.73e29, 1) = 2.97e26       | same bound               |
    | b far from 1      | up to 2.5e-4 relative (3789 ULP) for b ∉ [0.25, 4] | same bound               |
    | product < 2^-126  | flushed to 0                                       | + 2^-126 in the bound    |

    The kernel calls the bfloat16 log body for float32 too, so xlogy does not
    get the 1-ULP float32 log that ttnn.log has. This pins the current
    behavior; it is not a float32-grade contract.
    """
    input_a, input_b = pairwise_from_values(
        binary_grid_values(-1e30, 1e30, dtype=torch.float32),
        binary_grid_values(low=FLOAT32_TINY, dtype=torch.float32),
    )
    _, reference, result = _run_f64(device, ttnn.xlogy, torch.xlogy, input_a, input_b)

    assert torch.isfinite(result).all(), "xlogy over b > 0 and abs(a) ≤ 1e30 must stay finite"
    _assert_xlogy_within_log_error(input_a, reference, result, "xlogy")


def test_xlogy_nonpositive(device):
    """b ≤ 0, which the accuracy sweep drops. A float32 destination keeps NaN
    as NaN, so the classes are exact rather than the bfloat16 file's
    NaN-or-inf.

    | b     | a     | torch  | device | Note                                   |
    |-------|-------|--------|--------|----------------------------------------|
    | < 0   | ≠ 0   | NaN    | NaN    |                                        |
    | < 0   | 0     | 0      | NaN    | the kernel does not special-case a = 0 |
    | +0    | > 0   | -inf   | -inf   |                                        |
    | +0    | < 0   | +inf   | +inf   |                                        |
    | +0    | 0     | 0      | NaN    | 0 * log(0) = 0 * -inf                  |
    | -0    | any   | ±inf/0 | NaN    | -0 takes the b < 0 path                |
    """
    values_b = torch.cat([binary_grid_values(high=0.0, dtype=torch.float32), torch.tensor([-0.0])])
    input_a, input_b = pairwise_from_values(binary_grid_values(dtype=torch.float32), values_b)
    result = ttnn.to_torch(ttnn.xlogy(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))

    negative_b = (input_b < 0) | ((input_b == 0) & torch.signbit(input_b))
    zero_b = (input_b == 0) & ~torch.signbit(input_b)
    assert (negative_b & (input_a == 0)).any() and (input_b == 0).any()
    assert (zero_b & (input_a == 0)).any() and (zero_b & (input_a > 0)).any() and (zero_b & (input_a < 0)).any()

    assert torch.isnan(result[negative_b]).all(), "xlogy(a, b < 0 or -0) must be NaN"
    pos_a, neg_a, zero_a = zero_b & (input_a > 0), zero_b & (input_a < 0), zero_b & (input_a == 0)
    assert_equal(torch.full_like(result[pos_a], float("-inf")), result[pos_a])
    assert_equal(torch.full_like(result[neg_a], float("inf")), result[neg_a])
    assert torch.isnan(result[zero_a]).all(), "xlogy(0, +0) is 0 * -inf"


# ─────────────────────────────────────────────────────────────────────────────
# hypot — SQUARE on both operands, add, then a SQRT post-op
# ─────────────────────────────────────────────────────────────────────────────


def test_hypot(device):
    """Operands restricted to [2^-62, 2^62] plus 0, where both squares stay
    normal float32 values. Within 1 ULP of float64 hypot."""
    values = binary_grid_values(
        -HYPOT_MAX_OPERAND, HYPOT_MAX_OPERAND, min_magnitude=HYPOT_MIN_OPERAND, dtype=torch.float32
    )
    input_a, input_b = pairwise_from_values(values)
    golden, _, result = _run_f64(device, ttnn.hypot, torch.hypot, input_a, input_b)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_hypot_outside_square_range(device):
    """Over the full grid hypot is sqrt of whatever the SQUARE pre-ops left.

    | Operand      | SQUARE result         | Effect                                        |
    |--------------|-----------------------|-----------------------------------------------|
    | abs < 2^-63  | subnormal, flushed    | drops out: hypot(2^-126, b) = abs(b),         |
    |              |                       | hypot(2^-126, 2^-126) = 0                     |
    | abs ≥ 2^64   | overflows to +inf     | +inf, well before hypot itself would overflow |

    Checked to 1 ULP against sqrt(ftz(a²) + ftz(b²)) in float32, with the
    +infs matching.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values(dtype=torch.float32))
    result = ttnn.to_torch(ttnn.hypot(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))
    result = _float32_result(result, "hypot")

    squares = [flush_subnormal_values_to_zero(x * x) for x in (input_a, input_b)]
    expected = torch.sqrt(squares[0] + squares[1])

    assert (expected == 0).any() and torch.isinf(expected).any(), "expected both boundaries in this sweep"
    assert_with_ulp(expected_result=expected, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# bias_gelu — add, then a GELU post-op
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.bias_gelu, ttnn.bias_gelu_])
def test_bias_gelu(device, ttnn_op):
    """a, b over [-100, 100]. fast_and_approximate_mode defaults to False, so
    the post-op is the accurate gelu; see _assert_bias_gelu for the bands."""
    input_a, input_b = pairwise_from_values(binary_grid_values(-100.0, 100.0, dtype=torch.float32))
    _, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    x, golden = _bias_gelu_golden(input_a, input_b)

    assert (x.abs() < 2.0**-125).any(), "expected the flushed-sum band in this sweep"
    _assert_bias_gelu(x, golden, result)


def test_bias_gelu_matches_gelu_of_sum(device):
    """bias_gelu must be bitwise identical to accurate gelu over ttnn.add, the
    chain its default compiles to."""
    input_a, input_b = pairwise_from_values(binary_grid_values(-100.0, 100.0, dtype=torch.float32))

    tt_a, tt_b = to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)
    result = ttnn.to_torch(ttnn.bias_gelu(tt_a, tt_b))
    expected = ttnn.to_torch(ttnn.gelu(ttnn.add(tt_a, tt_b), fast_and_approximate_mode=False))

    assert torch.equal(expected.view(torch.int32), result.view(torch.int32))


# ─────────────────────────────────────────────────────────────────────────────
# prelu — where(a < 0, a * w, a), with one weight per channel
# ─────────────────────────────────────────────────────────────────────────────


def test_prelu(device):
    """a is the grid's outer product and the weight is one grid value per
    channel, so column j of the result is prelu(grid[i], grid[j]).

    | Case             | float32 behavior                     | Check           |
    |------------------|--------------------------------------|-----------------|
    | a ≥ 0            | identity                             | exact           |
    | a < 0            | float32 a * w, RNE                   | exact           |
    | subnormal a * w  | flushed to 0 on device               | flush both      |
    | overflow         | signed inf on both sides             | allow_nonfinite |

    The device's zero signs differ from torch's:

    | Input                 | torch               | device                        |
    |-----------------------|---------------------|-------------------------------|
    | a < 0, a * w flushed  | -0 (if sign kept)   | +0                            |
    | a = -0, w = -1        | +0                  | -0                            |
    | a = +0, w = -1        | -0                  | +0                            |
    | a = -0, w = 0.5       | -0                  | -0 with a weight tensor,      |
    |                       |                     | +0 with a scalar weight       |

    The grid carries no -0, so only the first row arises here.
    """
    weight = binary_grid_values(dtype=torch.float32)
    input_a, _ = pairwise_from_values(weight)

    golden = ttnn.get_golden_function(ttnn.prelu)(input_a, weight)
    result = ttnn.to_torch(ttnn.prelu(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(weight, device)))

    golden = flush_subnormal_values_to_zero(golden)
    result = flush_subnormal_values_to_zero(_float32_result(result, "prelu"))
    assert torch.isinf(golden).any(), "expected overflowing products in this sweep"
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# full mantissa — the grid only carries bfloat16 mantissas
# ─────────────────────────────────────────────────────────────────────────────


def _uniform(generator, low, high, shape=(256, 1024)):
    return (torch.rand(shape, generator=generator, dtype=torch.float64) * (high - low) + low).to(torch.float32)


@pytest.mark.parametrize("kind", ["ldexp", "logaddexp", "logaddexp2", "hypot", "bias_gelu"])
def test_full_mantissa(device, kind):
    """Seeded random float32 operands with all 23 mantissa bits populated.

    | Op         | a           | b           | Check                                   |
    |------------|-------------|-------------|-----------------------------------------|
    | ldexp      | [-4, 4]     | [-20, 20]   | ULP ≤ 2: EXP2 of a fractional b         |
    | logaddexp  | [-20, 20]   | [-20, 20]   | as test_logaddexp_ops                   |
    | logaddexp2 | [-20, 20]   | [-20, 20]   | as test_logaddexp_ops                   |
    | hypot      | [-1e3, 1e3] | [-1e3, 1e3] | ULP ≤ 1                                 |
    | bias_gelu  | [-6, 6]     | [-6, 6]     | as test_bias_gelu; sets its thresholds  |
    """
    generator = torch.Generator().manual_seed(4)
    low, high = {"ldexp": (-4.0, 4.0), "hypot": (-1e3, 1e3), "bias_gelu": (-6.0, 6.0)}.get(kind, (-20.0, 20.0))
    input_a = _uniform(generator, low, high)
    input_b = _uniform(generator, -20.0, 20.0) if kind == "ldexp" else _uniform(generator, low, high)

    if kind == "bias_gelu":
        _, result = run_binary(device, ttnn.bias_gelu, input_a, input_b, input_dtype=ttnn.float32)
        x, golden = _bias_gelu_golden(input_a, input_b)
        _assert_bias_gelu(x, golden, result)
        return

    ttnn_op, golden_fn = {
        "ldexp": (ttnn.ldexp, _ldexp_f64),
        "logaddexp": (ttnn.logaddexp, torch.logaddexp),
        "logaddexp2": (ttnn.logaddexp2, torch.logaddexp2),
        "hypot": (ttnn.hypot, torch.hypot),
    }[kind]
    golden, _, result = _run_f64(device, ttnn_op, golden_fn, input_a, input_b)
    if kind == "ldexp":
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)
    elif kind == "hypot":
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)
    else:
        _assert_logaddexp_bands(
            golden, result, ulp_threshold=1, small_atol=2.0**-23 if kind == "logaddexp" else 2.0**-22
        )


# ─────────────────────────────────────────────────────────────────────────────
# tensor-scalar — the other overload of the same kernels
# ─────────────────────────────────────────────────────────────────────────────


def _tile_rows(values):
    """Repeat each value across one tile width, (N, 32), so the scalar path
    crosses tiles. A 1-D tensor in tile layout is one tile row of padding."""
    return values.unsqueeze(1).expand(values.numel(), 32).contiguous()


@pytest.mark.parametrize(
    "ttnn_op, kind, scalar",
    [
        (ttnn.ldexp, "ldexp", -126.0),
        (ttnn.ldexp, "ldexp", 0.3),
        (ttnn.ldexp, "ldexp", 127.0),
        (ttnn.logaddexp, "logaddexp", -80.0),
        (ttnn.logaddexp, "logaddexp", 0.0),
        (ttnn.logaddexp, "logaddexp", 80.0),
        (ttnn.logaddexp2, "logaddexp2", -120.0),
        (ttnn.logaddexp2, "logaddexp2", 0.0),
        (ttnn.logaddexp2, "logaddexp2", 120.0),
        (ttnn.squared_difference, "squared_difference", -1.0e19),
        (ttnn.squared_difference, "squared_difference", 0.1),
        (ttnn.squared_difference, "squared_difference", 1.0 + 2.0**-20),
        (ttnn.bias_gelu, "bias_gelu", -100.0),
        (ttnn.bias_gelu, "bias_gelu", 0.1),
        (ttnn.bias_gelu, "bias_gelu", 100.0),
        (ttnn.xlogy, "xlogy", 1.0),
        (ttnn.xlogy, "xlogy", 0.1),
        (ttnn.xlogy, "xlogy", 16.0),
        (ttnn.prelu, "prelu", 0.5),
        (ttnn.prelu, "prelu", -1.0),
        (ttnn.prelu, "prelu", 0.1),
    ],
)
def test_tensor_scalar(device, ttnn_op, kind, scalar):
    """Tensor-scalar coverage, out-of-place, over the tensor-tensor sweep's a
    domain and at its thresholds. The scalar is packed at full float32
    precision, so float32-only scalars (0.1, 1 + 2^-20) are included.

    Measured: ldexp is exact; logaddexp(2) and bias_gelu stay within their
    sweep bands; squared_difference and prelu are exact vs torch float32;
    xlogy stays inside abs(a) * 5e-4.
    """
    domains = {
        "ldexp": binary_grid_values(dtype=torch.float32),
        "logaddexp": binary_grid_values(dtype=torch.float32),
        "logaddexp2": binary_grid_values(dtype=torch.float32),
        "squared_difference": binary_grid_values(dtype=torch.float32),
        "bias_gelu": binary_grid_values(-100.0, 100.0, dtype=torch.float32),
        "xlogy": binary_grid_values(-1.0e30, 1.0e30, dtype=torch.float32),
        "prelu": binary_grid_values(dtype=torch.float32),
    }
    input_a = _tile_rows(domains[kind])
    desc = f"{ttnn_op.__name__}(tensor, {scalar})"
    result = _float32_result(ttnn.to_torch(ttnn_op(to_float32_tt_tensor(input_a, device), scalar)), desc)
    scalar_tensor = torch.tensor(scalar, dtype=torch.float32)

    if kind == "ldexp":
        golden, reference = _golden_f64(_ldexp_f64, input_a, scalar_tensor)
        _assert_ldexp(golden, reference, result, input_a, scalar_tensor)
    elif kind in ("logaddexp", "logaddexp2"):
        golden_fn = torch.logaddexp if kind == "logaddexp" else torch.logaddexp2
        golden, _ = _golden_f64(golden_fn, input_a, scalar_tensor)
        _assert_logaddexp_bands(
            golden, result, ulp_threshold=1, small_atol=2.0**-23 if kind == "logaddexp" else 2.0**-22
        )
    elif kind == "squared_difference":
        golden = flush_subnormal_values_to_zero(ttnn.get_golden_function(ttnn_op)(input_a, scalar_tensor))
        assert_with_ulp(
            expected_result=golden,
            actual_result=flush_subnormal_values_to_zero(result),
            ulp_threshold=0,
            allow_nonfinite=True,
        )
    elif kind == "bias_gelu":
        x, golden = _bias_gelu_golden(input_a, scalar_tensor)
        _assert_bias_gelu(x, golden, result)
    elif kind == "xlogy":
        _, reference = _golden_f64(torch.xlogy, input_a, scalar_tensor)
        assert torch.isfinite(result).all(), f"{desc}: expected finite results"
        _assert_xlogy_within_log_error(input_a, reference, result, desc)
    else:
        golden = flush_subnormal_values_to_zero(ttnn.get_golden_function(ttnn_op)(input_a, scalar))
        assert_with_ulp(
            expected_result=golden,
            actual_result=flush_subnormal_values_to_zero(result),
            ulp_threshold=0,
            allow_nonfinite=True,
        )
