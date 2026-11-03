# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    flush_subnormal_values_to_zero,
    ulp_distance,
)
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    pairwise_inputs,
    ternary_inputs,
    to_float32_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: fused multiply-add family, plus select

| # | Op           | Computes             | Check (float32)                                          |
|---|--------------|----------------------|----------------------------------------------------------|
| 1 | ttnn.mac     | a * b + c            | exact (grid), 1.5 ULP of the larger term (full mantissa) |
| 2 | ttnn.lerp    | a + weight * (b - a) | 2.5 ULP of the larger term                               |
| 3 | ttnn.where   | predicate ? a : b    | bitwise                                                  |
| 4 | ttnn.addcmul | a + value * b * c    | exact (grid, value a power of two), else 1.5 / 3 ULP of  |
|   |              |                      | the larger term                                          |
| 5 | ttnn.addcdiv | a + value * b / c    | 3 / 4 ULP of the larger term                             |

Working range. "Full grid" is the 256-value ternary grid at float32 (finite
bfloat16 values, no -0, inf or NaN) swept as a full outer product, 256^3 =
16,777,216 triples; where uses the 2048-value binary grid pairwise instead,
since its predicate carries one bit per element. The grid only carries
bfloat16 mantissas, so every op is also swept on random full-mantissa inputs
(test_full_mantissa, test_where_full_mantissa).

| Op      | a, b                            | c, weight or predicate | value                |
|---------|---------------------------------|------------------------|----------------------|
| mac     | full grid                       | full grid              | —                    |
| lerp    | full grid                       | full grid              | —                    |
| addcmul | full grid                       | full grid              | ±1, ±0.5, ±2, 0, 0.3 |
| addcdiv | full grid                       | full grid, nonzero     | ±1, ±0.5, ±2, 0, 0.3 |
| where   | binary grid, with and without   | 0 / 1 patterns, and    | —                    |
|         | -0, ±inf, NaN                   | test_where_predicate_  |                      |
|         |                                 | special_values         |                      |

None of the four arithmetic kernels is a fused single-rounding operation in
float32. No step carries its intermediate into the next exactly: mac and
addcmul do not carry the full product into the add, lerp rounds b - a and
then the scaled difference, addcdiv rounds value * b, recip(c) and their
product before rounding. So when the two terms of the final add cancel,
the error measured in ULP of the output is unbounded, and a ULP-of-output budget
says nothing. What does hold is a bound in ULP of the larger of the two terms
of the final add, max(|term1|, |term2|) below, which is the natural error
model for an unfused evaluation: a correctly rounded fused op would sit at
1.0, and each extra rounded step adds about one. The constants are the
measured worst case over 16M random full-mantissa triples per op and value,
rounded up.

On the grid every product of two bfloat16 mantissas is exact in float32, so
mac, and addcmul with a power-of-two value, have no intermediate rounding to
show and are held to the correctly rounded result instead.

Goldens are float64 rounded to float32. The device flushes subnormals to
zero, so the golden is flushed too. Each grid test excuses its intermediate
under- and overflow in a table, and the names in its Stage column are the
ones apply_exceptions puts in a failure message. Every flush stage asserts
the value the device is documented to return over its range, so none of them
is a waiver.
"""

FLOAT32_MAX = float(torch.finfo(torch.float32).max)
MIN_NORMAL_F32 = 2.0**-126

# addcdiv divides by multiplying with recip(c), and for |c| >= 2^126 that
# reciprocal lands at or below the smallest normal and can flush to zero.
RECIPROCAL_FLUSH_DIVISOR = 2.0**126

# Bounds in ULP of the larger term of the final add, measured worst case over
# 4 x 4,194,304 random full-mantissa triples, exponents in [-20, 20] and
# [-60, 60], rounded up. "rounded value" is a value whose product with b is
# not exact in float32 (0.3 here), which adds one rounded step.
MAC_TERM_ULP = 1.5  # measured 1.24
LERP_TERM_ULP = 2.5  # measured 2.10
ADDCMUL_TERM_ULP = 1.5  # measured 1.23
ADDCMUL_ROUNDED_VALUE_TERM_ULP = 3.0  # measured 2.76
ADDCDIV_TERM_ULP = 3.0  # measured 2.95 (Blackhole), 2.42 (Wormhole)
ADDCDIV_ROUNDED_VALUE_TERM_ULP = 4.0  # measured 3.80 (Blackhole), 3.19 (Wormhole)

ADDC_VALUES = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.3]


def _is_exact_scale(value):
    """value * b is exact in float32 for every normal b: a power of two (or 0)."""
    return value == 0.0 or torch.frexp(torch.tensor(value, dtype=torch.float64)).mantissa.abs().item() == 0.5


def _ulp_f32(magnitude):
    """Spacing of float32 at ``magnitude`` (float64 in, float64 out)."""
    x = magnitude.abs().clamp(max=FLOAT32_MAX).to(torch.float32)
    return (torch.nextafter(x, torch.full_like(x, float("inf"))) - x).to(torch.float64)


def _golden(exact):
    return flush_subnormal_values_to_zero(exact.to(torch.float32))


def _run(device, ttnn_op, input_a, input_b, input_c, **op_kwargs):
    tensors = [to_float32_tt_tensor(t, device) for t in (input_a, input_b, input_c)]
    result = ttnn.to_torch(ttnn_op(*tensors, **op_kwargs))
    # comp_ulp sizes the ULP from the result's dtype, so a bfloat16 output
    # would be measured in bfloat16 ULPs and pass float32 thresholds.
    assert result.dtype == torch.float32, f"expected float32 output, got {result.dtype}"
    return flush_subnormal_values_to_zero(result)


def nonfinite_disagreement(golden, result):
    """Positions where the non-finite check inside comp_ulp would fire: a NaN
    facing an inf fails, and so do two infs of opposite sign."""
    return (
        (torch.isnan(golden) ^ torch.isnan(result))
        | (torch.isinf(golden) ^ torch.isinf(result))
        | (torch.isinf(golden) & torch.isinf(result) & (torch.signbit(golden) != torch.signbit(result)))
    )


def term_scaled_error(exact, golden, result, term1, term2):
    """|result - exact| in ULP of max(|term1|, |term2|).

    0 where both sides flushed to zero, and where either side is non-finite:
    those positions belong to the non-finite check instead.
    """
    measured = torch.isfinite(golden) & torch.isfinite(result) & ~((golden == 0) & (result == 0))
    error = (result.to(torch.float64) - exact).abs() / _ulp_f32(torch.maximum(term1.abs(), term2.abs()))
    return torch.where(measured, error, torch.zeros_like(error))


def term_scaled_failures(exact, golden, term1, term2, bound):
    return lambda result: term_scaled_error(exact, golden, result, term1, term2) > bound


def ulp_failures(golden):
    return lambda result: (ulp_distance(golden, result) > 0) & ~(torch.isnan(golden) & torch.isnan(result))


def assert_term_scaled(exact, golden, result, term1, term2, bound, desc):
    disagreeing = nonfinite_disagreement(golden, result)
    assert not disagreeing.any(), (
        f"{desc}: {int(disagreeing.sum())} non-finite mismatches, first golden "
        f"{golden[disagreeing][0].item()!r} vs result {result[disagreeing][0].item()!r}"
    )
    error = term_scaled_error(exact, golden, result, term1, term2)
    worst = int(error.argmax())
    assert error.flatten()[worst] <= bound, (
        f"{desc}: |result - exact| reaches {error.flatten()[worst].item():.3f} ULP of the larger term "
        f"(bound {bound}): result {result.flatten()[worst].item()!r}, exact {exact.flatten()[worst].item()!r}"
    )


def apply_exceptions(golden, result, stages):
    """Rewrite ``result`` to the golden wherever a documented exception holds.

    ``stages`` are ``(name, build_mask)`` pairs applied in order. Each
    ``build_mask`` receives the result as rewritten so far and asserts what
    the device is documented to do over the range it is about to excuse. A
    failure is re-raised carrying the stage name, and a stage that matches
    nothing fails too, since it is a hole in the sweep rather than a saving.
    """
    for name, build_mask in stages:
        try:
            mask = build_mask(result)
            assert mask.any(), "matched nothing, so this exception has gone stale"
        except AssertionError as error:
            raise AssertionError(f"exception stage '{name}': {error}") from error
        result = torch.where(mask, golden, result)
    return result


def intermediate_overflow_exception(golden, intermediates, zero_multiplier=None, saturates=False):
    """An intermediate past float32 max overflows on the device, which then
    returns ±inf where the float64 result is still finite (a = 2, b = 1.7014e38,
    c = -6.49e32: a * b = 3.4028e38 overflows, a * b + c does not).

    Where the overflowed intermediate next meets a zero multiplier
    (``zero_multiplier``) the device computes inf * 0 and returns NaN instead,
    while the float64 result keeps the finite addend.

    ``saturates`` also accepts exactly ±float32 max, same sign, where the
    golden is ±inf. Blackhole addcdiv does this when value * b / c is just
    past the float32 maximum.
    """
    region = intermediates[0].abs() > FLOAT32_MAX
    for intermediate in intermediates[1:]:
        region = region | (intermediate.abs() > FLOAT32_MAX)
    nan_allowed = torch.zeros_like(region) if zero_multiplier is None else zero_multiplier.expand_as(region)

    def build_mask(result):
        mask = region & nonfinite_disagreement(golden, result)
        expected = torch.isinf(result) | (nan_allowed & torch.isnan(result))
        if saturates:
            expected = expected | (
                torch.isinf(golden) & (result.abs() == FLOAT32_MAX) & (torch.signbit(result) == torch.signbit(golden))
            )
        assert expected[mask].all(), "device is expected to overflow to ±inf, or NaN against a zero multiplier" + (
            ", or ±float32 max where the golden is ±inf" if saturates else ""
        )
        return mask

    return ("intermediate overflow", build_mask)


def collapses_exception(name, region, expected, failures, detail, *, whole_region=False):
    """An intermediate lands on a float32 subnormal, the device flushes it,
    and the result collapses to ``expected`` (the addend).

    Asserted rather than assumed, so a regression landing on any other value
    still fails. By default only the positions that fail the test's final
    check are taken: the device keeps some products the golden would call
    subnormal (0.5 * 2^-126 survives as 2^-127 inside the multiply-add), and
    there it is simply right. ``whole_region`` asserts the collapse over the
    whole range, for flushes of a first intermediate that nothing can bring
    back.
    """

    def build_mask(result):
        mask = region.expand_as(result)
        if not whole_region:
            mask = mask & failures(result)
        assert torch.equal(result[mask], expected.expand_as(result)[mask]), detail
        return mask

    return (name, build_mask)


def _subnormal(intermediate):
    return (intermediate != 0) & (intermediate.abs() < MIN_NORMAL_F32)


# ─────────────────────────────────────────────────────────────────────────────
# grid — exhaustive 3-way outer product
# ─────────────────────────────────────────────────────────────────────────────


def test_mac(device):
    """ttnn.mac over the full 3-way grid: exact after two exceptions.

    Grid mantissas are 8 bits, so a * b is exact in float32 and the only
    rounding left is the final one: the device must match the correctly
    rounded result bit for bit. test_full_mantissa covers the rounded product.

    | # | Stage                 | Range                | Device returns        |
    |---|-----------------------|----------------------|-----------------------|
    | 1 | intermediate overflow | abs(a * b) > f32 max | ±inf                  |
    | 2 | a * b underflows      | abs(a * b) < 2^-126  | c, where it disagrees |

        a = 1.1755e-38, b = 2^-16, c = 1.1755e-38: a * b = 1.79e-43 is
        flushed, so the device returns c where the golden is c + 1.79e-43.
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=True, dtype=torch.float32)
    result = _run(device, ttnn.mac, input_a, input_b, input_c)

    product = input_a.double() * input_b.double()
    golden = _golden(product + input_c.double())
    stages = [
        intermediate_overflow_exception(golden, [product]),
        collapses_exception(
            "a * b underflows",
            _subnormal(product),
            input_c,
            ulp_failures(golden),
            "device is expected to return c when a * b flushes to zero",
        ),
    ]
    result = apply_exceptions(golden, result, stages)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


def test_lerp(device):
    """ttnn.lerp over the full 3-way grid, 2.5 ULP of the larger term.

    The terms are a and weight * (b - a). b - a is rounded before it is
    scaled, which is the rounding the grid does exercise: its operands have
    different exponents.

    | # | Stage                       | Range                          | Device returns         |
    |---|-----------------------------|--------------------------------|------------------------|
    | 1 | b - a underflows            | abs(b - a) < 2^-126            | a, whole range         |
    | 2 | intermediate overflow       | abs(b - a) or abs(weight *     | ±inf; NaN for weight 0 |
    |   |                             | (b - a)) > f32 max             | (inf * 0)              |
    | 3 | weight * (b - a) underflows | abs(weight * (b - a)) < 2^-126 | a, where it disagrees  |

        a = 1.1755e-38, b = 1.1847e-38, weight = 65536: b - a = 9.18e-41 is
        flushed, so the device returns a where the golden is 6.03e-36.
        a = 2.1268e37, b = -2.1268e37, weight = 8: weight * (b - a) is
        -3.4028e38 and overflows, a + weight * (b - a) = -3.1901e38 does not.
    """
    input_a, input_b, input_weight = ternary_inputs(include_zero=True, dtype=torch.float32)
    result = _run(device, ttnn.lerp, input_a, input_b, input_weight)

    a64, b64, w64 = (t.double() for t in (input_a, input_b, input_weight))
    difference = b64 - a64
    scaled = w64 * difference.to(torch.float32).double()
    exact = a64 + w64 * difference
    golden = _golden(exact)
    failures = term_scaled_failures(exact, golden, a64, w64 * difference, LERP_TERM_ULP)
    stages = [
        collapses_exception(
            "b - a underflows",
            _subnormal(difference),
            input_a,
            failures,
            "device is expected to return a when b - a flushes to zero",
            whole_region=True,
        ),
        intermediate_overflow_exception(golden, [difference, scaled], zero_multiplier=w64 == 0),
        collapses_exception(
            "weight * (b - a) underflows",
            _subnormal(scaled),
            input_a,
            failures,
            "device is expected to return a when weight * (b - a) flushes to zero",
        ),
    ]
    result = apply_exceptions(golden, result, stages)

    assert_term_scaled(exact, golden, result, a64, w64 * difference, LERP_TERM_ULP, "lerp")


@pytest.mark.parametrize("value", ADDC_VALUES)
def test_addcmul(device, value):
    """ttnn.addcmul over the full 3-way grid: a + value * b * c.

    Both sides form value * b first. With a power-of-two value that product
    is exact, so is (value * b) * c on the grid, and the device must match
    the correctly rounded result; with value = 0.3 the first product rounds
    and the bound is 3 ULP of the larger term. value = 0 has to return a.

    | # | Stage                    | Range                            | Device returns        |
    |---|--------------------------|----------------------------------|-----------------------|
    | 1 | intermediate overflow    | abs(value * b) or abs(value * b  | ±inf; NaN for c = 0   |
    |   |                          | * c) > f32 max                   | (inf * 0)             |
    | 2 | value * b underflows     | abs(value * b) < 2^-126,         | a, whole range        |
    |   |                          | only abs(value) < 1              |                       |
    | 3 | value * b * c underflows | abs(value * b * c) < 2^-126      | a, where it disagrees |
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=True, dtype=torch.float32)
    result = _run(device, ttnn.addcmul, input_a, input_b, input_c, value=value)

    if value == 0.0:
        assert torch.equal(result, input_a), "a + 0 * b * c is expected to return a unchanged"
        return

    a64 = input_a.double()
    scaled_b = value * input_b.double()
    term = scaled_b * input_c.double()
    exact = a64 + term
    golden = _golden(exact)
    exact_scale = _is_exact_scale(value)
    failures = (
        ulp_failures(golden)
        if exact_scale
        else term_scaled_failures(exact, golden, a64, term, ADDCMUL_ROUNDED_VALUE_TERM_ULP)
    )
    stages = [intermediate_overflow_exception(golden, [scaled_b, term], zero_multiplier=input_c == 0)]
    if abs(value) < 1.0:
        stages.append(
            collapses_exception(
                "value * b underflows",
                _subnormal(scaled_b),
                input_a,
                failures,
                "device is expected to return a when value * b flushes to zero",
                whole_region=True,
            )
        )
    stages.append(
        collapses_exception(
            "value * b * c underflows",
            _subnormal(term),
            input_a,
            failures,
            "device is expected to return a when value * b * c flushes to zero",
        )
    )
    result = apply_exceptions(golden, result, stages)

    if exact_scale:
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)
    else:
        assert_term_scaled(exact, golden, result, a64, term, ADDCMUL_ROUNDED_VALUE_TERM_ULP, "addcmul")


@pytest.mark.parametrize("value", ADDC_VALUES)
def test_addcdiv(device, value):
    """ttnn.addcdiv over the full 3-way grid: a + value * b / c, c nonzero.

    The device multiplies value * b by recip(c), two Newton steps, so even on
    the grid the quotient rounds; the bound is 3 ULP of the larger term, 4
    with value = 0.3. value = 0 has to return a. Zero divisors are excluded,
    as in the binary divide sweep.

    Three of the results below are Blackhole-specific. The ULP bounds are not.

    | # | Stage                    | Range                           | Device returns         |
    |---|--------------------------|---------------------------------|------------------------|
    | 1 | intermediate overflow    | abs(value * b) or abs(value * b | ±inf; NaN where        |
    |   |                          | / c) > f32 max                  | recip(c) flushed       |
    |   |                          |                                 | (inf * 0)              |
    | 2 | value * b underflows     | abs(value * b) < 2^-126,        | a, whole range         |
    |   |                          | only abs(value) < 1             |                        |
    | 3 | recip(c) flushes to zero | abs(c) >= 2^126                 | a, where it disagrees  |
    | 4 | value * b / c underflows | abs(value * b / c) < 2^-126     | a, where it disagrees  |
    | 5 | sum flushes at 2^-126    | abs(a + value * b / c) ==       | 0, where it disagrees  |
    |   | (Blackhole)              | 2^-126                          |                        |

    On Blackhole:

    - Stage 1 also accepts exactly ±float32 max (±3.4028234663852886e38),
      same sign, where the golden is ±inf. The quotient sits just past the
      float32 maximum and the device saturates instead of overflowing.
    - Stage 4 stays strictly below 2^-126. A subnormal quotient is flushed
      and the device returns a, so that cutoff is not widened.
    - Stage 5 is the next case. value * b / c is ±2^-125, a normal, and it
      cancels with a = ∓2^-126. The sum is the smallest normal, and the
      device returns 0. Only a power-of-two value makes the quotient exactly
      ±2^-125, so the stage is not added for value = 0.3.
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=False, dtype=torch.float32)
    result = _run(device, ttnn.addcdiv, input_a, input_b, input_c, value=value)

    if value == 0.0:
        assert torch.equal(result, input_a), "a + 0 * b / c is expected to return a unchanged"
        return

    a64 = input_a.double()
    scaled_b = value * input_b.double()
    term = scaled_b / input_c.double()
    exact = a64 + term
    golden = _golden(exact)
    bound = ADDCDIV_TERM_ULP if _is_exact_scale(value) else ADDCDIV_ROUNDED_VALUE_TERM_ULP
    failures = term_scaled_failures(exact, golden, a64, term, bound)
    reciprocal_flushes = input_c.double().abs() >= RECIPROCAL_FLUSH_DIVISOR
    stages = [
        intermediate_overflow_exception(
            golden,
            [scaled_b, term],
            zero_multiplier=reciprocal_flushes,
            saturates=is_blackhole(),
        )
    ]
    if abs(value) < 1.0:
        stages.append(
            collapses_exception(
                "value * b underflows",
                _subnormal(scaled_b),
                input_a,
                failures,
                "device is expected to return a when value * b flushes to zero",
                whole_region=True,
            )
        )
    stages += [
        collapses_exception(
            "recip(c) flushes to zero",
            reciprocal_flushes,
            input_a,
            failures,
            "device is expected to return a when recip(c) flushes to zero",
        ),
        collapses_exception(
            "value * b / c underflows",
            _subnormal(term),
            input_a,
            failures,
            "device is expected to return a when value * b / c flushes to zero",
        ),
    ]
    # ±2^-126 is the smallest normal. A subnormal cutoff does not reach a sum
    # that lands exactly there, and the device returns 0 rather than a. The
    # flush comes from a = ∓2^-126 cancelling a quotient of exactly ±2^-125,
    # which only a power-of-two value can produce on this grid.
    sum_is_min_normal = exact.abs() == MIN_NORMAL_F32
    if is_blackhole() and _is_exact_scale(value):
        stages.append(
            collapses_exception(
                "sum flushes at 2^-126",
                sum_is_min_normal,
                torch.zeros_like(input_a),
                failures,
                "device is expected to return 0 when the sum is ±2^-126",
            )
        )
    result = apply_exceptions(golden, result, stages)

    assert_term_scaled(exact, golden, result, a64, term, bound, "addcdiv")


# ─────────────────────────────────────────────────────────────────────────────
# full mantissa — the grid only carries bfloat16 mantissas
# ─────────────────────────────────────────────────────────────────────────────


def _random_full_mantissa(generator, min_exponent, max_exponent, shape=(256, 4096)):
    """Random sign, an exponent uniform in [min_exponent, max_exponent] and all
    23 mantissa bits random. Spreading the exponent rather than the value is
    what makes the two terms of the final add cancel often enough to matter."""
    mantissa = torch.randint(0, 1 << 23, shape, generator=generator, dtype=torch.int64)
    exponent = torch.randint(min_exponent + 127, max_exponent + 128, shape, generator=generator, dtype=torch.int64)
    sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    return ((sign << 31) | (exponent << 23) | mantissa).to(torch.int32).view(torch.float32)


@pytest.mark.parametrize(
    "op_name, value",
    [("mac", None), ("lerp", None)]
    + [("addcmul", value) for value in (1.0, -0.5, 2.0, 0.3)]
    + [("addcdiv", value) for value in (1.0, -0.5, 2.0, 0.3)],
)
def test_full_mantissa(device, op_name, value):
    """Seeded random float32 operands with all 23 mantissa bits populated.

    Exponents span [-60, 60] (weight [-10, 3]), so no intermediate under- or
    overflows and the bound applies everywhere, cancellation included:

    | Op      | Terms of the final add | Bound, ULP of the larger term          |
    |---------|------------------------|----------------------------------------|
    | mac     | a * b, c               | 1.5                                    |
    | lerp    | a, weight * (b - a)    | 2.5                                    |
    | addcmul | a, value * b * c       | 1.5, 3 for value = 0.3                 |
    | addcdiv | a, value * b / c       | 3, 4 for value = 0.3                   |
    """
    generator = torch.Generator().manual_seed(4)
    input_a = _random_full_mantissa(generator, -60, 60)
    input_b = _random_full_mantissa(generator, -60, 60)
    input_c = (
        _random_full_mantissa(generator, -10, 3) if op_name == "lerp" else _random_full_mantissa(generator, -60, 60)
    )
    a64, b64, c64 = (t.double() for t in (input_a, input_b, input_c))

    if op_name == "mac":
        result = _run(device, ttnn.mac, input_a, input_b, input_c)
        term1, term2, bound = a64 * b64, c64, MAC_TERM_ULP
    elif op_name == "lerp":
        result = _run(device, ttnn.lerp, input_a, input_b, input_c)
        term1, term2, bound = a64, c64 * (b64 - a64), LERP_TERM_ULP
    elif op_name == "addcmul":
        result = _run(device, ttnn.addcmul, input_a, input_b, input_c, value=value)
        exact_scale = _is_exact_scale(value)
        term1, term2 = a64, value * b64 * c64
        bound = ADDCMUL_TERM_ULP if exact_scale else ADDCMUL_ROUNDED_VALUE_TERM_ULP
    else:
        result = _run(device, ttnn.addcdiv, input_a, input_b, input_c, value=value)
        exact_scale = _is_exact_scale(value)
        term1, term2 = a64, value * b64 / c64
        bound = ADDCDIV_TERM_ULP if exact_scale else ADDCDIV_ROUNDED_VALUE_TERM_ULP

    exact = term1 + term2
    assert_term_scaled(exact, _golden(exact), result, term1, term2, bound, op_name)


# ─────────────────────────────────────────────────────────────────────────────
# where — a selection, so the bar is bit equality
# ─────────────────────────────────────────────────────────────────────────────


def build_predicate(pattern, shape):
    """0/1 predicates laid out to vary within a tile, across tiles, and not at all.

    The kernel reads the predicate a face at a time, so a selection that
    changes every element, every tile row, or not at all exercises different
    paths through it.
    """
    rows, cols = shape
    row_index = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    col_index = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    patterns = {
        "all_true": torch.ones(shape),
        "all_false": torch.zeros(shape),
        "checkerboard": (row_index + col_index) % 2,
        "row_blocks": (row_index // ttnn.TILE_SIZE) % 2,
        "col_blocks": (col_index // ttnn.TILE_SIZE) % 2,
        "tile_diagonal": row_index // ttnn.TILE_SIZE == col_index // ttnn.TILE_SIZE,
    }
    return patterns[pattern].to(torch.float32)


def _assert_bitwise_selection(predicate, input_true, input_false, result):
    golden = torch.where(predicate != 0, input_true, input_false)
    differing = golden.view(torch.int32) != result.view(torch.int32)
    assert not differing.any(), (
        f"{int(differing.sum())} selected values came back altered, first golden "
        f"{golden[differing][0].item()!r} vs result {result[differing][0].item()!r}"
    )


def _run_where(device, predicate, input_true, input_false):
    tensors = [to_float32_tt_tensor(t, device) for t in (predicate, input_true, input_false)]
    result = ttnn.to_torch(ttnn.where(*tensors))
    assert result.dtype == torch.float32, f"expected float32 output, got {result.dtype}"
    return result


@pytest.mark.parametrize(
    "pattern", ["all_true", "all_false", "checkerboard", "row_blocks", "col_blocks", "tile_diagonal"]
)
@pytest.mark.parametrize("include_spl_values", [False, True])
def test_where_ttt(device, pattern, include_spl_values):
    """Pairwise coverage of the two ttnn.where branches over the binary grid.

    Unlike bfloat16, the float32 selection is a pure 32-bit copy: NaN keeps
    its encoding and -0 keeps its sign, so special values are held to bit
    equality as well.
    """
    input_true, input_false = pairwise_inputs(
        include_spl_values=include_spl_values, include_zero=True, dtype=torch.float32
    )
    predicate = build_predicate(pattern, input_true.shape)
    result = _run_where(device, predicate, input_true, input_false)
    _assert_bitwise_selection(predicate, input_true, input_false, result)


def test_where_full_mantissa(device):
    """Random full-mantissa branches under a checkerboard predicate.

    The grid carries bfloat16 values, whose low 16 bits are all zero, so a
    selection that dropped them would pass test_where_ttt untouched.
    """
    generator = torch.Generator().manual_seed(4)
    input_true = _random_full_mantissa(generator, -126, 127)
    input_false = _random_full_mantissa(generator, -126, 127)
    predicate = build_predicate("checkerboard", input_true.shape)
    result = _run_where(device, predicate, input_true, input_false)
    _assert_bitwise_selection(predicate, input_true, input_false, result)


@pytest.mark.parametrize(
    "predicate_value",
    [
        0.0,
        pytest.param(
            -0.0,
            marks=pytest.mark.xfail(
                strict=True,
                reason="#56522: the float32 kernel tests the raw bits against zero, so -0.0 selects the true branch",
            ),
        ),
        1.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        MIN_NORMAL_F32,
        1.0e-40,
    ],
    ids=["+0", "-0", "1", "-1", "nan", "inf", "-inf", "min_normal", "subnormal"],
)
def test_where_predicate_special_values(device, predicate_value):
    """A float32 predicate follows torch: true wherever it is nonzero.

    NaN and the infinities are nonzero, so they select the true branch, and
    so does a subnormal predicate, which the float32 path does not flush.
    -0.0 equals zero and should select the false branch.
    """
    generator = torch.Generator().manual_seed(4)
    shape = (ttnn.TILE_SIZE * 2, ttnn.TILE_SIZE * 2)
    input_true = _random_full_mantissa(generator, -20, 20, shape)
    input_false = _random_full_mantissa(generator, -20, 20, shape)
    predicate = torch.full(shape, predicate_value, dtype=torch.float32)
    result = _run_where(device, predicate, input_true, input_false)
    _assert_bitwise_selection(predicate, input_true, input_false, result)
