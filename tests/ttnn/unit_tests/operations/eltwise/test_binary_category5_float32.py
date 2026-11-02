# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import (
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
Category 5: division family + power

| # | Op                                | Computes             | Check (float32)                                   |
|---|-----------------------------------|----------------------|---------------------------------------------------|
| 1 | ttnn.divide, rounding_mode        | trunc/floor(a / b)   | trunc/floor of a quotient within 1 ULP            |
|   | "trunc" / "floor"                 |                      |                                                   |
| 2 | ttnn.remainder                    | a - floor(a / b) * b | exact for abs(a / b) < 2^16 on the grid; within   |
| 3 | ttnn.fmod                         | a - trunc(a / b) * b | 1 ULP(max(abs(a), abs(b))) modulo b elsewhere;    |
|   |                                   |                      | abs(a / b) ≥ 2^128 overflows and is not checked.  |
|   |                                   |                      | Scalar b: 0.5 ULP (Blackhole xfail, #63732). Sign |
|   |                                   |                      | and abs(result) ≤ abs(b) wherever the tolerance   |
|   |                                   |                      | is below abs(b)                                   |
| 4 | ttnn.div_no_nan                   | a / b, 0 where b = 0 | ULP ≤ 1                                           |
| 5 | ttnn.floor_div                    | floor(a / b)         | floor of a quotient within 1 ULP                  |
| 6 | ttnn.pow                          | a ** exponent        | ULP ≤ 2, more where the error grows with          |
|   |                                   |                      | abs(y * log2(a)) (#59761)                         |

Plain divide (rounding_mode=None) is test_divide in the category 1 float32
file. Here it only appears as the quotient the other ops are built on, plus
the fast_and_approximate_mode and output_tensor variants.

Working range of the accuracy sweeps. "Full grid" is the stratified binary
grid at float32: finite bfloat16 values and +0, no -0, inf or NaN.

| Op                  | a                       | b or exponent          | Outside the range, pinned by          |
|---------------------|-------------------------|------------------------|---------------------------------------|
| divide trunc/floor  | full grid, no 0         | full grid, no 0        | —                                     |
| div_no_nan          | full grid               | full grid, ±0, ±inf,   | —                                     |
| floor_div           | full grid               | NaN                    | —                                     |
| remainder, fmod     | ±[2^-118, 2^118]        | ±[2^-118, 2^118]       | — (see test_modulo_ops)               |
| remainder, fmod,    | grid ±[2^-8, 2^7]; ±4   | listed scalars         | —                                     |
| scalar              | ULP of every multiple   |                        |                                       |
|                     | of s in ±[abs(s), 100]  |                        |                                       |
| pow, scalar         | full grid               | listed exponents       | test_pow_scalar_exponent_underflow,   |
|                     |                         |                        | test_pow_scalar_zero_base_negative_   |
|                     |                         |                        | exponent (both xfail, #62383)         |
| pow, tensor         | positive grid           | grid in [-20, 20]      | test_pow_tensor_exponent_nonpositive_ |
|                     |                         |                        | base                                  |

The grid only carries bfloat16 mantissas. For the modulo ops that matters:
q * b is exact on the grid while q < 2^16, but for a full 24-bit divisor
only while q ≤ 2.
So divide trunc/floor, floor_div, remainder, fmod and tensor pow are also
swept on random full-mantissa inputs (test_full_mantissa).

Goldens are float64 rounded to float32. The device flushes subnormal results
to zero, so those elements are expected to be exactly 0 wherever the float64
result is below 2^-126.

On float32 every op here runs on the SFPU, so fast_and_approximate_mode has no
effect, and the bfloat16 file's NaN-as-inf and reciprocal-underflow
corrections do not apply.
"""

FLOAT32_TINY = float(torch.finfo(torch.float32).tiny)

# remainder/fmod compute a - q * b with q = trunc/floor(a * recip(b)), rounding
# the product q * b. On the grid (8-bit mantissas) that product is exact while
# q < 2^16. Operands below 2^-118 are excluded: some of those return 0 where
# the answer is a.
MODULO_MIN_OPERAND = 2.0**-118
MODULO_MAX_OPERAND = 2.0**118
MODULO_EXACT_QUOTIENT = 2.0**16
# a * recip(b) overflows float32 from here on.
MODULO_OVERFLOW_QUOTIENT = 2.0**128

_DIV_SCALAR_VALUES = [-5.1, 0.0, 10.9, 0.1, 3.0]

_POW_INT_EXPONENTS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
_POW_FLOAT_EXPONENTS = [-2.75, -1.5, -0.5, 0.5, 1.5, 2.5, 7.25, 0.3]

# Exponents with a short mantissa stay within 2 ULP. 0.3 uses all 24 bits, and
# the kernel's single-precision y * log2(x) loses the rest (#59761).
_POW_SCALAR_ULP = {0.3: 17}
_POW_SCALAR_DEFAULT_ULP = 2

# pow(tensor, tensor), banded on abs(y * log2(x)) (#59761).
POW_SMALL_PRODUCT = 1.0
POW_SMALL_PRODUCT_ULP = 3
POW_LARGE_PRODUCT_ULP = 16


def _float32_result(result, desc):
    """comp_ulp sizes the ULP from the result's dtype, so a bfloat16 output
    would be measured in bfloat16 ULPs and pass float32 thresholds."""
    assert result.dtype == torch.float32, f"{desc}: expected float32 output, got {result.dtype}"
    return result


def _quotient(input_a, input_b):
    """a / b in float64, rounded to float32, subnormals flushed."""
    return flush_subnormal_values_to_zero((input_a.double() / input_b.double()).to(torch.float32))


def _ulp32(values):
    """Size of one float32 ULP at abs(values), as float64."""
    magnitude = values.abs().to(torch.float32)
    return (torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude).double()


def _assert_rounded_quotient(round_fn, quotient, result, desc):
    """result is round_fn of a quotient within 1 ULP of the correctly rounded one.

    The device rounds its own divide, which is 1 ULP accurate (category 1).
    That ULP crosses an integer once abs(q) ≥ 2^22, or for floor when a
    negative q sits just below an integer.
    """
    _float32_result(result, desc)
    up = torch.nextafter(quotient, torch.full_like(quotient, float("inf")))
    down = torch.nextafter(quotient, torch.full_like(quotient, -float("inf")))
    both_nan = torch.isnan(quotient) & torch.isnan(result)
    allowed = (result == round_fn(quotient)) | (result == round_fn(up)) | (result == round_fn(down)) | both_nan
    if not allowed.all():
        first = tuple((~allowed).nonzero()[0].tolist())
        raise AssertionError(
            f"{desc}: {int((~allowed).sum())} results are not the rounding of a quotient within 1 ULP; "
            f"first: quotient={quotient[first].item()!r}, result={result[first].item()!r}"
        )


def _assert_modulo_within(golden_fn, result, input_a, input_b, ulp_scale, max_ulp, desc):
    """remainder/fmod results against golden_fn (torch.remainder or torch.fmod) in float64.

    | Check                         | Where                     | Bound                             |
    |-------------------------------|---------------------------|-----------------------------------|
    | abs(result - golden), modulo  | everywhere                | max_ulp ULPs of ulp_scale         |
    | abs(b)                        |                           |                                   |
    | sign: a for fmod, b for       | tolerance < abs(b),       | exact                             |
    | remainder                     | result nonzero            |                                   |
    | abs(result) ≤ abs(b)          | tolerance < abs(b)        | plus the tolerance                |

    Modulo b because a quotient off by one moves the result by exactly b:
    fmod(a, 0.1) can come back as 0.1 - tiny where the golden is +tiny. That
    alone would also accept golden - b, so sign and range are checked too.
    Once the tolerance reaches abs(b), either sign is within it. Wormhole
    returns up to 17 ULP(b) past abs(b) for negative a whose magnitude is
    just below a multiple of a small scalar; that stays inside the tolerance.
    """
    _float32_result(result, desc)
    golden = golden_fn(input_a.double(), input_b.double())
    error = (result.double() - golden).abs()
    divisor = input_b.double().abs()
    circular = torch.minimum(error, (divisor - error).abs())
    tolerance = max_ulp * _ulp32(ulp_scale)
    in_range = tolerance < divisor
    sign_operand = input_a if golden_fn is torch.fmod else input_b

    def describe(index):
        return (
            f"a={input_a.reshape(-1)[index].item()!r}, b={input_b.reshape(-1)[index].item()!r}, "
            f"golden={golden.reshape(-1)[index].item()!r}, actual={result.reshape(-1)[index].item()!r}"
        )

    failed = ~(circular <= tolerance)
    if failed.any():
        worst = int(torch.argmax(circular - tolerance))
        raise AssertionError(
            f"{desc}: {int(failed.sum())} results are more than {max_ulp} ULP off modulo b; worst is {describe(worst)}"
        )
    wrong_sign = in_range & (result != 0) & (torch.signbit(result) != torch.signbit(sign_operand))
    too_large = in_range & ~(result.double().abs() <= divisor + tolerance)
    for failed, what in ((wrong_sign, "have the wrong sign"), (too_large, "exceed abs(b)")):
        if failed.any():
            first = int(failed.reshape(-1).nonzero()[0])
            raise AssertionError(f"{desc}: {int(failed.sum())} results {what}; first is {describe(first)}")


def _assert_pow_bands(golden, reference, result, input_a, exponent, small_ulp, desc):
    """pow banded on abs(y * log2(x)) (#59761): ULP ≤ small_ulp below 1,
    ULP ≤ POW_LARGE_PRODUCT_ULP above. Underflow is exactly 0, and the
    non-finite classes match the golden."""
    _float32_result(result, desc)
    underflow = reference.abs() < FLOAT32_TINY
    assert_equal(torch.zeros_like(result[underflow]), result[underflow])

    golden, result = golden[~underflow], result[~underflow]
    input_a, exponent = input_a[~underflow], exponent[~underflow]
    assert torch.equal(torch.isnan(golden), torch.isnan(result)), f"{desc}: NaN positions differ"
    overflow = torch.isinf(golden)
    assert torch.equal(overflow, torch.isinf(result)), f"{desc}: overflow positions differ"
    assert torch.equal(golden[overflow], result[overflow]), f"{desc}: overflow signs differ"

    finite = torch.isfinite(golden)
    product = (exponent.double() * torch.log2(input_a.double().abs())).abs()
    small = finite & (product < POW_SMALL_PRODUCT)
    large = finite & ~small
    if small.any():
        assert_with_ulp(expected_result=golden[small], actual_result=result[small], ulp_threshold=small_ulp)
    if large.any():
        assert_with_ulp(expected_result=golden[large], actual_result=result[large], ulp_threshold=POW_LARGE_PRODUCT_ULP)


def _assert_divide_by_zero(input_a, result, rounding_mode, desc):
    """torch's a / 0: ±inf by the sign of a, NaN for 0 / 0; trunc and floor
    keep both."""
    golden = torch.div(input_a, torch.zeros_like(input_a), rounding_mode=rounding_mode)
    nan = torch.isnan(golden)
    assert torch.equal(nan, torch.isnan(_float32_result(result, desc))), f"{desc}: NaN positions differ"
    assert_equal(golden[~nan], result[~nan])


def _scalar_quotient(input_a, scalar):
    """a / scalar in float64, with the scalar as the Python float. The device
    multiplies by float32(1 / float32(scalar)), which for 0.1 is exactly 10."""
    return flush_subnormal_values_to_zero((input_a.double() / scalar).to(torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# divide with rounding_mode — the SFPU divide, then a unary trunc/floor
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rounding_mode, round_fn", [("trunc", torch.trunc), ("floor", torch.floor)])
def test_divide_rounding_modes(device, rounding_mode, round_fn):
    """Full grid without 0.

    | Case                    | float32 behavior                              | Check             |
    |-------------------------|-----------------------------------------------|-------------------|
    | abs(q) < 2^22           | trunc is exact; floor differs only where a    | rounding of a     |
    |                         | negative q is 1 ULP below an integer          | quotient within   |
    | abs(q) ≥ 2^22           | the 1 ULP divide error crosses an integer     | 1 ULP             |
    | subnormal q             | flushed to 0, so trunc and floor give 0       | (q is flushed)    |
    | overflow                | ±inf on both sides                            |                   |

    The result is also bitwise round_fn(ttnn.divide(a, b)): the rounding adds
    no error of its own.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values(include_zero=False, dtype=torch.float32))
    tt_a, tt_b = to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)
    result = ttnn.to_torch(ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode))

    _assert_rounded_quotient(round_fn, _quotient(input_a, input_b), result, f"divide {rounding_mode}")
    plain = ttnn.to_torch(ttnn.divide(tt_a, tt_b))
    assert_equal(round_fn(plain), result)


def _divide_tile_inputs(device):
    """One tile of mixed-sign fractions with no zero divisor, for tests that
    compare two calls bitwise: what they check does not depend on the values."""
    input_a = (torch.arange(-512, 512, dtype=torch.float32) / 3).reshape(32, 32)
    input_b = (torch.arange(-511.5, 512.5, dtype=torch.float32) / 7).reshape(32, 32)
    return input_a, to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide_fast_mode_is_accurate(device, rounding_mode):
    """Float32 always takes the SFPU divide, so fast_and_approximate_mode=True
    is bitwise identical to False for every rounding mode. bfloat16 instead
    takes the FPU path for rounding_mode=None."""
    _, tt_a, tt_b = _divide_tile_inputs(device)

    accurate = ttnn.to_torch(ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode, fast_and_approximate_mode=False))
    fast = ttnn.to_torch(ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode, fast_and_approximate_mode=True))
    assert torch.equal(accurate.view(torch.int32), fast.view(torch.int32))


def test_divide_output_tensor(device):
    """A preallocated output_tensor= receives bitwise the plain call's result."""
    input_a, tt_a, tt_b = _divide_tile_inputs(device)
    output_tensor = to_float32_tt_tensor(torch.zeros_like(input_a), device)

    ttnn.divide(tt_a, tt_b, output_tensor=output_tensor)
    result = _float32_result(ttnn.to_torch(output_tensor), "divide output_tensor")
    expected = ttnn.to_torch(ttnn.divide(tt_a, tt_b))
    assert torch.equal(expected.view(torch.int32), result.view(torch.int32))


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True])
def test_divide_scalar(device, fast_and_approximate_mode, rounding_mode, scalar):
    """Tensor-scalar divide over the full grid. The overload multiplies by a
    host-computed float32 reciprocal: two roundings, within 1 ULP of the
    quotient. trunc/floor round that product, as in test_divide_rounding_modes.

    | scalar | Check                                              |
    |--------|----------------------------------------------------|
    | 0      | exact class: ±inf by the sign of a, NaN for 0 / 0  |
    | other  | ULP ≤ 1, or rounding of a quotient within 1 ULP    |

    The scalar is packed at float32 precision, so 0.1 is a float32-only value.
    """
    input_a = binary_grid_values(dtype=torch.float32).reshape(-1, 32)
    result = ttnn.to_torch(
        ttnn.divide(
            to_float32_tt_tensor(input_a, device),
            scalar,
            rounding_mode=rounding_mode,
            fast_and_approximate_mode=fast_and_approximate_mode,
        )
    )
    desc = f"divide(tensor, {scalar}, rounding_mode={rounding_mode})"
    _float32_result(result, desc)

    if scalar == 0.0:
        _assert_divide_by_zero(input_a, result, rounding_mode, desc)
        return

    quotient = _scalar_quotient(input_a, scalar)
    if rounding_mode is None:
        assert_with_ulp(expected_result=quotient, actual_result=result, ulp_threshold=1, allow_nonfinite=True)
    else:
        _assert_rounded_quotient(torch.trunc if rounding_mode == "trunc" else torch.floor, quotient, result, desc)


# ─────────────────────────────────────────────────────────────────────────────
# remainder / fmod — a - q * b, q = floor / trunc(a * recip(b))
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn", [(ttnn.remainder, torch.remainder), (ttnn.fmod, torch.fmod)], ids=["remainder", "fmod"]
)
def test_modulo_ops(device, ttnn_op, golden_fn):
    """Grid operands in ±[2^-118, 2^118], all pairs.

    | abs(a / b)        | float32 behavior                                 | Check            |
    |-------------------|--------------------------------------------------|------------------|
    | < 2^16            | q * b is exact for 8-bit mantissas, so only the  | exact            |
    |                   | final subtract (remainder: add b) rounds         |                  |
    | [2^16, 2^128)     | q * b rounds. Once ULP(a) > abs(b) (from about   | ULP(a) ≤ 1,      |
    |                   | 2^24) the result can exceed abs(b)               | modulo b         |
    | ≥ 2^128           | a * recip(b) overflows: Wormhole returns ±inf or | not checked      |
    |                   | NaN everywhere, Blackhole on most lanes          |                  |

    torch returns a finite remainder in the last band. Below 2^-118 some
    pairs return 0 where the answer is a (e.g. fmod(2^-126, 1.5 * 2^-126));
    that range is not swept.
    """
    values = binary_grid_values(
        -MODULO_MAX_OPERAND,
        MODULO_MAX_OPERAND,
        min_magnitude=MODULO_MIN_OPERAND,
        include_zero=False,
        dtype=torch.float32,
    )
    input_a, input_b = pairwise_from_values(values)
    result = ttnn.to_torch(ttnn_op(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))
    result = _float32_result(result, ttnn_op.__name__)
    golden = flush_subnormal_values_to_zero(golden_fn(input_a.double(), input_b.double()).to(torch.float32))

    quotient = (input_a.double() / input_b.double()).abs()
    exact = quotient < MODULO_EXACT_QUOTIENT
    overflow = quotient >= MODULO_OVERFLOW_QUOTIENT
    rounded = ~exact & ~overflow
    assert exact.any() and rounded.any() and overflow.any(), "expected all three quotient bands in this sweep"

    assert_equal(golden[exact], result[exact])
    _assert_modulo_within(
        golden_fn,
        result[rounded],
        input_a[rounded],
        input_b[rounded],
        ulp_scale=input_a[rounded],
        max_ulp=1,
        desc=f"{ttnn_op.__name__}, abs(a / b) in [2^16, 2^128)",
    )


@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
def test_modulo_ops_by_zero(device, ttnn_op):
    """A zero divisor gives NaN for every a, as torch does; tensor and scalar
    overloads alike. (bfloat16 returns -inf for the scalar.)"""
    input_a = binary_grid_values(dtype=torch.float32).reshape(-1, 32)
    tt_a = to_float32_tt_tensor(input_a, device)
    tensor_result = ttnn.to_torch(ttnn_op(tt_a, to_float32_tt_tensor(torch.zeros_like(input_a), device)))
    scalar_result = ttnn.to_torch(ttnn_op(tt_a, 0.0))
    assert torch.isnan(tensor_result).all(), f"{ttnn_op.__name__}(a, zeros) must be NaN"
    assert torch.isnan(scalar_result).all(), f"{ttnn_op.__name__}(a, 0.0) must be NaN"


_MODULO_SCALAR_VALUES = [-0.002, 0.001, 0.1, 2.5, -3.0, 7.0]


@pytest.mark.parametrize(
    "ttnn_op, golden_fn", [(ttnn.remainder, torch.remainder), (ttnn.fmod, torch.fmod)], ids=["remainder", "fmod"]
)
@pytest.mark.parametrize("scalar", _MODULO_SCALAR_VALUES)
def test_modulo_ops_scalar(device, ttnn_op, golden_fn, scalar):
    """Tensor-scalar remainder/fmod (the unary SFPU kernels) over the grid in
    ±[2^-8, 2^7].

    | scalar           | Check                                  | Measured        |
    |------------------|----------------------------------------|-----------------|
    | 2.5, -3, 7       | exact                                  | exact           |
    | -0.002, 0.001,   | ULP(max(abs(a), abs(scalar))) ≤ 0.5,   | 0.5             |
    | 0.1              | modulo the scalar                      |                 |

    The second group has full float32 mantissas, so q * scalar rounds.
    """
    values = binary_grid_values(-128.0, 128.0, min_magnitude=2.0**-8, include_zero=False, dtype=torch.float32)
    input_a = values.reshape(1, -1).expand(32, -1).contiguous()
    result = _float32_result(ttnn.to_torch(ttnn_op(to_float32_tt_tensor(input_a, device), scalar)), "modulo scalar")
    scalar_tensor = torch.full_like(input_a, scalar)

    if scalar in (2.5, -3.0, 7.0):
        golden = golden_fn(input_a.double(), scalar_tensor.double())
        assert_equal(flush_subnormal_values_to_zero(golden.to(torch.float32)), result)
        return
    _assert_modulo_within(
        golden_fn,
        result,
        input_a,
        scalar_tensor,
        ulp_scale=torch.maximum(input_a.abs(), scalar_tensor.abs()),
        max_ulp=0.5,
        desc=f"{ttnn_op.__name__}(tensor, {scalar})",
    )


MODULO_NEAR_MULTIPLE_LIMIT = 100.0
MODULO_NEAR_MULTIPLE_NEIGHBORS = 4


def _near_multiple_dividends(scalar):
    """float32(n * s) and its MODULO_NEAR_MULTIPLE_NEIGHBORS float32 neighbors
    on each side, for every n ≥ 1 with n * abs(s) ≤ MODULO_NEAR_MULTIPLE_LIMIT,
    both signs. s is the float32 scalar the device sees."""
    magnitude = abs(float(torch.tensor(scalar, dtype=torch.float32)))
    count = int(MODULO_NEAR_MULTIPLE_LIMIT / magnitude)
    centers = (torch.arange(1, count + 1, dtype=torch.float64) * magnitude).to(torch.float32)
    columns, up, down = [centers], centers, centers
    for _ in range(MODULO_NEAR_MULTIPLE_NEIGHBORS):
        up = torch.nextafter(up, torch.full_like(up, float("inf")))
        down = torch.nextafter(down, torch.zeros_like(down))
        columns += [up, down]
    values = torch.cat(columns)
    values = values[values <= MODULO_NEAR_MULTIPLE_LIMIT]
    return torch.cat([values, -values])


@pytest.mark.parametrize(
    "ttnn_op, golden_fn", [(ttnn.remainder, torch.remainder), (ttnn.fmod, torch.fmod)], ids=["remainder", "fmod"]
)
@pytest.mark.parametrize("scalar", [-0.002, 0.001])
@pytest.mark.xfail(
    is_blackhole(),
    strict=True,
    reason="#63732: Blackhole scalar remainder/fmod exceeds 0.5 ULP next to multiples of the scalar",
)
def test_modulo_ops_scalar_near_multiples(device, ttnn_op, golden_fn, scalar):
    """Full-mantissa dividends within ±4 ULP of every multiple of the scalar
    in ±[abs(s), 100].

    Here a / s is within a fraction of an ULP of an integer, so
    trunc(a * recip(s)) can come out one too large. The Blackhole misses in
    #63732 are all of this kind.

    | Check                                            | Measured                  |
    |--------------------------------------------------|---------------------------|
    | ULP(max(abs(a), abs(s))) ≤ 0.5, modulo the       | Wormhole 0.5 (also over   |
    | scalar; sign and range as in                     | every float32 in          |
    | _assert_modulo_within                            | [-100, 100]);             |
    |                                                  | Blackhole 0.77 (#63732)   |
    """
    input_a = _near_multiple_dividends(scalar)
    result = _float32_result(
        ttnn.to_torch(ttnn_op(to_float32_tt_tensor(input_a, device), scalar)).reshape(-1), "modulo near multiple"
    )
    scalar_tensor = torch.full_like(input_a, scalar)
    _assert_modulo_within(
        golden_fn,
        result,
        input_a,
        scalar_tensor,
        ulp_scale=torch.maximum(input_a.abs(), scalar_tensor.abs()),
        max_ulp=0.5,
        desc=f"{ttnn_op.__name__}(near multiples of {scalar}, {scalar})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# div_no_nan / floor_div — composites over the SFPU divide
# ─────────────────────────────────────────────────────────────────────────────


def _special_divisor_inputs():
    """a over the full grid with +0; b over the full grid plus ±0, ±inf, NaN."""
    specials = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")])
    values_b = torch.cat([binary_grid_values(include_zero=False, dtype=torch.float32), specials])
    return pairwise_from_values(binary_grid_values(dtype=torch.float32), values_b)


def test_div_no_nan(device):
    """where(b == 0, 0, a / b), with 0 / 0 included.

    | b        | float32 behavior            | Check   |
    |----------|-----------------------------|---------|
    | ±0       | +0 for every a              | exact   |
    | ±inf     | a / inf = ±0                | ULP ≤ 1 |
    | NaN      | NaN, as a / NaN             | ULP ≤ 1 |
    | other    | the 1 ULP SFPU divide       | ULP ≤ 1 |

    bfloat16 returns 0 for a NaN divisor; float32 keeps the NaN.
    """
    input_a, input_b = _special_divisor_inputs()
    _, result = run_binary(device, ttnn.div_no_nan, input_a, input_b, input_dtype=ttnn.float32)
    result = _float32_result(result, "div_no_nan")

    zero_divisor = input_b == 0
    assert_equal(torch.zeros_like(result[zero_divisor]), result[zero_divisor])
    assert not torch.signbit(result[zero_divisor]).any(), "div_no_nan(a, 0) must be +0"
    golden = _quotient(input_a, input_b)
    assert_with_ulp(
        expected_result=golden[~zero_divisor],
        actual_result=flush_subnormal_values_to_zero(result[~zero_divisor]),
        ulp_threshold=1,
        allow_nonfinite=True,
    )


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
def test_div_no_nan_scalar(device, scalar):
    """Tensor-scalar div_no_nan: zeros for a zero scalar, otherwise a multiply
    by the host float32 reciprocal, within 1 ULP."""
    input_a = binary_grid_values(dtype=torch.float32).reshape(-1, 32)
    result = _float32_result(
        ttnn.to_torch(ttnn.div_no_nan(to_float32_tt_tensor(input_a, device), scalar)), "div_no_nan scalar"
    )
    if scalar == 0.0:
        assert_equal(torch.zeros_like(result), result)
        return
    assert_with_ulp(
        expected_result=_scalar_quotient(input_a, scalar), actual_result=result, ulp_threshold=1, allow_nonfinite=True
    )


def test_floor_div(device):
    """floor(a / b), keeping a / b where it is ±inf.

    | b          | float32 behavior                     | Check                       |
    |------------|--------------------------------------|-----------------------------|
    | ±0         | ±inf by the signs, NaN for 0 / 0     | floor of a quotient within  |
    | ±inf, NaN  | a / ±inf = ±0, floored to ±0; NaN    | 1 ULP, NaN positions match  |
    | other      | floor of the 1 ULP SFPU divide       |                             |

    bfloat16 returns 0 for a NaN divisor; float32 keeps the NaN.
    """
    input_a, input_b = _special_divisor_inputs()
    result = ttnn.to_torch(ttnn.floor_div(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))
    _assert_rounded_quotient(torch.floor, _quotient(input_a, input_b), result, "floor_div")


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
def test_floor_div_scalar(device, scalar):
    """Tensor-scalar floor_div: floor(a * float32(1 / scalar)). A zero scalar
    gives ±inf by the sign of a and NaN for 0, as torch does."""
    input_a = binary_grid_values(dtype=torch.float32).reshape(-1, 32)
    result = ttnn.to_torch(ttnn.floor_div(to_float32_tt_tensor(input_a, device), scalar))
    desc = f"floor_div(tensor, {scalar})"
    if scalar == 0.0:
        _assert_divide_by_zero(input_a, result, "floor", desc)
        return
    _assert_rounded_quotient(torch.floor, _scalar_quotient(input_a, scalar), result, desc)


# ─────────────────────────────────────────────────────────────────────────────
# pow — scalar exponents (unary power, power_iterative for 0..3) and tensor
# exponents (binary POWER)
# ─────────────────────────────────────────────────────────────────────────────


def _pow_scalar_run(device, exponent):
    input_a = binary_grid_values(dtype=torch.float32).reshape(-1, 32)
    result = ttnn.to_torch(ttnn.pow(to_float32_tt_tensor(input_a, device), exponent))
    reference = torch.pow(input_a.double(), exponent)
    return input_a, reference, _float32_result(result, f"pow(tensor, {exponent})")


@pytest.mark.parametrize("exponent", _POW_INT_EXPONENTS + _POW_FLOAT_EXPONENTS)
def test_pow_scalar_exponent(device, exponent):
    """tensor ** scalar over the full grid, both signs of the base.

    | Case                      | float32 behavior                     | Check                  |
    |---------------------------|--------------------------------------|------------------------|
    | integer or k/4 exponent   | ≤ 2 ULP                              | ULP ≤ 2                |
    | exponent 0.3              | y * log2(x) loses the low bits of a  | ULP ≤ 17 (#59761)      |
    |                           | 24-bit exponent                      |                        |
    | negative base, non-int y  | NaN                                  | NaN positions match    |
    | overflow                  | ±inf                                 | class and sign match   |
    | float64 result < 2^-126   | 0, or NaN for some exponents         | not here: test_pow_    |
    |                           | (#62383)                             | scalar_exponent_       |
    |                           |                                      | underflow              |
    | 0 ** negative             | NaN instead of inf (#62383)          | not here               |
    """
    input_a, reference, result = _pow_scalar_run(device, exponent)
    keep = (reference.abs() >= FLOAT32_TINY) | torch.isnan(reference) | ((reference == 0) & (input_a == 0))
    if exponent < 0:
        keep &= input_a != 0
    golden = reference.to(torch.float32)
    golden, result = golden[keep], result[keep]

    assert torch.equal(torch.isnan(golden), torch.isnan(result)), f"pow(tensor, {exponent}): NaN positions differ"
    overflow = torch.isinf(golden)
    assert torch.equal(overflow, torch.isinf(result)), f"pow(tensor, {exponent}): overflow positions differ"
    assert torch.equal(golden[overflow], result[overflow]), f"pow(tensor, {exponent}): overflow signs differ"
    finite = torch.isfinite(golden)
    assert_with_ulp(
        expected_result=golden[finite],
        actual_result=result[finite],
        ulp_threshold=_POW_SCALAR_ULP.get(exponent, _POW_SCALAR_DEFAULT_ULP),
    )


def _pow_underflow_xfail(exponent):
    return pytest.param(
        exponent,
        marks=pytest.mark.xfail(
            strict=True, reason="#62383: float32 unary pow returns NaN for some results below 2^-126"
        ),
    )


@pytest.mark.parametrize(
    "exponent",
    [-3, -2, -1, -4, 2, 3, 1.5, -1.5] + [_pow_underflow_xfail(e) for e in (-5, 4, 5, -2.75, 2.5, 7.25)],
)
def test_pow_scalar_exponent_underflow(device, exponent):
    """Bases whose float64 x ** y is below 2^-126 must give exactly 0.

    | Exponent                       | Device                               |
    |--------------------------------|--------------------------------------|
    | -4..-1, 2, 3, ±1.5             | 0                                    |
    | -5, 4, 5, -2.75, 2.5, 7.25     | NaN on part of the band (#62383)     |
    """
    input_a, reference, result = _pow_scalar_run(device, exponent)
    underflow = (reference.abs() < FLOAT32_TINY) & (input_a != 0)
    assert underflow.any(), "expected underflowing results in this sweep"
    assert_equal(torch.zeros_like(result[underflow]), result[underflow])


@pytest.mark.parametrize("exponent", [-3, -2, -1, -0.5, -1.5])
@pytest.mark.xfail(strict=True, reason="#62383: float32 unary pow(0, negative) returns NaN instead of inf")
def test_pow_scalar_zero_base_negative_exponent(device, exponent):
    """0 ** negative is +inf (torch, IEEE 754). The tensor-exponent path
    already returns it; the scalar path returns NaN."""
    input_a = torch.zeros((32, 32), dtype=torch.float32)
    result = ttnn.to_torch(ttnn.pow(to_float32_tt_tensor(input_a, device), exponent))
    assert_equal(torch.full_like(result, float("inf")), result)


def test_pow_tensor_exponent(device):
    """Positive grid bases against grid exponents in [-20, 20], all pairs.

    | Case                    | Check                     | Measured |
    |-------------------------|---------------------------|----------|
    | abs(y * log2(x)) < 1    | ULP ≤ 3                   | 2        |
    | abs(y * log2(x)) ≥ 1    | ULP ≤ 16 (#59761)         | 15       |
    | float64 result < 2^-126 | exactly 0                 | 0        |
    | overflow                | +inf                      | +inf     |

    abs(y) is capped at 20 here; #55129 covers abs(y) > 8.3e34.
    """
    values = binary_grid_values(dtype=torch.float32)
    input_a, exponent = pairwise_from_values(values[values > 0], binary_grid_values(-20.0, 20.0, dtype=torch.float32))
    result = ttnn.to_torch(ttnn.pow(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(exponent, device)))
    reference = torch.pow(input_a.double(), exponent.double())

    assert (reference.abs() < FLOAT32_TINY).any(), "expected underflowing results in this sweep"
    _assert_pow_bands(
        reference.to(torch.float32), reference, result, input_a, exponent, POW_SMALL_PRODUCT_ULP, "pow(tensor, tensor)"
    )


_POW_NONPOSITIVE_BASE_EXPONENTS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, -0.5, 1.5, 2.25]


def test_pow_tensor_exponent_nonpositive_base(device):
    """Negative grid bases and +0 against integer and non-integer exponents.

    | Base   | Exponent     | Check                                      |
    |--------|--------------|--------------------------------------------|
    | < 0    | integer      | sign from the parity of y; ULP ≤ 3         |
    | < 0    | non-integer  | NaN                                        |
    | +0     | < 0          | +inf (the scalar path's NaN is #62383)     |
    | +0     | 0 / > 0      | 1 / 0                                      |
    """
    values = binary_grid_values(dtype=torch.float32)
    bases = torch.cat([values[values < 0], torch.tensor([0.0])])
    input_a, exponent = pairwise_from_values(bases, torch.tensor(_POW_NONPOSITIVE_BASE_EXPONENTS))
    result = ttnn.to_torch(ttnn.pow(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(exponent, device)))
    result = _float32_result(result, "pow(nonpositive tensor, tensor)")
    reference = torch.pow(input_a.double(), exponent.double())

    zero_base = input_a == 0
    assert_equal(reference[zero_base].to(torch.float32), result[zero_base])
    reference, result = reference[~zero_base], result[~zero_base]
    underflow = reference.abs() < FLOAT32_TINY
    assert_equal(torch.zeros_like(result[underflow]), result[underflow])
    assert_with_ulp(
        expected_result=reference[~underflow].to(torch.float32),
        actual_result=result[~underflow],
        ulp_threshold=3,
        allow_nonfinite=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# full mantissa — the grid only carries bfloat16 mantissas
# ─────────────────────────────────────────────────────────────────────────────


def _uniform(generator, low, high, shape=(256, 1024)):
    return (torch.rand(shape, generator=generator, dtype=torch.float64) * (high - low) + low).to(torch.float32)


@pytest.mark.parametrize("kind", ["divide_trunc", "divide_floor", "floor_div", "remainder", "fmod", "pow"])
def test_full_mantissa(device, kind):
    """Seeded random float32 operands with all 23 mantissa bits populated.

    | Op                   | a            | b              | Check                                 |
    |----------------------|--------------|----------------|---------------------------------------|
    | divide trunc/floor,  | [-1e3, 1e3]  | [-1e3, 1e3]    | rounding of a quotient within 1 ULP   |
    | floor_div            |              |                | (measured: exact)                     |
    | remainder, fmod      | [-1e4, 1e4]  | [-10, 10]      | ULP(max(abs(a), abs(b))) ≤ 1, modulo  |
    |                      |              |                | b (measured: remainder 1, fmod 0.84)  |
    | pow                  | (0.1, 10]    | [-10, 10]      | as test_pow_tensor_exponent           |
    |                      |              |                | (measured: 3 / 13)                    |

    A full 24-bit divisor rounds q * b once q > 2, so remainder and fmod are
    exact only on the grid.
    """
    generator = torch.Generator().manual_seed(4)
    if kind == "pow":
        input_a = _uniform(generator, 0.1, 10.0)
        exponent = _uniform(generator, -10.0, 10.0)
        result = ttnn.to_torch(ttnn.pow(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(exponent, device)))
        reference = torch.pow(input_a.double(), exponent.double())
        _assert_pow_bands(
            reference.to(torch.float32), reference, result, input_a, exponent, POW_SMALL_PRODUCT_ULP, "pow"
        )
        return

    if kind in ("remainder", "fmod"):
        input_a, input_b = _uniform(generator, -1e4, 1e4), _uniform(generator, -10.0, 10.0)
        ttnn_op, golden_fn = (ttnn.remainder, torch.remainder) if kind == "remainder" else (ttnn.fmod, torch.fmod)
        result = ttnn.to_torch(ttnn_op(to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)))
        _assert_modulo_within(
            golden_fn,
            result,
            input_a,
            input_b,
            ulp_scale=torch.maximum(input_a.abs(), input_b.abs()),
            max_ulp=1,
            desc=kind,
        )
        return

    input_a, input_b = _uniform(generator, -1e3, 1e3), _uniform(generator, -1e3, 1e3)
    tt_a, tt_b = to_float32_tt_tensor(input_a, device), to_float32_tt_tensor(input_b, device)
    if kind == "floor_div":
        result = ttnn.to_torch(ttnn.floor_div(tt_a, tt_b))
        round_fn = torch.floor
    else:
        rounding_mode = kind.removeprefix("divide_")
        result = ttnn.to_torch(ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode))
        round_fn = torch.trunc if rounding_mode == "trunc" else torch.floor
    _assert_rounded_quotient(round_fn, _quotient(input_a, input_b), result, kind)
