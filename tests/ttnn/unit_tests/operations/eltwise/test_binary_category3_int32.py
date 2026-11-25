# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tt_py_test_utils_common.utils_for_testing import ulp_distance
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_binary_grid,
    int32_pairwise_inputs,
    run_int32_binary,
    scalar_rows,
    wrap_int32,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the division family and power for ttnn.int32 input:
the int32 subset of category 5 (division family + power) of the float32 and
bfloat16 files, plus plain divide from category 1.

Tensor-tensor inputs come from int32_pairwise_inputs: the outer product of
the 2048-value int32 binary grid (every integer in [-256, 256), ±2^k and
±(2^k ± 1) for every k including 2^24 + 1, INT32_MIN and INT32_MAX,
log-uniform and uniform random values), 4,194,304 pairs.

Int32 category 3: division family and power
(float32 / bfloat16 category 5, and divide from category 1)

| # | Op                                 | Computes                | Output  |
|---|------------------------------------|-------------------------|---------|
| 1 | ttnn.divide                        | a / b                   | float32 |
| 2 | ttnn.divide, rounding_mode "trunc" | trunc(a / b)            | int32   |
|   | / "floor"                          | / floor(a / b)          |         |
| 3 | ttnn.remainder                     | a - floor(a / b) * b    | int32   |
| 4 | ttnn.fmod                          | a - trunc(a / b) * b    | int32   |
| 5 | ttnn.div_no_nan                    | a / b, 0 where b = 0    | float32 |
| 6 | ttnn.pow (int scalar exponent)     | a ** n                  | int32   |

Plain int32 divide returns float32, as torch's int / int does. The kernel
converts both operands to float32 (rounding |x| > 2^24, as torch does) and
divides with one residual correction step, so the golden is torch.div on the
int32 tensors: float32(a) / float32(b). The rounding modes, remainder and
fmod are integer kernels and exact, as is torch.floor_divide; INT32_MIN / -1
wraps to INT32_MIN.

The int32 divide kernel loads its operands in sign-magnitude form, where
INT32_MIN (0x80000000) is -0, so INT32_MIN / b gives 0 and a / INT32_MIN
gives NaN on Blackhole and ±inf on Wormhole (NaN for 0 / INT32_MIN) in
divide and div_no_nan; those pairs are pinned by strict xfails. The
integer rounding modes, remainder and fmod are correct at INT32_MIN, as is
div_no_nan with a scalar divisor, which multiplies by the float32 reciprocal
of the scalar instead of running the divide kernel.

Integer division by zero raises in torch. On the device the float32 divide
gives ±inf and NaN as for float32 operands (pinned below); the rounding
modes, remainder and fmod have no defined result and are not swept at b = 0.

ttnn.floor_div is left out: for int32 tensors it returns the int32 quotient
words in a tensor marked float32, and its tensor-scalar form multiplies by
the float32 reciprocal of the scalar, so exact multiples of the scalar can
floor one low. ttnn.divide with rounding_mode="floor" is the int32 floor
division and is swept here. ttnn.pow rejects a Tensor exponent and a
negative exponent for int32 input.

Accuracy criteria

| Op                    | Input range                   | Check                 |
|-----------------------|-------------------------------|-----------------------|
| divide                | b ≠ 0, no INT32_MIN           | ULP ≤ 1 vs torch.div; |
|                       |                               | ULP ≤ 1 vs the        |
|                       |                               | correctly rounded     |
|                       |                               | quotient for          |
|                       |                               | abs(a), abs(b) ≤ 2^24 |
|                       | b = 0                         | ±inf, NaN for 0 / 0   |
|                       | an INT32_MIN operand, and     | strict xfail          |
|                       | INT32_MIN / 0                 |                       |
| divide trunc / floor, | b ≠ 0                         | exact                 |
| remainder, fmod       |                               |                       |
| div_no_nan            | no INT32_MIN                  | ULP ≤ 1 vs torch.div, |
|                       |                               | 0 where b = 0         |
|                       | an INT32_MIN tensor operand   | strict xfail          |
| div_no_nan (scalar)   | full grid, DIVISOR_SCALARS    | ULP ≤ 1 vs torch.div, |
|                       | and 0                         | 0 for scalar 0        |
| pow                   | exponent 0, 1, 2, 3, 5, 16,   | exact (wraps)         |
|                       | 31, 32                        |                       |
"""

DIVISOR_SCALARS = [1, -1, 2, 3, 7, -7, 100, 2**24 + 1, INT32_MIN, INT32_MAX]
POW_EXPONENTS = [0, 1, 2, 3, 5, 16, 31, 32]


def _assert_ulp(golden, result, ulp_threshold, desc):
    distance = ulp_distance(result, golden)
    worst = int(distance.max()) if distance.numel() else 0
    assert worst <= ulp_threshold, f"{desc}: worst {worst} ULP exceeds {ulp_threshold}"


def _nonzero(tensor):
    """Replace 0 by 1 so a divisor tensor keeps the grid's shape without dividing by zero."""
    return torch.where(tensor == 0, torch.ones_like(tensor), tensor)


def _int_division_golden(rounding_mode, input_a, input_b):
    b = input_b.to(torch.int64) if torch.is_tensor(input_b) else input_b
    a = input_a.to(torch.int64)
    if rounding_mode == "remainder":
        return wrap_int32(torch.remainder(a, b))
    if rounding_mode == "fmod":
        return wrap_int32(torch.fmod(a, b))
    return wrap_int32(torch.div(a, b, rounding_mode=rounding_mode))


INT_DIVISION_CASES = [
    pytest.param(ttnn.divide, "trunc", id="divide_trunc"),
    pytest.param(ttnn.divide, "floor", id="divide_floor"),
    pytest.param(ttnn.remainder, "remainder", id="remainder"),
    pytest.param(ttnn.fmod, "fmod", id="fmod"),
]


def _run_int_division(device, ttnn_op, mode, input_a, input_b):
    kwargs = {"rounding_mode": mode} if ttnn_op is ttnn.divide else {}
    return run_int32_binary(device, ttnn_op, input_a, input_b, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# divide (true division) — float32 result
# ─────────────────────────────────────────────────────────────────────────────


def test_int32_divide(device):
    """Pairwise coverage over the int32 grid without INT32_MIN, b ≠ 0.

    | Input range           | Check                   | Reason                 |
    |-----------------------|-------------------------|------------------------|
    | b ≠ 0, no INT32_MIN   | ULP ≤ 1 vs torch.div    | torch rounds both      |
    |                       |                         | operands to float32    |
    |                       |                         | first, as the kernel   |
    |                       |                         | does                   |
    | abs(a), abs(b) ≤ 2^24 | ULP ≤ 1 vs the          | the operands convert   |
    |                       | correctly rounded       | exactly, so only the   |
    |                       | quotient                | division rounds        |
    """
    input_a, input_b = int32_pairwise_inputs(include_int_min=False)
    input_b = _nonzero(input_b)

    result = run_int32_binary(device, ttnn.divide, input_a, input_b)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    _assert_ulp(torch.div(input_a, input_b), result, 1, "divide vs torch.div")

    exact_operands = (input_a.abs() <= 2**24) & (input_b.abs() <= 2**24)
    correctly_rounded = (input_a.double() / input_b.double()).float()
    _assert_ulp(correctly_rounded[exact_operands], result[exact_operands], 1, "divide vs the rounded quotient")


@pytest.mark.parametrize("scalar", [s for s in DIVISOR_SCALARS if s != INT32_MIN])
def test_int32_divide_scalar(device, scalar):
    """The grid without INT32_MIN divided by an int scalar.

    | Input range   | Check                | Reason                        |
    |---------------|----------------------|-------------------------------|
    | no INT32_MIN  | ULP ≤ 1 vs torch.div | as the tensor-tensor sweep    |
    """
    input_a = scalar_rows(generate_int32_binary_grid(include_int_min=False))

    result = run_int32_binary(device, ttnn.divide, input_a, scalar)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    _assert_ulp(torch.div(input_a, scalar), result, 1, f"divide by {scalar} vs torch.div")


def test_int32_divide_by_zero(device):
    """The grid without INT32_MIN divided by 0.

    | Input | Check | Reason                                                 |
    |-------|-------|--------------------------------------------------------|
    | a > 0 | +inf  | as float32 division; torch raises for int / int 0 only |
    | a < 0 | -inf  | under the rounding modes, not for true division        |
    | a = 0 | NaN   |                                                        |
    """
    input_a = scalar_rows(generate_int32_binary_grid(include_int_min=False))
    input_b = torch.zeros_like(input_a)

    result = run_int32_binary(device, ttnn.divide, input_a, input_b)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    golden = torch.div(input_a, input_b)
    assert torch.equal(torch.isnan(golden), torch.isnan(result)), "NaN only where a = 0"
    nonzero = input_a != 0
    assert_equal(golden[nonzero], result[nonzero])


@pytest.mark.xfail(
    strict=True,
    reason="the int32 divide kernel loads its operands as sign-magnitude, where INT32_MIN (0x80000000) is -0",
)
@pytest.mark.parametrize("divisor", ["tensor", "scalar"])
def test_int32_divide_int_min_by_zero(device, divisor):
    """INT32_MIN divided by a zero tensor or the scalar 0.

    | Input         | Check | Reason                                         |
    |---------------|-------|------------------------------------------------|
    | INT32_MIN / 0 | -inf  | INT32_MIN is negative; the device reads it as  |
    |               |       | -0 and returns NaN, as for 0 / 0               |
    """
    input_a = torch.full((32, 32), INT32_MIN, dtype=torch.int32)
    input_b = torch.zeros_like(input_a) if divisor == "tensor" else 0

    result = run_int32_binary(device, ttnn.divide, input_a, input_b)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    assert torch.isneginf(result).all(), f"INT32_MIN / 0 gave {result.flatten()[0].item()!r}, expected -inf"


@pytest.mark.xfail(
    strict=True,
    reason="the int32 divide kernel loads its operands as sign-magnitude, where INT32_MIN (0x80000000) is -0",
)
@pytest.mark.parametrize("operand", ["dividend", "divisor"])
def test_int32_divide_int_min(device, operand):
    """divide with INT32_MIN as the dividend or the divisor.

    | Input            | Check                | Reason                       |
    |------------------|----------------------|------------------------------|
    | INT32_MIN / b    | ULP ≤ 1 vs torch.div | the device returns 0         |
    | a / INT32_MIN    | ULP ≤ 1 vs torch.div | the device returns NaN on    |
    |                  |                      | Blackhole: the integer b ≠ 0 |
    |                  |                      | guard passes, so the         |
    |                  |                      | residual step runs on        |
    |                  |                      | 1 / -0 = -inf. On Wormhole   |
    |                  |                      | the guard skips it, giving   |
    |                  |                      | ±inf (NaN for 0 / INT32_MIN) |
    """
    values = _nonzero(generate_int32_binary_grid(include_int_min=False))
    int_min = torch.full_like(values, INT32_MIN)
    input_a, input_b = (int_min, values) if operand == "dividend" else (values, int_min)
    input_a, input_b = scalar_rows(input_a), scalar_rows(input_b)

    result = run_int32_binary(device, ttnn.divide, input_a, input_b)
    _assert_ulp(torch.div(input_a, input_b), result, 1, f"divide with INT32_MIN {operand}")


# ─────────────────────────────────────────────────────────────────────────────
# divide trunc / floor, remainder, fmod — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op, mode", INT_DIVISION_CASES)
def test_int32_integer_division_ops(device, ttnn_op, mode):
    """Pairwise coverage over the int32 grid, b ≠ 0.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | b ≠ 0       | exact | integer kernels; INT32_MIN / -1 wraps to        |
    |             |       | INT32_MIN under trunc and floor                 |
    """
    input_a, input_b = int32_pairwise_inputs()
    input_b = _nonzero(input_b)

    result = _run_int_division(device, ttnn_op, mode, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_int_division_golden(mode, input_a, input_b), result)


@pytest.mark.parametrize("scalar", DIVISOR_SCALARS)
@pytest.mark.parametrize("ttnn_op, mode", INT_DIVISION_CASES)
def test_int32_integer_division_ops_scalar(device, ttnn_op, mode, scalar):
    """The grid against an int scalar divisor.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | the scalar is packed as an int32                |
    """
    input_a = scalar_rows(generate_int32_binary_grid())

    result = _run_int_division(device, ttnn_op, mode, input_a, scalar)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_int_division_golden(mode, input_a, scalar), result)


# ─────────────────────────────────────────────────────────────────────────────
# div_no_nan — float32 result, 0 where b = 0
# ─────────────────────────────────────────────────────────────────────────────


def _div_no_nan_golden(input_a, input_b):
    quotient = torch.div(input_a, _nonzero(input_b) if torch.is_tensor(input_b) else input_b)
    if torch.is_tensor(input_b):
        quotient = torch.where(input_b == 0, torch.zeros_like(quotient), quotient)
    return quotient


def test_int32_div_no_nan(device):
    """Pairwise coverage over the int32 grid without INT32_MIN, b = 0 included.

    | Input range   | Check                | Reason                         |
    |---------------|----------------------|--------------------------------|
    | b ≠ 0         | ULP ≤ 1 vs torch.div | the divide kernel              |
    | b = 0         | exact 0              | selected by eqz(b)             |
    """
    input_a, input_b = int32_pairwise_inputs(include_int_min=False)

    result = run_int32_binary(device, ttnn.div_no_nan, input_a, input_b)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    golden = _div_no_nan_golden(input_a, input_b)
    _assert_ulp(golden, result, 1, "div_no_nan vs torch.div")
    assert_equal(torch.zeros_like(result[input_b == 0]), result[input_b == 0])


@pytest.mark.parametrize("scalar", [0] + DIVISOR_SCALARS)
def test_int32_div_no_nan_scalar(device, scalar):
    """The grid against an int scalar.

    | Input range   | Check                | Reason                         |
    |---------------|----------------------|--------------------------------|
    | scalar ≠ 0    | ULP ≤ 1 vs torch.div | a multiply by the float32      |
    |               |                      | reciprocal of the scalar, not  |
    |               |                      | the int32 divide kernel, so    |
    |               |                      | INT32_MIN is correct on either |
    |               |                      | side                           |
    | scalar = 0    | exact 0, int32       | zeros_like(a), keeping a's     |
    |               |                      | dtype                          |
    """
    input_a = scalar_rows(generate_int32_binary_grid())

    result = run_int32_binary(device, ttnn.div_no_nan, input_a, scalar)
    if scalar == 0:
        assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
        assert_equal(torch.zeros_like(input_a), result)
        return
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    _assert_ulp(_div_no_nan_golden(input_a, scalar), result, 1, f"div_no_nan by {scalar} vs torch.div")


@pytest.mark.xfail(
    strict=True,
    reason="div_no_nan divides with the int32 divide kernel, which reads INT32_MIN (0x80000000) as -0; "
    "on Wormhole its eqz(b) mask also reads an INT32_MIN divisor as zero (#60985)",
)
@pytest.mark.parametrize("operand", ["dividend", "divisor"])
def test_int32_div_no_nan_int_min(device, operand):
    """div_no_nan with INT32_MIN as the dividend or the divisor.

    | Input            | Check                | Reason                       |
    |------------------|----------------------|------------------------------|
    | INT32_MIN / b    | ULP ≤ 1 vs torch.div | the device returns 0         |
    | a / INT32_MIN    | ULP ≤ 1 vs torch.div | the device returns NaN, or 0 |
    |                  |                      | on Wormhole where eqz(b) is  |
    |                  |                      | true                         |
    """
    values = _nonzero(generate_int32_binary_grid(include_int_min=False))
    int_min = torch.full_like(values, INT32_MIN)
    input_a, input_b = (int_min, values) if operand == "dividend" else (values, int_min)
    input_a, input_b = scalar_rows(input_a), scalar_rows(input_b)

    result = run_int32_binary(device, ttnn.div_no_nan, input_a, input_b)
    _assert_ulp(_div_no_nan_golden(input_a, input_b), result, 1, f"div_no_nan with INT32_MIN {operand}")


# ─────────────────────────────────────────────────────────────────────────────
# pow (int scalar exponent) — exact, wrapping
# ─────────────────────────────────────────────────────────────────────────────


def _pow_golden(input_tensor, exponent):
    """a ** n with every partial product wrapped mod 2^32, as int32 multiplication does."""
    result = torch.ones_like(input_tensor, dtype=torch.int64)
    for _ in range(exponent):
        result = wrap_int32(result * input_tensor.to(torch.int64)).to(torch.int64)
    return result.to(torch.int32)


@pytest.mark.parametrize("exponent", POW_EXPONENTS)
def test_int32_pow_scalar(device, exponent):
    """The grid raised to a non-negative int exponent.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | repeated int32 multiplication; overflowing      |
    |             |       | powers wrap mod 2^32, and 0 ** 0 is 1           |
    """
    if exponent not in (0, 1) and device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("integer pow is not supported on Quasar")
    input_a = scalar_rows(generate_int32_binary_grid())

    result = run_int32_binary(device, ttnn.pow, input_a, exponent)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_pow_golden(input_a, exponent), result)
