# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_binary_grid,
    int32_pairwise_inputs,
    pairwise_from_values,
    run_int32_binary,
    scalar_rows,
    wrap_int32,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the integer-only binary ops: bitwise, shift, gcd and
lcm. They have no float32 or bfloat16 counterpart file; int32 is their main
dtype (the bitwise and shift ops also take uint32 and uint16).

Tensor-tensor inputs come from int32_pairwise_inputs: the outer product of
the 2048-value int32 binary grid (every integer in [-256, 256), ±2^k and
±(2^k ± 1) for every k including 2^24 + 1, INT32_MIN and INT32_MAX,
log-uniform and uniform random values), 4,194,304 pairs. Shift amounts are
swept separately, against every grid value.

Int32 category 4: integer-only ops

| #  | Op                       | Computes                    | Output |
|----|--------------------------|-----------------------------|--------|
| 1  | ttnn.bitwise_and         | a & b                       | int32  |
| 2  | ttnn.bitwise_or          | a | b                       | int32  |
| 3  | ttnn.bitwise_xor         | a ^ b                       | int32  |
| 4  | ttnn.bitwise_left_shift  | a << b (wraps mod 2^32)     | int32  |
| 5  | ttnn.logical_left_shift  | a << b (wraps mod 2^32)     | int32  |
| 6  | ttnn.bitwise_right_shift | a >> b (arithmetic)         | int32  |
| 7  | ttnn.logical_right_shift | a >> b (zero fill)          | int32  |
| 8  | ttnn.gcd                 | gcd(a, b), non-negative     | int32  |
| 9  | ttnn.lcm                 | lcm(a, b), non-negative     | int32  |

A shift amount outside [0, 32) gives 0 for every shift, a documented contract
of the shift kernels. torch also gives 0 for the left shifts, but its
arithmetic right shift fills with the sign bit, giving -1 for a negative a,
so out-of-range shifts are checked against 0 rather than torch.

gcd(INT32_MIN, 0) and gcd(INT32_MIN, INT32_MIN) are 2^31, which wraps to
INT32_MIN. lcm is documented for [-32768, 32767] and swept on
the grid values in that range.

The tensor-scalar forms of the bitwise ops and of bitwise_left_shift,
bitwise_right_shift and logical_left_shift lower to UnaryDeviceOperation and
are in the int32 unary category 2 file; logical_right_shift with a scalar
runs on BinaryNgDeviceOperation and is swept here.

Accuracy criteria

| Op                    | Input range                   | Check            |
|-----------------------|-------------------------------|------------------|
| bitwise_and, _or,     | full grid                     | exact            |
| _xor                  |                               |                  |
| shifts                | shift in [0, 32)              | exact            |
|                       | shift < 0 or ≥ 32             | exact 0          |
| logical_right_shift   | scalar shift 0, 1, 4, 16, 30, | exact            |
|   (scalar)            | 31, 32                        |                  |
| gcd                   | full grid                     | exact            |
| lcm                   | grid ∩ [-32768, 32767]        | exact            |
"""

SHIFT_OPS = [ttnn.bitwise_left_shift, ttnn.logical_left_shift, ttnn.bitwise_right_shift, ttnn.logical_right_shift]
OUT_OF_RANGE_SHIFTS = [-1, -31, -32, 32, 33, 64, INT32_MAX, INT32_MIN]
SCALAR_SHIFTS = [0, 1, 4, 16, 30, 31, 32]
LCM_RANGE = (-32768, 32767)


def _shift_golden(ttnn_op, input_a, shift):
    """In-range shifts only: shift in [0, 32)."""
    a = input_a.to(torch.int64)
    s = shift.to(torch.int64) if torch.is_tensor(shift) else shift
    if ttnn_op in (ttnn.bitwise_left_shift, ttnn.logical_left_shift):
        return wrap_int32(a << s)
    if ttnn_op is ttnn.bitwise_right_shift:
        return wrap_int32(a >> s)
    return wrap_int32((a & 0xFFFFFFFF) >> s)


def _shift_inputs(shifts):
    """Every grid value against every shift amount."""
    return pairwise_from_values(generate_int32_binary_grid(), torch.tensor(shifts, dtype=torch.int32))


# ─────────────────────────────────────────────────────────────────────────────
# bitwise_and, bitwise_or, bitwise_xor — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.bitwise_and, ttnn.bitwise_or, ttnn.bitwise_xor])
def test_int32_bitwise_ops(device, ttnn_op):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | bit operations on the two's complement words    |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    golden = ttnn.get_golden_function(ttnn_op)(input_a, input_b)
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# Shifts — exact in range, 0 out of range
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", SHIFT_OPS)
def test_int32_shift_ops(device, ttnn_op):
    """Every grid value shifted by every amount in [0, 32).

    | Input range      | Check | Reason                                     |
    |------------------|-------|--------------------------------------------|
    | shift in [0, 32) | exact | left shifts wrap mod 2^32; bitwise_right_  |
    |                  |       | shift fills with the sign bit,             |
    |                  |       | logical_right_shift with zeros             |
    """
    input_a, shift = _shift_inputs(list(range(32)))

    result = run_int32_binary(device, ttnn_op, input_a, shift)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_shift_golden(ttnn_op, input_a, shift), result)


@pytest.mark.parametrize("ttnn_op", SHIFT_OPS)
def test_int32_shift_ops_out_of_range(device, ttnn_op):
    """Every grid value shifted by a negative amount or by 32 or more.

    | Input range          | Check   | Reason                                 |
    |----------------------|---------|----------------------------------------|
    | shift < 0 or ≥ 32    | exact 0 | the shift kernels' contract; torch     |
    |                      |         | agrees except that its arithmetic      |
    |                      |         | right shift of a negative a gives -1   |
    """
    input_a, shift = _shift_inputs(OUT_OF_RANGE_SHIFTS)

    result = run_int32_binary(device, ttnn_op, input_a, shift)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(torch.zeros_like(input_a), result)


@pytest.mark.parametrize("shift", SCALAR_SHIFTS)
def test_int32_logical_right_shift_scalar(device, shift):
    """The grid shifted right, zero fill, by an int scalar.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | a shift of 32 gives 0                           |
    """
    input_a = scalar_rows(generate_int32_binary_grid())

    result = run_int32_binary(device, ttnn.logical_right_shift, input_a, shift)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    golden = _shift_golden(ttnn.logical_right_shift, input_a, shift) if shift < 32 else torch.zeros_like(input_a)
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# gcd, lcm — exact
# ─────────────────────────────────────────────────────────────────────────────


def test_int32_gcd(device):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | non-negative result; gcd(a, 0) = abs(a), and    |
    |             |       | 2^31 wraps to INT32_MIN                         |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn.gcd, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(wrap_int32(torch.gcd(input_a.to(torch.int64), input_b.to(torch.int64))), result)


def test_int32_lcm(device):
    """Pairwise coverage over the grid values in lcm's documented range.

    | Input range       | Check | Reason                                    |
    |-------------------|-------|-------------------------------------------|
    | [-32768, 32767]   | exact | non-negative result below 2^30, so no     |
    |                   |       | wrap; lcm(a, 0) = 0                       |
    """
    values = generate_int32_binary_grid()
    values = values[(values >= LCM_RANGE[0]) & (values <= LCM_RANGE[1])]
    input_a, input_b = pairwise_from_values(values)

    result = run_int32_binary(device, ttnn.lcm, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(wrap_int32(torch.lcm(input_a.to(torch.int64), input_b.to(torch.int64))), result)
