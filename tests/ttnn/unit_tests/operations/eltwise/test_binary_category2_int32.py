# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_binary_grid,
    int32_pairwise_inputs,
    run_int32_binary,
    scalar_rows,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the binary comparison, min/max and logical ops that
accept ttnn.int32 input: the int32 subset of categories 2 (comparison +
min/max) and 3 (logical) of the float32 and bfloat16 files.

Tensor-tensor inputs come from int32_pairwise_inputs: the outer product of
the 2048-value int32 binary grid (every integer in [-256, 256), ±2^k and
±(2^k ± 1) for every k including 2^24 + 1, INT32_MIN and INT32_MAX,
log-uniform and uniform random values), 4,194,304 pairs.

Int32 category 2: comparison, min/max and logical
(float32 / bfloat16 categories 2 and 3)

| #  | Op                                 | Computes                | Output         |
|----|------------------------------------|-------------------------|----------------|
| 1  | ttnn.eq / ne / lt / le / gt / ge   | a op b                  | int32 (0 / 1)  |
| 2  | ttnn.eq_ / ne_ / lt_ / le_ / gt_ / | a op b (inplace)        | int32 (0 / 1)  |
|    | ge_                                |                         |                |
| 3  | ttnn.maximum / minimum             | elementwise max / min   | int32          |
| 4  | ttnn.isclose                       | abs(a - b) <= atol +    | float32 (0 / 1)|
|    |                                    | rtol * abs(b)           |                |
| 5  | ttnn.logical_and / or / xor        | (a != 0) op (b != 0)    | int32 (0 / 1)  |
| 6  | ttnn.logical_and_ / or_ / xor_     | logical op (inplace)    | int32 (0 / 1)  |

The relational ops compare int32 words directly, so values above 2^24 that
float32 would merge (2^24 and 2^24 + 1) stay distinct. isclose is the
exception: it converts both operands to float32 and returns 0 / 1 in
float32, so with zero tolerance 2^24 and 2^24 + 1 are close, where torch's
int32 isclose says they are not. The logical ops run nez on each operand,
the SFPU mul / add / sub of the two 0 / 1 values, then nez again.

The tensor-scalar forms of the relational and logical ops run on
BinaryNgDeviceOperation and are swept here. maximum and minimum with a scalar
lower to UnaryDeviceOperation and are in the int32 unary category 2 file.

Accuracy criteria

| Op                    | Input range              | Check        |
|-----------------------|--------------------------|--------------|
| relational (+ inplace)| full grid, and SCALARS   | exact        |
| maximum, minimum      | full grid                | exact        |
| isclose               | full grid                | exact vs     |
|                       |                          | float32      |
|                       |                          | operands     |
| logical (+ inplace)   | no INT32_MIN operand,    | exact        |
|                       | and LOGICAL_SCALARS      |              |
|   on Wormhole         | an INT32_MIN operand     | strict xfail |
"""

RELATIONAL_OPS = [ttnn.eq, ttnn.ne, ttnn.lt, ttnn.le, ttnn.gt, ttnn.ge]
RELATIONAL_INPLACE_OPS = [ttnn.eq_, ttnn.ne_, ttnn.lt_, ttnn.le_, ttnn.gt_, ttnn.ge_]
LOGICAL_OPS = [ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor]
LOGICAL_INPLACE_OPS = [ttnn.logical_and_, ttnn.logical_or_, ttnn.logical_xor_]

# 2^24 and 2^24 + 1 are one apart but the same float32, so a compare through
# float32 would call the grid's 2^24 + 1 equal to the scalar 2^24.
SCALARS = [0, 1, -1, 7, -100, 2**24, 2**24 + 1, -(2**24 + 1), INT32_MIN, INT32_MAX]
LOGICAL_SCALARS = [0, 1, -1, 7, INT32_MAX]

INT_MIN_WRONG_REASON = (
    "#60985: the logical ops run nez on each operand, and int32 nez reads INT32_MIN (0x80000000) as zero on Wormhole"
)


def _golden(ttnn_op, input_a, input_b):
    golden_function = ttnn.get_golden_function(ttnn_op)
    b = input_b if torch.is_tensor(input_b) else torch.tensor(input_b, dtype=input_a.dtype)
    return golden_function(input_a, b).to(torch.int32)


# ─────────────────────────────────────────────────────────────────────────────
# eq, ne, lt, le, gt, ge — exact 0 / 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", RELATIONAL_OPS + RELATIONAL_INPLACE_OPS)
def test_int32_relational_ops(device, ttnn_op):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | 0 / 1 result; torch returns bool, compared as   |
    |             |       | int32. INT32_MIN orders below every other value |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, input_b), result)


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("ttnn_op", RELATIONAL_OPS)
def test_int32_relational_ops_scalar(device, ttnn_op, scalar):
    """The grid against an int scalar, each value repeated across one tile width.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | the scalar is packed as an int32, so 2^24 and   |
    |             |       | 2^24 + 1 compare unequal                        |
    """
    input_a = scalar_rows(generate_int32_binary_grid())

    result = run_int32_binary(device, ttnn_op, input_a, scalar)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, scalar), result)


# ─────────────────────────────────────────────────────────────────────────────
# maximum, minimum — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.maximum, ttnn.minimum])
def test_int32_minmax_ops(device, ttnn_op):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                       |
    |-------------|-------|----------------------------------------------|
    | full grid   | exact | a select of one operand                      |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, input_b), result)


# ─────────────────────────────────────────────────────────────────────────────
# isclose — exact 0 / 1, float32 output
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rtol, atol", [(1e-05, 1e-08), (0.0, 0.0), (0.0, 5.0), (1e-3, 0.0)])
def test_int32_isclose(device, rtol, atol):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | 0 / 1 result, returned as float32. The operands |
    |             |       | are converted to float32 before the compare, so |
    |             |       | the golden is torch.isclose on float32 operands |
    |             |       | (torch on int32 operands agrees except at zero  |
    |             |       | tolerance, where distinct values above 2^24     |
    |             |       | that round to one float32 compare close here)   |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn.isclose, input_a, input_b, rtol=rtol, atol=atol)
    assert result.dtype == torch.float32, f"expected a float32 result, got {result.dtype}"
    golden = torch.isclose(input_a.float(), input_b.float(), rtol=rtol, atol=atol)
    assert_equal(golden.to(torch.float32), result)


# ─────────────────────────────────────────────────────────────────────────────
# logical_and, logical_or, logical_xor — exact 0 / 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", LOGICAL_OPS + LOGICAL_INPLACE_OPS)
def test_int32_logical_ops(device, ttnn_op):
    """Pairwise coverage over the int32 binary grid without INT32_MIN.

    | Input range   | Check | Reason                                        |
    |---------------|-------|-----------------------------------------------|
    | no INT32_MIN  | exact | 0 / 1 result; any nonzero operand is True     |
    | INT32_MIN     | n/a   | covered by test_int32_logical_ops_int_min     |
    """
    input_a, input_b = int32_pairwise_inputs(include_int_min=False)

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, input_b), result)


@pytest.mark.parametrize("scalar", LOGICAL_SCALARS)
@pytest.mark.parametrize("ttnn_op", LOGICAL_OPS)
def test_int32_logical_ops_scalar(device, ttnn_op, scalar):
    """The grid without INT32_MIN against an int scalar.

    | Input range   | Check | Reason                                        |
    |---------------|-------|-----------------------------------------------|
    | no INT32_MIN  | exact | 0 / 1 result                                  |
    """
    input_a = scalar_rows(generate_int32_binary_grid(include_int_min=False))

    result = run_int32_binary(device, ttnn_op, input_a, scalar)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, scalar), result)


@pytest.mark.xfail(condition=not is_blackhole(), strict=True, reason=INT_MIN_WRONG_REASON)
@pytest.mark.parametrize("ttnn_op", LOGICAL_OPS)
def test_int32_logical_ops_int_min(device, ttnn_op):
    """INT32_MIN against 0, ±1 and itself, as a tensor operand on either side.

    | Input     | Check | Reason                                             |
    |-----------|-------|----------------------------------------------------|
    | INT32_MIN | exact | INT32_MIN is nonzero, so True. On Wormhole it is   |
    |           |       | read as False: logical_and(INT32_MIN, 1) is 0 and  |
    |           |       | logical_or(INT32_MIN, 0) is 0                      |
    """
    values = torch.tensor([INT32_MIN, 0, 1, -1], dtype=torch.int32)
    input_a, input_b = torch.meshgrid(values, values, indexing="ij")
    input_a = scalar_rows(input_a.flatten())
    input_b = scalar_rows(input_b.flatten())

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert_equal(_golden(ttnn_op, input_a, input_b), result)


@pytest.mark.xfail(condition=not is_blackhole(), strict=True, reason=INT_MIN_WRONG_REASON)
@pytest.mark.parametrize("ttnn_op", LOGICAL_OPS)
def test_int32_logical_ops_int_min_scalar(device, ttnn_op):
    """The grid without INT32_MIN against the scalar INT32_MIN.

    | Input             | Check | Reason                                     |
    |-------------------|-------|--------------------------------------------|
    | scalar INT32_MIN  | exact | True; on Wormhole read as False, so        |
    |                   |       | logical_and is 0 for every element         |
    """
    input_a = scalar_rows(generate_int32_binary_grid(include_int_min=False))

    result = run_int32_binary(device, ttnn_op, input_a, INT32_MIN)
    assert_equal(_golden(ttnn_op, input_a, INT32_MIN), result)
