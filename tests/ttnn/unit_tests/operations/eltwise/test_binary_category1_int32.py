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
    wrap_int32,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the binary arithmetic ops that accept ttnn.int32
input: the int32 subset of category 1 (basic_binary_arithmetic) and
category 4 (composite binary math) of the float32 and bfloat16 files.

Tensor-tensor inputs come from int32_pairwise_inputs: the outer product of
the 2048-value int32 binary grid (every integer in [-256, 256), ±2^k and
±(2^k ± 1) for every k including 2^24 + 1, INT32_MIN and INT32_MAX,
log-uniform and uniform random values), 4,194,304 pairs.

Int32 category 1: arithmetic
(float32 / bfloat16 category 1, and the int32 ops of category 4)

| #  | Op                           | Computes                 | Output |
|----|------------------------------|--------------------------|--------|
| 1  | ttnn.add / add_              | a + b                    | int32  |
| 2  | ttnn.subtract / subtract_    | a - b                    | int32  |
| 3  | ttnn.rsub / rsub_            | b - a                    | int32  |
| 4  | ttnn.multiply / multiply_    | a * b                    | int32  |
| 5  | ttnn.squared_difference / _  | (a - b)**2               | int32  |
| 6  | ttnn.prelu (Tensor weight)   | a if a >= 0 else a * w   | int32  |

addalpha and subalpha reject int32 input (their alpha step is
MUL_UNARY_SFPU, refused in UnaryDeviceOperation validation), as does prelu
with a scalar weight (PRELU_SFPU). ldexp, logaddexp, logaddexp2, xlogy,
hypot and bias_gelu are float-only in BinaryNgDeviceOperation; bias_gelu with
a scalar fails on GELU in unary validation.

Every result wraps mod 2^32, as int32 arithmetic in torch does, so the
goldens are computed in int64 and wrapped with wrap_int32.

The tensor-scalar forms of add, subtract, rsub, multiply and
squared_difference run on BinaryNgDeviceOperation with the scalar packed as
an int32, so they are swept here; the int32 unary category 2 file covers the
tensor-scalar ops that lower to UnaryDeviceOperation.

Accuracy criteria

| Op                      | Input range            | Check        |
|-------------------------|------------------------|--------------|
| add, subtract, rsub,    | full grid, and every   | exact        |
| multiply,               | scalar in SCALARS      |              |
| squared_difference      |                        |              |
| prelu                   | x ≠ INT32_MIN          | exact        |
|   on Wormhole           | x = INT32_MIN          | strict xfail |
"""

SCALARS = [0, 1, -1, 7, -100, 2**24 + 1, -(2**24 + 1), 123456789, INT32_MIN, INT32_MAX]
PRELU_WEIGHTS = [0, 1, -1, 2, -3, 7, 100, -100, 2**16, 2**24 + 1, INT32_MIN, INT32_MAX]

ARITHMETIC_GOLDENS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "rsub": lambda a, b: b - a,
    "multiply": lambda a, b: a * b,
    "squared_difference": lambda a, b: (a - b) ** 2,
}
ARITHMETIC_OPS = [ttnn.add, ttnn.subtract, ttnn.rsub, ttnn.multiply, ttnn.squared_difference]
ARITHMETIC_INPLACE_OPS = [ttnn.add_, ttnn.subtract_, ttnn.rsub_, ttnn.multiply_, ttnn.squared_difference_]


def _golden(ttnn_op, input_a, input_b):
    b = input_b.to(torch.int64) if torch.is_tensor(input_b) else input_b
    name = ttnn_op.__name__.split(".")[-1].rstrip("_")
    return wrap_int32(ARITHMETIC_GOLDENS[name](input_a.to(torch.int64), b))


# ─────────────────────────────────────────────────────────────────────────────
# add, subtract, rsub, multiply, squared_difference — exact, wrapping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", ARITHMETIC_OPS + ARITHMETIC_INPLACE_OPS)
def test_int32_arithmetic_ops(device, ttnn_op):
    """Pairwise coverage over the int32 binary grid.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | integer result; sums, differences and products  |
    |             |       | past the int32 range wrap mod 2^32, as in torch |
    """
    input_a, input_b = int32_pairwise_inputs()

    result = run_int32_binary(device, ttnn_op, input_a, input_b)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, input_b), result)


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("ttnn_op", ARITHMETIC_OPS)
def test_int32_arithmetic_ops_scalar(device, ttnn_op, scalar):
    """The grid against an int scalar, each value repeated across one tile width.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | full grid   | exact | the scalar reaches the kernel as an int32, so   |
    |             |       | ±(2^24 + 1) and the int32 limits are not        |
    |             |       | rounded through float32                         |
    """
    input_a = scalar_rows(generate_int32_binary_grid())

    result = run_int32_binary(device, ttnn_op, input_a, scalar)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_golden(ttnn_op, input_a, scalar), result)


# ─────────────────────────────────────────────────────────────────────────────
# prelu (per-channel Tensor weight) — exact
# ─────────────────────────────────────────────────────────────────────────────


def _prelu_inputs(include_int_min):
    """(1, C, 64, 64) grid values, one channel per weight in PRELU_WEIGHTS."""
    values = generate_int32_binary_grid(include_int_min=include_int_min)
    channel = len(PRELU_WEIGHTS)
    repeats = -(-channel * 64 * 64 // values.numel())
    input_tensor = values.repeat(repeats)[: channel * 64 * 64].reshape(1, channel, 64, 64)
    weight = torch.tensor(PRELU_WEIGHTS, dtype=torch.int32)
    return input_tensor, weight


def _prelu_golden(input_tensor, weight):
    x = input_tensor.to(torch.int64)
    w = weight.to(torch.int64).reshape(1, -1, 1, 1)
    return wrap_int32(torch.where(x >= 0, x, x * w))


def test_int32_prelu(device):
    """prelu with one int32 weight per channel.

    | Input range   | Check | Reason                                        |
    |---------------|-------|-----------------------------------------------|
    | x ≠ INT32_MIN | exact | a select of x or the wrapping product x * w   |
    | INT32_MIN     | n/a   | covered by test_int32_prelu_int_min           |
    """
    input_tensor, weight = _prelu_inputs(include_int_min=False)

    result = run_int32_binary(device, ttnn.prelu, input_tensor, weight)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_prelu_golden(input_tensor, weight), result)


@pytest.mark.xfail(
    condition=not is_blackhole(),
    strict=True,
    reason="#60985: prelu selects with ltz, and int32 ltz reads INT32_MIN (0x80000000) as zero on Wormhole",
)
def test_int32_prelu_int_min(device):
    """prelu at x = INT32_MIN.

    | Input     | Check | Reason                                             |
    |-----------|-------|----------------------------------------------------|
    | INT32_MIN | exact | negative, so the result is INT32_MIN * w: 0 for an |
    |           |       | even w. On Wormhole ltz(INT32_MIN) is 0, so x is   |
    |           |       | returned unchanged for every weight                |
    """
    input_tensor, weight = _prelu_inputs(include_int_min=True)

    result = run_int32_binary(device, ttnn.prelu, input_tensor, weight)
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(_prelu_golden(input_tensor, weight), result)
