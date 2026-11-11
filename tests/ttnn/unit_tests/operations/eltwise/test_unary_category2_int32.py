# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_values,
    to_int32_tt_tensor,
    INT32_MAX,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the unary ops with one scalar parameter that accept
ttnn.int32 input: the int32 subset of category 4 of the float32 and
bfloat16 files, plus the tensor-scalar forms of binary ops that run as a
single UnaryDeviceOperation with the scalar as the parameter.

Inputs come from generate_int32_values: every integer in [-2^14, 2^14),
±2^k and ±(2^k ± 1) for every k (including 2^24 + 1, INT32_MIN and
INT32_MAX), and uniformly random int32 values, 65,536 in total.

Int32 category 2: ops with a scalar parameter
(float32 / bfloat16 category 4, plus the tensor-scalar binary forms above)

| #  | Op                       | Parameter   | Formula                         |
|----|--------------------------|-------------|---------------------------------|
| 1  | ttnn.relu_max            | upper_limit | max(0, min(x, upper_limit))     |
| 2  | ttnn.relu_min            | lower_limit | max(x, lower_limit)             |
| 3  | ttnn.fill                | fill_value  | fill_value (input ignored)      |
| 4  | ttnn.maximum             | scalar      | max(x, scalar)                  |
| 5  | ttnn.minimum             | scalar      | min(x, scalar)                  |
| 6  | ttnn.bitwise_and         | scalar      | x & scalar                      |
| 7  | ttnn.bitwise_or          | scalar      | x | scalar                      |
| 8  | ttnn.bitwise_xor         | scalar      | x ^ scalar                      |
| 9  | ttnn.bitwise_left_shift  | shift       | x << shift (wraps mod 2^32)     |
| 10 | ttnn.bitwise_right_shift | shift       | x >> shift (arithmetic)         |
| 11 | ttnn.logical_left_shift  | shift       | x << shift (wraps mod 2^32)     |

elu, celu, heaviside, leaky_relu, hardshrink, softshrink and rpow reject
int32 input in UnaryDeviceOperation validation. The other tensor-scalar
binary forms (add, subtract, rsub, multiply, the comparisons, remainder,
fmod, logical_right_shift) run on BinaryNgDeviceOperation and belong to
the binary tests.

Every parameter is an int, so it reaches the kernel as an int32 and is not
rounded through float32.

Accuracy criteria

| Op                    | Parameter                      | Input range | Check |
|-----------------------|--------------------------------|-------------|-------|
| relu_max, relu_min,   | 0, ±1, ±6, ±100, ±(2^24 + 1),  | all         | exact |
| fill, maximum,        | 123456789, INT32_MIN,          |             |       |
| minimum               | INT32_MAX                      |             |       |
| bitwise_and, _or,     | 0, -1, 0xFF, 0x0F0F0F0F,       | all         | exact |
| _xor                  | -0xABCD, INT32_MIN, INT32_MAX  |             |       |
| bitwise_left_shift,   | 0, 1, 4, 16, 30, 31, 32        | all         | exact |
| bitwise_right_shift,  |                                |             |       |
| logical_left_shift    |                                |             |       |
"""

SCALARS = [0, 1, -1, 6, -6, 100, -100, 2**24 + 1, -(2**24 + 1), 123456789, INT32_MIN, INT32_MAX]
BITMASKS = [0, -1, 0xFF, 0x0F0F0F0F, -0xABCD, INT32_MIN, INT32_MAX]
SHIFTS = [0, 1, 4, 16, 30, 31, 32]


def _run_scalar_op(device, ttnn_op, scalar):
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, scalar)

    result = ttnn.to_torch(ttnn_op(tt_in, scalar))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# relu_max, relu_min, fill, maximum, minimum — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("ttnn_op", [ttnn.relu_max, ttnn.relu_min, ttnn.fill, ttnn.maximum, ttnn.minimum])
def test_int32_scalar_select_ops(device, ttnn_op, scalar):
    """Ops whose result is x, 0 or the scalar.

    | Input range | Check | Reason                                             |
    |-------------|-------|----------------------------------------------------|
    | all         | exact | no arithmetic; ±(2^24 + 1) and the int32 limits    |
    |             |       | check the scalar is not rounded through float32    |

    relu_max with a negative upper_limit returns 0 everywhere, as its
    formula max(0, min(x, upper_limit)) gives.
    """
    _run_scalar_op(device, ttnn_op, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# bitwise_and, bitwise_or, bitwise_xor — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scalar", BITMASKS)
@pytest.mark.parametrize("ttnn_op", [ttnn.bitwise_and, ttnn.bitwise_or, ttnn.bitwise_xor])
def test_int32_bitwise_scalar_ops(device, ttnn_op, scalar):
    """Bitwise ops with a scalar mask.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | bit operations; the masks set the sign bit, the |
    |             |       | low byte, alternating nibbles and all 32 bits   |
    """
    _run_scalar_op(device, ttnn_op, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# bitwise_left_shift, bitwise_right_shift, logical_left_shift — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("ttnn_op", [ttnn.bitwise_left_shift, ttnn.bitwise_right_shift, ttnn.logical_left_shift])
def test_int32_shift_scalar_ops(device, ttnn_op, shift):
    """Shifts by a scalar amount.

    | Input range | Check | Reason                                           |
    |-------------|-------|--------------------------------------------------|
    | all         | exact | left shifts wrap mod 2^32; right shifts fill     |
    |             |       | with the sign bit; a shift of 32 gives 0 for a   |
    |             |       | left shift and 0 or -1 for a right shift, as in  |
    |             |       | torch                                            |
    """
    _run_scalar_op(device, ttnn_op, shift)
