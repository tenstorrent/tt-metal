# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_int32_values,
    to_int32_tt_tensor,
    INT32_MIN,
)

pytestmark = pytest.mark.use_module_device


"""
Int32 accuracy sweep of the unary ops without parameters that accept
ttnn.int32 input: the int32 subset of categories 1 (basic math) and 2
(activations) of the float32 and bfloat16 files.

Inputs come from generate_int32_values: every integer in [-2^14, 2^14),
±2^k and ±(2^k ± 1) for every k (including 2^24 + 1, INT32_MIN and
INT32_MAX), and uniformly random int32 values, 65,536 in total.

Int32 category 1: ops without parameters
(float32 / bfloat16 categories 1, basic_unary_math, and 2, basic_unary_activation)

| #  | Op               | Formula                   | Output        |
|----|------------------|---------------------------|---------------|
| 1  | ttnn.abs         | |x|                       | int32         |
| 2  | ttnn.neg         | -x                        | int32         |
| 3  | ttnn.square      | x * x                     | int32         |
| 4  | ttnn.bitwise_not | ~x                        | int32         |
| 5  | ttnn.identity    | x                         | int32         |
| 6  | ttnn.relu        | max(x, 0)                 | int32         |
| 7  | ttnn.relu6       | min(max(x, 0), 6)         | int32         |
| 8  | ttnn.signbit     | x < 0                     | int32 (0 / 1) |
| 9  | ttnn.eqz         | x == 0                    | int32 (0 / 1) |
| 10 | ttnn.nez         | x != 0                    | int32 (0 / 1) |
| 11 | ttnn.gtz         | x > 0                     | int32 (0 / 1) |
| 12 | ttnn.ltz         | x < 0                     | int32 (0 / 1) |
| 13 | ttnn.gez         | x >= 0                    | int32 (0 / 1) |
| 14 | ttnn.lez         | x <= 0                    | int32 (0 / 1) |
| 15 | ttnn.logical_not | x == 0                    | int32 (0 / 1) |

Every other op in these categories rejects int32 input in
UnaryDeviceOperation validation (unary_op_supports_integer_dtype). That
includes sign, floor, ceil, trunc and round, which torch defines for
integers.

Accuracy criteria

| Op                     | Input range   | Check        | Reason                                    |
|------------------------|---------------|--------------|-------------------------------------------|
| abs, neg, square,      | all           | exact        | integer result; abs and neg of INT32_MIN  |
| bitwise_not, identity, |               |              | and every overflowing square wrap mod     |
| relu, relu6            |               |              | 2^32, as in torch                         |
| signbit, logical_not   | all           | exact        | 0 / 1 result                              |
| eqz, nez, ltz, gez     | all           | exact        | 0 / 1 result (Blackhole)                  |
|   on Wormhole          | x ≠ INT32_MIN | exact        | 0 / 1 result                              |
|   on Wormhole          | INT32_MIN     | strict xfail | INT32_MIN is read as zero: eqz and gez    |
|                        |               |              | return 1, nez and ltz return 0            |
| gtz, lez               | all           | exact        | 0 / 1 result (Wormhole)                   |
|   on Blackhole         | x ≠ INT32_MIN | exact        | 0 / 1 result                              |
|   on Blackhole         | INT32_MIN     | strict xfail | INT32_MIN is read as positive: gtz        |
|                        |               |              | returns 1, lez returns 0                  |
"""


# ─────────────────────────────────────────────────────────────────────────────
# abs, neg, square, bitwise_not, identity, relu, relu6 — exact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [ttnn.abs, ttnn.neg, ttnn.square, ttnn.bitwise_not, ttnn.identity, ttnn.relu, ttnn.relu6],
)
def test_int32_value_ops(device, ttnn_op):
    """Ops that return an int32 value.

    | Input range | Check | Reason                                           |
    |-------------|-------|--------------------------------------------------|
    | all         | exact | abs(INT32_MIN) and neg(INT32_MIN) are INT32_MIN, |
    |             |       | and square wraps mod 2^32 past |x| = 46340       |
    """
    input_tensor = generate_int32_values()

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    result = ttnn.to_torch(ttnn_op(tt_in))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# signbit, comparison to zero, logical_not — exact 0 / 1
# ─────────────────────────────────────────────────────────────────────────────

INT_MIN_WRONG = [ttnn.gtz, ttnn.lez] if is_blackhole() else [ttnn.eqz, ttnn.nez, ttnn.ltz, ttnn.gez]
INT_MIN_WRONG_REASON = (
    "#58138: int32 gtz and lez read INT32_MIN (0x80000000) as positive"
    if is_blackhole()
    else "#60985: int32 eqz, nez, ltz and gez read INT32_MIN (0x80000000) as zero"
)


@pytest.mark.parametrize(
    "ttnn_op",
    [ttnn.signbit, ttnn.eqz, ttnn.nez, ttnn.gtz, ttnn.ltz, ttnn.gez, ttnn.lez, ttnn.logical_not],
)
def test_int32_predicate_ops(device, ttnn_op):
    """Ops that return 0 or 1 as int32.

    | Input range     | Check | Reason                                       |
    |-----------------|-------|----------------------------------------------|
    | all             | exact | 0 / 1 result; torch returns bool, compared   |
    |                 |       | as int32                                     |
    | INT32_MIN       | n/a   | left out for eqz, nez, ltz and gez on WH and |
    |                 |       | for gtz and lez on BH; covered by            |
    |                 |       | test_int32_predicate_int_min                 |
    """
    input_tensor = generate_int32_values(include_int_min=ttnn_op not in INT_MIN_WRONG)

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device).to(torch.int32)

    result = ttnn.to_torch(ttnn_op(tt_in))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden, result)


@pytest.mark.xfail(strict=True, reason=INT_MIN_WRONG_REASON)
@pytest.mark.parametrize("ttnn_op", INT_MIN_WRONG)
def test_int32_predicate_int_min(device, ttnn_op):
    """The comparisons to zero that are wrong at INT32_MIN.

    | Input     | Check | Reason                                             |
    |-----------|-------|----------------------------------------------------|
    | INT32_MIN | exact | INT32_MIN is negative and non-zero; on WH the      |
    |           |       | device returns eqz = 1, nez = 0, ltz = 0 and       |
    |           |       | gez = 1, on BH gtz = 1 and lez = 0                 |
    | 0, ±1     | exact | control                                            |

    signbit and logical_not are correct at INT32_MIN on both; gtz and lez are
    correct on WH, eqz, nez, ltz and gez on BH.
    """
    input_tensor = torch.zeros(32, 32, dtype=torch.int32)
    input_tensor[:16] = INT32_MIN
    input_tensor[16:20] = 1
    input_tensor[20:24] = -1

    tt_in = to_int32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device).to(torch.int32)

    result = ttnn.to_torch(ttnn_op(tt_in))
    assert result.dtype == torch.int32, f"expected an int32 result, got {result.dtype}"
    assert_equal(golden[16:], result[16:])
    assert_equal(golden[:16], result[:16])
