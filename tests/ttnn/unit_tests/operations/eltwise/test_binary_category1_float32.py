# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, flush_subnormal_values_to_zero
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    binary_grid_values,
    pairwise_from_values,
    pairwise_inputs,
    run_binary,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: basic_binary_arithmetic + corresponding inplace op

| #  | Op              | Computes                  | ULP threshold (float32) |
|----|-----------------|---------------------------|-------------------------|
| 1  | ttnn.add        | a + b                     | 0                       |
| 2  | ttnn.sub        | a - b                     | 0                       |
| 3  | ttnn.multiply   | a * b                     | 0                       |
| 4  | ttnn.divide     | a / b                     | 1                       |
| 5  | ttnn.add_       | a + b (inplace)           | 0                       |
| 6  | ttnn.sub_       | a - b (inplace)           | 0                       |
| 7  | ttnn.multiply_  | a * b (inplace)           | 0                       |
| 8  | ttnn.divide_    | a / b (inplace)           | 1                       |
| 9  | ttnn.rsub       | b - a                     | 0                       |
| 10 | ttnn.rsub_      | b - a (inplace)           | 0                       |
| 11 | ttnn.addalpha   | a + alpha * b             | 0                       |
| 12 | ttnn.subalpha   | a - alpha * b             | 0                       |
"""


@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.sub, ttnn.rsub, ttnn.add_, ttnn.sub_, ttnn.rsub_])
def test_addlike_ops(device, ttnn_op):
    """Pairwise coverage of add/sub/rsub over the stratified grid at float32.

    | Case                    | float32 behavior                               | Handling        |
    |-------------------------|------------------------------------------------|-----------------|
    | Rounding                | SFPU rounds to nearest even, matches torch     | 0 ULP           |
    | Subnormal result        | Flushed to 0 on device                         | Flush both      |
    | Sum near bf16 max       | Stays finite (0x7F7F0000 + 0x7AFF0000)         | No mask         |
    | Overflow                | ±inf on both sides                             | allow_nonfinite |
    """
    input_a, input_b = pairwise_inputs(include_zero=True, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)

    # Device flushes subnormal sums to zero.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.multiply, ttnn.multiply_])
def test_multiply(device, ttnn_op):
    """Pairwise coverage of ttnn.multiply over the stratified grid at float32.

    | Case                    | float32 behavior                               | Handling        |
    |-------------------------|------------------------------------------------|-----------------|
    | Rounding                | 8-bit x 8-bit mantissa product is exact        | 0 ULP           |
    | Min-normal 2^{-126}     | Kept (bf16 SFPU flushes it)                    | No mask         |
    | Subnormal result        | Flushed to 0 on device                         | Flush both      |
    | Overflow                | ±inf on both sides                             | allow_nonfinite |
    """
    input_a, input_b = pairwise_inputs(include_zero=True, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.divide, ttnn.divide_])
def test_divide(device, ttnn_op):
    """Pairwise coverage of ttnn.divide over the stratified grid at float32.

    | Case                    | float32 behavior                               | Handling        |
    |-------------------------|------------------------------------------------|-----------------|
    | Rounding                | Within 1 ULP of torch                          | 1 ULP           |
    | Reciprocal flush        | None, even for abs(b) ≥ 2^126                  | No mask         |
    | Overflow-to-zero        | None                                           | No mask         |
    | Subnormal result        | Flushed to 0 on device                         | Flush both      |
    | Overflow                | ±inf on both sides                             | allow_nonfinite |

    1 ULP example: -1.2369e-10 / 3.6238e-12 → torch -34.133335, device -34.133331.
    """
    input_a, input_b = pairwise_inputs(include_zero=False, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# For |alpha| > 1 the scale alpha * b overflows to ±inf on device once it
# exceeds float32 max, while torch computes the sum without the intermediate
# overflow (1.02e31 + 2*(-2^127) → torch -max float32, device -inf).
# Bounding b to 2^126 keeps the scale finite for |alpha| < 4
# (float32 max ≈ 3.9999998 × 2^126).
_ADDALPHA_LARGE_SCALE_BOUND = 2.0**126


@pytest.mark.parametrize("ttnn_op", [ttnn.addalpha, ttnn.subalpha])
@pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.75])
def test_addalpha_subalpha(device, ttnn_op, alpha):
    """Pairwise coverage of ttnn.addalpha / ttnn.subalpha at float32: a ± alpha * b.

    The rhs is scaled by an SFPU mul, then added or subtracted on the SFPU
    (float32 operands do not take the FPU kernel), so the threshold is the
    float32 add threshold (0 ULP).

    | alpha      | Input grid           | Handling                                   |
    |------------|----------------------|--------------------------------------------|
    | 0, ±1      | Full grid            | None                                       |
    | ±0.5       | Full grid            | Expect a where alpha * b underflows        |
    | ±2, 3.75   | [-2^126, 2^126]      | Bounded so alpha * b stays finite          |
    """
    if abs(alpha) > 1:
        values = binary_grid_values(
            low=-_ADDALPHA_LARGE_SCALE_BOUND,
            high=_ADDALPHA_LARGE_SCALE_BOUND,
            include_zero=True,
            dtype=torch.float32,
        )
        input_a, input_b = pairwise_from_values(values)
    else:
        input_a, input_b = pairwise_inputs(include_zero=True, dtype=torch.float32)
    golden, result = run_binary(
        device, ttnn_op, input_a, input_b, golden_kwargs={"alpha": alpha}, input_dtype=ttnn.float32
    )

    # Device flushes subnormal sums to zero, same as add/sub.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # SFPU scale underflow: 0 < |alpha| < 1 can flush alpha*b to 0 while b is
    # still a normal, so the device returns a. Torch keeps the subnormal
    # alpha*b, so its sum differs from a whenever |a| is small:
    #   a = b = 2^-126, alpha = 0.5
    #   torch a + 0.5*b = 1.763e-38; device 0.5*b → 0, then a + 0 = 1.175e-38.
    # The torch sum can also be a subnormal that flushes to 0:
    #   a = -2.342e-38, b = 2.342e-38, alpha = 0.5
    #   torch a + 0.5*b ≈ -1.17e-38 → 0; device returns a.
    # The expected value for these pairs is a, and the device output is still
    # checked against it. The grid holds only normals and zero, so a needs no
    # flush. alpha == 0 needs no mask: alpha*b is exactly 0 in torch too.
    if 0.0 < abs(alpha) < 1:
        scale_underflow = (input_b != 0) & (input_b.abs() * abs(alpha) < 2.0**-126)
        assert scale_underflow.any(), "expected the SFPU scale of b to underflow on this grid"
        golden = torch.where(scale_underflow, input_a, golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)
