# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_add(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise addition
    output = ttnn.add(tensor1, tensor2)
    logger.info(f"Addition result: {output}")


def test_subtract(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise subtraction
    output = ttnn.subtract(tensor1, tensor2)
    logger.info(f"Subtraction result: {output}")


def test_multiply(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise multiplication
    output = ttnn.multiply(tensor1, tensor2)
    logger.info(f"Multiplication result: {output}")


def test_eq(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise equality
    output = ttnn.eq(tensor1, tensor2)
    logger.info(f"Element-wise equality: {output}")


def test_ne(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise inequality
    output = ttnn.ne(tensor1, tensor2)
    logger.info(f"Element-wise inequality: {output}")


def test_lt(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise less than comparison
    output = ttnn.lt(tensor1, tensor2)
    logger.info(f"Less than comparison: {output}")


def test_le(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise less than or equal comparison
    output = ttnn.le(tensor1, tensor2)
    logger.info(f"Less than or equal comparison: {output}")


def test_gt(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise greater than comparison
    output = ttnn.gt(tensor1, tensor2)
    logger.info(f"Greater than comparison: {output}")


def test_ge(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Check element-wise greater than or equal comparison
    output = ttnn.ge(tensor1, tensor2)
    logger.info(f"Greater than or equal comparison: {output}")


def test_logical_and(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise logical AND
    output = ttnn.logical_and(tensor1, tensor2)
    logger.info(f"Logical AND result: {output}")


def test_logical_or(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise logical OR
    output = ttnn.logical_or(tensor1, tensor2)
    logger.info(f"Logical OR result: {output}")


def test_logical_xor(device):
    # Create two tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise logical XOR
    output = ttnn.logical_xor(tensor1, tensor2)
    logger.info(f"Logical XOR result: {output}")


def test_ldexp(device):
    # Create two tensors for ldexp operation (x * 2^y)
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute x * 2^y for each element
    output = ttnn.ldexp(tensor1, tensor2)
    logger.info(f"LDEXP result (x * 2^y): {output}")


def test_logaddexp(device):
    # Create two tensors for log-add-exp operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute log(exp(x) + exp(y)) for each element
    output = ttnn.logaddexp(tensor1, tensor2)
    logger.info(f"Log-add-exp result: {output}")


def test_logaddexp2(device):
    # Create two tensors for log-add-exp2 operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute log2(2^x + 2^y) for each element
    output = ttnn.logaddexp2(tensor1, tensor2)
    logger.info(f"Log-add-exp2 result: {output}")


def test_squared_difference(device):
    # Create two tensors for squared difference
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute (x - y)^2 for each element
    output = ttnn.squared_difference(tensor1, tensor2)
    logger.info(f"Squared difference result: {output}")


def test_bias_gelu(device):
    # Create two tensors for bias GELU activation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply GELU activation with bias (GELU(x + bias))
    output = ttnn.bias_gelu(tensor1, tensor2)
    logger.info(f"Bias GELU result: {output}")


def test_divide(device):
    # Create two tensors for division
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise division with fast approximation mode
    output = ttnn.divide(tensor1, tensor2, fast_and_approximate_mode=True)
    logger.info(f"Division (tensor-tensor) result: {output}")

    # Create tensor and scalar for division
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 3

    # Perform division by a scalar value
    output = ttnn.divide(tensor, scalar, fast_and_approximate_mode=False)
    logger.info(f"Division (tensor-scalar) result: {output}")


def test_xlogy(device):
    # Create two tensors for x*log(y) operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute x * log(y) for each element
    output = ttnn.xlogy(tensor1, tensor2)
    logger.info(f"X*log(Y) result: {output}")


def test_rpow(device):
    # Create tensor for reverse power operation
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    exponent = 3

    # Compute exponent^tensor for each element
    output = ttnn.rpow(tensor, exponent)
    logger.info(f"Reverse power (3^tensor) result: {output}")


def test_rsub(device):
    # Create two tensors for reverse subtraction
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute tensor2 - tensor1 for each element
    output = ttnn.rsub(tensor1, tensor2)
    logger.info(f"Reverse subtraction result: {output}")


def test_bitwise_and(device):
    # Create two integer tensors for bitwise AND
    tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Perform bitwise AND operation
    output = ttnn.bitwise_and(tensor1, tensor2)
    logger.info(f"Bitwise AND result: {output}")


def test_bitwise_or(device):
    # Create two integer tensors for bitwise OR
    tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Perform bitwise OR operation
    output = ttnn.bitwise_or(tensor1, tensor2)
    logger.info(f"Bitwise OR result: {output}")


def test_bitwise_xor(device):
    # Create two integer tensors for bitwise XOR
    tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Perform bitwise XOR operation
    output = ttnn.bitwise_xor(tensor1, tensor2)
    logger.info(f"Bitwise XOR result: {output}")


def test_hypot(device):
    # Create two tensors for hypotenuse calculation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute sqrt(x^2 + y^2) for each element
    output = ttnn.hypot(tensor1, tensor2)
    logger.info(f"Hypotenuse calculation result: {output}")


def test_nextafter(device):
    # Create two tensors for next representable value calculation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Find next representable floating-point value after x toward y
    output = ttnn.nextafter(tensor1, tensor2)
    logger.info(f"Next representable value result: {output}")


def test_minimum(device):
    # Create two tensors for element-wise minimum
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Find minimum value between corresponding elements
    output = ttnn.minimum(tensor1, tensor2)
    logger.info(f"Element-wise minimum result: {output}")


def test_atan2(device):
    # Create two tensors for atan2 calculation (y, x format)
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute arctangent of y/x with proper quadrant handling
    output = ttnn.atan2(tensor1, tensor2)
    logger.info(f"Atan2 result: {output}")


def test_gcd(device):
    # Create two integer tensors for greatest common divisor
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute greatest common divisor for each pair of elements
    output = ttnn.gcd(tensor1, tensor2)
    logger.info(f"Greatest common divisor result: {output}")


def test_lcm(device):
    # Create two integer tensors for least common multiple
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute least common multiple for each pair of elements
    output = ttnn.lcm(tensor1, tensor2)
    logger.info(f"Least common multiple result: {output}")


def test_addalpha(device):
    # Create two tensors and alpha parameter for scaled addition
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    alpha = 1.0

    # Compute x + alpha * y for each element
    output = ttnn.addalpha(tensor1, tensor2, alpha)
    logger.info(f"Add-alpha result: {output}")


def test_subalpha(device):
    # Create two tensors and alpha parameter for scaled subtraction
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    alpha = 1.0

    # Compute x - alpha * y for each element
    output = ttnn.subalpha(tensor1, tensor2, alpha)
    logger.info(f"Sub-alpha result: {output}")


def test_isclose(device):
    # Create two tensors for closeness comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    rtol = 1e-4
    atol = 1e-5
    equal_nan = False

    # Check if tensors are element-wise close within tolerances
    output = ttnn.isclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan)
    logger.info(f"Element-wise closeness result: {output}")


def test_div(device):
    # Create two tensors for division
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn.div(tensor1, tensor2, fast_and_approximate_mode=True, round_mode=None)
    logger.info(f"Division result: {output}")

    # Create tensor and scalar for division
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 3
    output = ttnn.div(tensor, scalar, round_mode="floor")
    logger.info(f"Division (tensor-scalar) result: {output}")


def test_div_no_nan(device):
    # Create two tensors for division without NaN
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform division that returns 0 instead of NaN
    output = ttnn.div_no_nan(tensor1, tensor2)
    logger.info(f"Division (no NaN) result: {output}")


def test_floor_div(device):
    # Create two tensors for floor division
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform element-wise floor division
    output = ttnn.floor_div(tensor1, tensor2)
    logger.info(f"Floor division result: {output}")


def test_maximum(device):
    # Create two tensors for element-wise maximum
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Find maximum value between corresponding elements
    output = ttnn.maximum(tensor1, tensor2)
    logger.info(f"Element-wise maximum result: {output}")


def test_outer(device):
    # Create two tensors for outer product
    tensor1 = ttnn.from_torch(torch.rand([1, 1, 32, 1], dtype=torch.bfloat16), device=device)
    tensor2 = ttnn.from_torch(torch.rand([1, 1, 1, 32], dtype=torch.bfloat16), device=device)

    # Compute outer product of the tensors
    output = ttnn.outer(tensor1, tensor2)
    logger.info(f"Outer product result: {output}")


def test_polyval(device):
    # Create input tensor
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    coeffs = [1, 2, 3, 4]

    # Evaluate polynomial at each tensor element
    output = ttnn.polyval(tensor, coeffs)
    logger.info(f"Polynomial evaluation result: {output}")


def test_scatter(device):
    # Create input, index, and source tensors
    input_torch = torch.randn([10, 20, 30, 20, 10], dtype=torch.float32)
    index_torch = torch.randint(0, 10, [10, 20, 30, 20, 5], dtype=torch.int64)
    source_torch = torch.randn([10, 20, 30, 20, 10], dtype=input_torch.dtype)

    input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    dim = -1

    # Perform scatter operation
    output = ttnn.scatter(input_ttnn, dim, index_ttnn, source_ttnn)
    logger.info(f"Scatter operation result: {output}")


def test_fmod(device):
    # Create two tensors for floating-point modulo
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute floating-point remainder of division
    output = ttnn.fmod(tensor1, tensor2)
    logger.info(f"Floating-point modulo result: {output}")


def test_remainder(device):
    # Create two tensors for remainder operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute remainder of division
    output = ttnn.remainder(tensor1, tensor2)
    logger.info(f"Remainder operation result: {output}")


def test_pow(device):
    # Create tensor and integer exponent
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    exponent = 2
    output = ttnn.pow(tensor, exponent)
    logger.info(f"Power with integer exponent result: {output}")

    # Create tensor and float exponent
    exponent = 2.5
    output = ttnn.pow(tensor, exponent)
    logger.info(f"Power with float exponent result: {output}")

    # Create two tensors for exponentiation
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.pow(tensor1, tensor2)
    logger.info(f"Power with tensor exponent result: {output}")


# Inplace operations
def test_add_(device):
    # Create two tensors for inplace addition
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.add_(tensor1, tensor2)
    logger.info("Inplace addition completed")


def test_subtract_(device):
    # Create two tensors for inplace subtraction
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.subtract_(tensor1, tensor2)
    logger.info("Inplace subtraction completed")


def test_multiply_(device):
    # Create two tensors for inplace multiplication
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.multiply_(tensor1, tensor2)
    logger.info("Inplace multiplication completed")


def test_gt_(device):
    # Create two tensors for inplace greater than comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.gt_(tensor1, tensor2)
    logger.info("Inplace greater than completed")


def test_ge_(device):
    # Create two tensors for inplace greater/equal comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.ge_(tensor1, tensor2)
    logger.info("Inplace greater/equal completed")


def test_lt_(device):
    # Create two tensors for inplace less than comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.lt_(tensor1, tensor2)
    logger.info("Inplace less than completed")


def test_le_(device):
    # Create two tensors for inplace less/equal comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.le_(tensor1, tensor2)
    logger.info("Inplace less/equal completed")


def test_eq_(device):
    # Create two tensors for inplace equality comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.eq_(tensor1, tensor2)
    logger.info("Inplace equality completed")


def test_ne_(device):
    # Create two tensors for inplace inequality comparison
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.ne_(tensor1, tensor2)
    logger.info("Inplace inequality completed")


def test_ldexp_(device):
    # Create two tensors for inplace ldexp operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.ldexp_(tensor1, tensor2)
    logger.info("Inplace ldexp completed")


def test_logaddexp_(device):
    # Create two tensors for inplace log-add-exp
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.logaddexp_(tensor1, tensor2)
    logger.info("Inplace log-add-exp completed")


def test_logaddexp2_(device):
    # Create two tensors for inplace log-add-exp2
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.logaddexp2_(tensor1, tensor2)
    logger.info("Inplace log-add-exp2 completed")


def test_squared_difference_(device):
    # Create two tensors for inplace squared difference
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.squared_difference_(tensor1, tensor2)
    logger.info("Inplace squared difference completed")


def test_divide_(device):
    # Create two tensors for inplace division
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.divide_(tensor1, tensor2, fast_and_approximate_mode=True)
    logger.info("Inplace division completed")


def test_rsub_(device):
    # Create two tensors for inplace reverse subtraction
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.rsub_(tensor1, tensor2)
    logger.info("Inplace reverse subtraction completed")


def test_bias_gelu_(device):
    # Create two tensors for inplace bias GELU
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.bias_gelu_(tensor1, tensor2)
    logger.info("Inplace bias GELU completed")


def test_logical_or_(device):
    # Create two tensors for inplace logical OR
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.logical_or_(tensor1, tensor2)
    logger.info("Inplace logical OR completed")


def test_logical_xor_(device):
    # Create two tensors for inplace logical XOR
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.logical_xor_(tensor1, tensor2)
    logger.info("Inplace logical XOR completed")


def test_logical_and_(device):
    # Create two tensors for inplace logical AND
    tensor1 = ttnn.from_torch(
        torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform inplace operation
    ttnn.logical_and_(tensor1, tensor2)
    logger.info("Inplace logical AND completed")


# Backward operations
def test_add_bw(device):
    # Create gradient and input tensors for addition backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for addition operation
    output = ttnn.add_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Addition backward result: {output}")


def test_atan2_bw(device):
    # Create gradient and input tensors for atan2 backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for atan2 operation
    output = ttnn.atan2_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Atan2 backward result: {output}")


def test_bias_gelu_bw(device):
    # Create gradient and input tensors for bias GELU backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    approximate = "none"

    # Compute gradients for bias GELU operation
    output = ttnn.bias_gelu_bw(grad_tensor, tensor1, tensor2, approximate=approximate)
    logger.info(f"Bias GELU backward result: {output}")


def test_div_bw(device):
    # Create gradient and input tensors for division backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.div_bw(grad_tensor, tensor1, tensor2, round_mode=None)
    logger.info(f"Division backward result: {output}")

    # Create gradient and input tensors for division backward with tensor-scalar
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 2
    output = ttnn.div_bw(grad_tensor, tensor, scalar, round_mode=None)
    logger.info(f"Division backward with tensor-scalar result: {output}")


def test_embedding_bw(device):
    # Create input, weights, and gradient tensors for embedding backward
    batch_size, seq_len, embedding_dim, num_embeddings = 2, 1024, 4096, 3200
    input_shape = (batch_size, seq_len)
    input_index = torch.randint(0, num_embeddings, input_shape)
    input_tensor = ttnn.from_torch(input_index, dtype=ttnn.uint32, device=device)

    # Create weights tensor
    weights_shape = (num_embeddings, embedding_dim)
    weights = torch.randn(weights_shape, requires_grad=True)
    weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Create gradient tensor
    grad_shape = (1, 1, batch_size * seq_len, embedding_dim)
    grad_data = torch.randn(grad_shape, requires_grad=True)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute gradients for embedding operation
    output = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=ttnn.bfloat16)
    logger.info(f"Embedding backward result: {output}")


def test_fmod_bw(device):
    # Create gradient and input tensors for fmod backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.fmod_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Fmod backward result: {output}")

    # Create gradient and input tensors for fmod backward with tensor-scalar
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 2
    output = ttnn.fmod_bw(grad_tensor, tensor1, scalar)
    logger.info(f"Fmod backward with tensor-scalar result: {output}")


def test_remainder_bw(device):
    # Create gradient and input tensors for remainder backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.remainder_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Remainder backward result: {output}")

    # Create gradient and input tensors for remainder backward with tensor-scalar
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 2
    output = ttnn.remainder_bw(grad_tensor, tensor1, scalar)
    logger.info(f"Remainder backward with tensor-scalar result: {output}")


def test_addalpha_bw(device):
    # Create gradient and input tensors for addalpha backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    alpha = 1.0

    # Compute gradients for addalpha operation
    output = ttnn.addalpha_bw(grad_tensor, tensor1, tensor2, alpha)
    logger.info(f"Addalpha backward result: {output}")


def test_subalpha_bw(device):
    # Create gradient and input tensors for subalpha backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    alpha = 1

    # Compute gradients for subalpha operation
    output = ttnn.subalpha_bw(grad_tensor, tensor1, tensor2, alpha)
    logger.info(f"Subalpha backward result: {output}")


def test_xlogy_bw(device):
    # Create gradient and input tensors for xlogy backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for x*log(y) operation
    output = ttnn.xlogy_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"X*log(y) backward result: {output}")


def test_hypot_bw(device):
    # Create gradient and input tensors for hypot backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for hypotenuse operation
    output = ttnn.hypot_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Hypot backward result: {output}")


def test_ldexp_bw(device):
    # Create gradient and input tensors for ldexp backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for ldexp operation
    output = ttnn.ldexp_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Ldexp backward result: {output}")


def test_logaddexp_bw(device):
    # Create gradient and input tensors for logaddexp backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for log-add-exp operation
    output = ttnn.logaddexp_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Log-add-exp backward result: {output}")


def test_logaddexp2_bw(device):
    # Create gradient and input tensors for logaddexp2 backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for log-add-exp2 operation
    output = ttnn.logaddexp2_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Log-add-exp2 backward result: {output}")


def test_mul_bw(device):
    # Create gradient and input tensors for multiplication backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for multiplication operation
    output = ttnn.mul_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Multiplication backward result: {output}")

    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 4
    output = ttnn.mul_bw(grad_tensor, tensor1, scalar)
    logger.info(f"Multiplication backward with scalar result: {output}")


def test_squared_difference_bw(device):
    # Create gradient and input tensors for squared difference backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for squared difference operation
    output = ttnn.squared_difference_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Squared difference backward result: {output}")


def test_concat_bw(device):
    # Create gradient and input tensors for concatenation backward
    grad_tensor = ttnn.from_torch(
        torch.rand([14, 1, 30, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.rand([12, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.rand([2, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    dim = 0

    # Compute gradients for concatenation operation
    output = ttnn.concat_bw(grad_tensor, tensor1, tensor2, dim)
    logger.info(f"Concatenation backward result: {output}")


def test_rsub_bw(device):
    # Create gradient and input tensors for reverse subtraction backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for reverse subtraction operation
    output = ttnn.rsub_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Reverse subtraction backward result: {output}")


def test_sub_bw(device):
    # Create gradient and input tensors for subtraction backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for subtraction operation
    output = ttnn.sub_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Subtraction backward result: {output}")

    # Compute gradients for subtraction with scalar
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 2

    # Compute gradients for subtraction operation with scalar
    output = ttnn.sub_bw(grad_tensor, tensor1, scalar)
    logger.info(f"Subtraction backward with scalar result: {output}")


def test_min_bw(device):
    # Create gradient and input tensors for minimum backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for minimum operation
    output = ttnn.min_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Minimum backward result: {output}")


def test_max_bw(device):
    # Create gradient and input tensors for maximum backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute gradients for maximum operation
    output = ttnn.max_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Maximum backward result: {output}")


def test_assign_bw(device):
    # Create gradient and input tensors for assignment backward
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.assign_bw(grad_tensor, tensor1)
    logger.info(f"Assignment backward result: {output}")

    # Create second input tensor for assignment backward
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.assign_bw(grad_tensor, tensor1, tensor2)
    logger.info(f"Assignment backward with two inputs result: {output}")
