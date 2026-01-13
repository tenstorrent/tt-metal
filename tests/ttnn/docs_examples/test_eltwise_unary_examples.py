# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_abs(device):
    # Create a tensor
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the absolute value
    output = ttnn.abs(tensor)
    logger.info(f"Absolute value: {output}")


def test_acos(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the arccosine
    output = ttnn.acos(tensor)
    logger.info(f"Arccosine: {output}")


def test_asin(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the arcsine
    output = ttnn.asin(tensor)
    logger.info(f"Arcsine: {output}")


def test_atan(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the arctangent
    output = ttnn.atan(tensor)
    logger.info(f"Arctangent: {output}")


def test_atanh(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the hyperbolic arctangent
    output = ttnn.atanh(tensor)
    logger.info(f"Hyperbolic arctangent: {output}")


def test_cos(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the cosine
    output = ttnn.cos(tensor)
    logger.info(f"Cosine: {output}")


def test_acosh(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the hyperbolic arccosine
    output = ttnn.acosh(tensor)
    logger.info(f"Hyperbolic arccosine: {output}")


def test_erfinv(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the inverse error function
    output = ttnn.erfinv(tensor)
    logger.info(f"Inverse error function: {output}")


def test_exp2(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute 2^x
    output = ttnn.exp2(tensor)
    logger.info(f"2^x: {output}")


def test_expm1(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute e^x - 1
    output = ttnn.expm1(tensor)
    logger.info(f"e^x - 1: {output}")


def test_floor(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the floor (largest integer less than or equal to each element)
    output = ttnn.floor(tensor)
    logger.info(f"Floor: {output}")


def test_trunc(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the truncated value (integer part)
    output = ttnn.trunc(tensor)
    logger.info(f"Truncated: {output}")


def test_frac(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the fractional part
    output = ttnn.frac(tensor)
    logger.info(f"Fractional part: {output}")


def test_eqz(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are equal to zero
    output = ttnn.eqz(tensor)
    logger.info(f"Equal to zero: {output}")


def test_ceil(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the ceiling (smallest integer greater than or equal to each element)
    output = ttnn.ceil(tensor)
    logger.info(f"Ceiling: {output}")


def test_mish(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute Mish activation function
    output = ttnn.mish(tensor)
    logger.info(f"Mish: {output}")


def test_gez(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are greater than or equal to zero
    output = ttnn.gez(tensor)
    logger.info(f"Greater than or equal to zero: {output}")


def test_gtz(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are greater than zero
    output = ttnn.gtz(tensor)
    logger.info(f"Greater than zero: {output}")


def test_i0(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the zeroth order modified Bessel function
    output = ttnn.i0(tensor)
    logger.info(f"Bessel i0: {output}")


def test_isfinite(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are finite
    output = ttnn.isfinite(tensor)
    logger.info(f"Is finite: {output}")


def test_isinf(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are infinite
    output = ttnn.isinf(tensor)
    logger.info(f"Is infinite: {output}")


def test_isnan(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are NaN (Not a Number)
    output = ttnn.isnan(tensor)
    logger.info(f"Is NaN: {output}")


def test_isneginf(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are negative infinity
    output = ttnn.isneginf(tensor)
    logger.info(f"Is negative infinity: {output}")


def test_isposinf(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are positive infinity
    output = ttnn.isposinf(tensor)
    logger.info(f"Is positive infinity: {output}")


def test_lez(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are less than or equal to zero
    output = ttnn.lez(tensor)
    logger.info(f"Less than or equal to zero: {output}")


def test_logical_not(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute logical NOT
    output = ttnn.logical_not(tensor)
    logger.info(f"Logical NOT: {output}")


def test_ltz(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are less than zero
    output = ttnn.ltz(tensor)
    logger.info(f"Less than zero: {output}")


def test_neg(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the negative values
    output = ttnn.neg(tensor)
    logger.info(f"Negative: {output}")


def test_nez(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Check if elements are not equal to zero
    output = ttnn.nez(tensor)
    logger.info(f"Not equal to zero: {output}")


def test_reciprocal(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the reciprocal (1/x)
    output = ttnn.reciprocal(tensor)
    logger.info(f"Reciprocal: {output}")


def test_relu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply ReLU activation function
    output = ttnn.relu(tensor)
    logger.info(f"ReLU: {output}")


def test_relu6(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply ReLU6 activation function (clamps values between 0 and 6)
    output = ttnn.relu6(tensor)
    logger.info(f"ReLU6: {output}")


def test_sign(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Extract the sign of each element
    output = ttnn.sign(tensor)
    logger.info(f"Sign: {output}")


def test_signbit(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Test the sign bit of each element
    output = ttnn.signbit(tensor)
    logger.info(f"Sign bit: {output}")


def test_silu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply SiLU (Swish) activation function
    output = ttnn.silu(tensor)
    logger.info(f"SiLU: {output}")


def test_sin(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the sine
    output = ttnn.sin(tensor)
    logger.info(f"Sine: {output}")


def test_square(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the square of each element
    output = ttnn.square(tensor)
    logger.info(f"Square: {output}")


def test_tan(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the tangent
    output = ttnn.tan(tensor)
    logger.info(f"Tangent: {output}")


def test_log_sigmoid(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the logarithm of sigmoid
    output = ttnn.log_sigmoid(tensor)
    logger.info(f"Log sigmoid: {output}")


def test_bitwise_not(device):
    # Create a tensor with specific integer values
    tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Apply bitwise NOT operation
    output = ttnn.bitwise_not(tensor)
    logger.info(f"Bitwise NOT: {output}")


def test_deg2rad(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert degrees to radians
    output = ttnn.deg2rad(tensor)
    logger.info(f"Degrees to radians: {output}")


def test_rad2deg(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert radians to degrees
    output = ttnn.rad2deg(tensor)
    logger.info(f"Radians to degrees: {output}")


def test_asinh(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the hyperbolic arcsine
    output = ttnn.asinh(tensor)
    logger.info(f"Hyperbolic arcsine: {output}")


def test_hardsigmoid(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply hard sigmoid activation function
    output = ttnn.hardsigmoid(tensor)
    logger.info(f"Hard sigmoid: {output}")


def test_hardswish(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply hard swish activation function
    output = ttnn.hardswish(tensor)
    logger.info(f"Hard swish: {output}")


def test_softsign(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply softsign activation function
    output = ttnn.softsign(tensor)
    logger.info(f"Softsign: {output}")


def test_cbrt(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the cube root
    output = ttnn.cbrt(tensor)
    logger.info(f"Cube root: {output}")


def test_sqrt(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the square root
    output = ttnn.sqrt(tensor, fast_and_approximate_mode=True)
    logger.info(f"Square root: {output}")


def test_rsqrt(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the reciprocal square root (1/sqrt(x))
    output = ttnn.rsqrt(tensor, fast_and_approximate_mode=True)
    logger.info(f"Reciprocal square root: {output}")


def test_exp(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the exponential function (e^x)
    output = ttnn.exp(tensor, fast_and_approximate_mode=True)
    logger.info(f"Exponential: {output}")


def test_erf(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the error function
    output = ttnn.erf(tensor, fast_and_approximate_mode=True)
    logger.info(f"Error function: {output}")


def test_erfc(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the complementary error function
    output = ttnn.erfc(tensor, fast_and_approximate_mode=True)
    logger.info(f"Complementary error function: {output}")


def test_gelu(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply GELU activation function
    output = ttnn.gelu(tensor, fast_and_approximate_mode=True)
    logger.info(f"GELU: {output}")


def test_log(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the natural logarithm
    output = ttnn.log(tensor, fast_and_approximate_mode=True)
    logger.info(f"Natural logarithm: {output}")


def test_log10(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the base-10 logarithm
    output = ttnn.log10(tensor, fast_and_approximate_mode=True)
    logger.info(f"Base-10 logarithm: {output}")


def test_log2(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the base-2 logarithm
    output = ttnn.log2(tensor, fast_and_approximate_mode=True)
    logger.info(f"Base-2 logarithm: {output}")


def test_log1p(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute log(1 + x)
    output = ttnn.log1p(tensor, fast_and_approximate_mode=True)
    logger.info(f"Log(1 + x): {output}")


def test_elu(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply ELU activation function with alpha parameter
    output = ttnn.elu(tensor, alpha=5)
    logger.info(f"ELU: {output}")


def test_heaviside(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    value = 3

    # Apply Heaviside step function
    output = ttnn.heaviside(tensor, value)
    logger.info(f"Heaviside: {output}")


def test_leaky_relu(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    negative_slope = 3

    # Apply Leaky ReLU activation function
    output = ttnn.leaky_relu(tensor, negative_slope)
    logger.info(f"Leaky ReLU: {output}")


def test_relu_max(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    upper_limit = 3

    # Apply ReLU with upper limit
    output = ttnn.relu_max(tensor, upper_limit)
    logger.info(f"ReLU max: {output}")


def test_relu_min(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    lower_limit = 3

    # Apply ReLU with lower limit
    output = ttnn.relu_min(tensor, lower_limit)
    logger.info(f"ReLU min: {output}")


def test_rpow(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    exponent = 3

    # Compute exponent^tensor
    output = ttnn.rpow(tensor, exponent)
    logger.info(f"Reverse power: {output}")


def test_celu(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    alpha = 3

    # Apply CELU activation function
    output = ttnn.celu(tensor, alpha=alpha)
    logger.info(f"CELU: {output}")


def test_fill(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    fill_value = 3

    # Fill tensor with a specific value
    output = ttnn.fill(tensor, fill_value)
    logger.info(f"Fill: {output}")


def test_glu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dim = 3

    # Apply GLU (Gated Linear Unit)
    output = ttnn.glu(tensor, dim)
    logger.info(f"GLU: {output}")


def test_reglu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dim = 3

    # Apply ReGLU (ReLU Gated Linear Unit)
    output = ttnn.reglu(tensor, dim)
    logger.info(f"ReGLU: {output}")


def test_geglu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dim = 3

    # Apply GeGLU (GELU Gated Linear Unit)
    output = ttnn.geglu(tensor, dim)
    logger.info(f"GeGLU: {output}")


def test_swiglu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dim = 3

    # Apply SwiGLU (Swish Gated Linear Unit)
    output = ttnn.swiglu(tensor, dim)
    logger.info(f"SwiGLU: {output}")


def test_softplus(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply Softplus activation function
    output = ttnn.softplus(tensor, beta=1.0, threshold=20.0)
    logger.info(f"Softplus: {output}")


def test_tanh(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the hyperbolic tangent
    output = ttnn.tanh(tensor, fast_and_approximate_mode=False)
    logger.info(f"Hyperbolic tangent: {output}")


def test_tanhshrink(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply tanh shrink function
    output = ttnn.tanhshrink(tensor, fast_and_approximate_mode=False)
    logger.info(f"Tanh shrink: {output}")


def test_sigmoid_accurate(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply accurate sigmoid activation function
    output = ttnn.sigmoid_accurate(tensor)
    logger.info(f"Sigmoid accurate: {output}")

    # Test with fast_and_approximate_mode=False
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.sigmoid_accurate(tensor, fast_and_approximate_mode=False)
    logger.info(f"Sigmoid accurate (precise): {output}")


def test_sigmoid(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply sigmoid activation function with vector mode
    output = ttnn.sigmoid(tensor, vector_mode=4, fast_and_approximate_mode=True)
    logger.info(f"Sigmoid: {output}")


def test_unary_chain(device):
    # Create a tensor with random normal values
    tensor = ttnn.from_torch(torch.randn([32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
    ]

    # Apply a chain of unary operations
    output = ttnn.unary_chain(tensor, ops_chain)
    logger.info(f"Unary chain: {output}")


def test_bitcast(device):
    # Create a tensor with uint16 values
    tensor = ttnn.from_torch(
        torch.tensor([[16457, 16429], [32641, 31744]], dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Bitcast uint16 to bfloat16 (reinterprets bit pattern)
    output = ttnn.bitcast(tensor, ttnn.bfloat16)
    logger.info(f"Bitcast uint16->bfloat16: {output}")


def test_identity(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.float16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply identity function (returns input unchanged)
    output = ttnn.identity(tensor)
    logger.info(f"Identity: {output}")


def test_cosh(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the hyperbolic cosine
    output = ttnn.cosh(tensor)
    logger.info(f"Hyperbolic cosine: {output}")


def test_digamma(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the digamma function (logarithmic derivative of gamma)
    output = ttnn.digamma(tensor)
    logger.info(f"Digamma: {output}")


def test_lgamma(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the log gamma function
    output = ttnn.lgamma(tensor)
    logger.info(f"Log gamma: {output}")


def test_multigammaln(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the multivariate log gamma function
    output = ttnn.multigammaln(tensor)
    logger.info(f"Multivariate log gamma: {output}")


def test_sinh(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute the hyperbolic sine
    output = ttnn.sinh(tensor)
    logger.info(f"Hyperbolic sine: {output}")


def test_swish(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply Swish activation function
    output = ttnn.swish(tensor)
    logger.info(f"Swish: {output}")


def test_normalize_hw(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Normalize along height and width dimensions
    output = ttnn.normalize_hw(tensor)
    logger.info(f"Normalize HW: {output}")


def test_logical_not_(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply logical NOT (in-place)
    output = ttnn.logical_not_(tensor)
    logger.info(f"Logical NOT (in-place): {output}")


def test_normalize_global(device):
    # Create a tensor with random values
    tensor = ttnn.rand([1, 1, 32, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Normalize globally across all dimensions
    output = ttnn.normalize_global(tensor)
    logger.info(f"Normalize global: {output}")


def test_clip(device):
    # Create tensors for clipping with tensor bounds
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    min_tensor = ttnn.from_torch(
        torch.tensor([[0, 2], [0, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    max_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Clip values using tensor bounds
    output = ttnn.clip(input_tensor, min_tensor, max_tensor)
    logger.info(f"Clip with tensor bounds: {output}")

    # Create tensor for clipping with scalar bounds
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Clip values using scalar bounds
    output = ttnn.clip(input_tensor, min=2, max=9)
    logger.info(f"Clip with scalar bounds: {output}")


def test_clamp(device):
    # Create tensors for clamping with tensor bounds
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    min_tensor = ttnn.from_torch(
        torch.tensor([[0, 2], [0, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    max_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Clamp values using tensor bounds
    output = ttnn.clamp(input_tensor, min_tensor, max_tensor)
    logger.info(f"Clamp with tensor bounds: {output}")

    # Create tensor for clamping with scalar bounds
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Clamp values using scalar bounds
    output = ttnn.clamp(input_tensor, min=2, max=9)
    logger.info(f"Clamp with scalar bounds: {output}")


def test_clone(device):
    # Create a tensor with specific values
    tensor = ttnn.rand([1, 32, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Clone the tensor
    output = ttnn.clone(tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Clone: {output}")


def test_selu(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply SELU activation function
    output = ttnn.selu(tensor, scale=1.0507, alpha=1.67326)
    logger.info(f"SELU: {output}")


def test_hardtanh(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply hard tanh activation function
    output = ttnn.hardtanh(tensor, min_val=-1.0, max_val=1.0)
    logger.info(f"Hard tanh: {output}")


def test_threshold(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    threshold = 1.0
    value = 10.0

    # Apply threshold function
    output = ttnn.threshold(tensor, threshold, value)
    logger.info(f"Threshold: {output}")


def test_tril(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Extract lower triangular part
    output = ttnn.tril(tensor, diagonal=0)
    logger.info(f"Lower triangular: {output}")


def test_triu(device):
    # Create a tensor with random values
    tensor = ttnn.rand([2, 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Extract upper triangular part
    output = ttnn.triu(tensor, diagonal=0)
    logger.info(f"Upper triangular: {output}")


def test_round(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    decimals = 3

    # Round to specified decimal places
    output = ttnn.round(tensor, decimals=decimals)
    logger.info(f"Round: {output}")


def test_polygamma(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the polygamma function
    output = ttnn.polygamma(tensor, 3)
    logger.info(f"Polygamma: {output}")


def test_hardshrink(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply hard shrinkage function
    output = ttnn.hardshrink(tensor, lambd=5)
    logger.info(f"Hard shrink: {output}")


def test_softshrink(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply soft shrinkage function
    output = ttnn.softshrink(tensor, lambd=5)
    logger.info(f"Soft shrink: {output}")


def test_logit(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute the logit function
    output = ttnn.logit(tensor, eps=5)
    logger.info(f"Logit: {output}")


def test_rdiv(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    value = 2

    # Compute reverse division (value / tensor)
    output = ttnn.rdiv(tensor, value, rounding_mode=None)
    logger.info(f"Reverse division: {output}")


def test_bitwise_left_shift(device):
    # Create tensors with specific integer values
    tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Apply bitwise left shift operation
    output = ttnn.bitwise_left_shift(tensor1, tensor2)
    logger.info(f"Bitwise left shift: {output}")


def test_bitwise_right_shift(device):
    # Create tensors with specific integer values
    tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Apply bitwise right shift operation
    output = ttnn.bitwise_right_shift(tensor1, tensor2)
    logger.info(f"Bitwise right shift: {output}")


def test_prelu(device):
    # Create tensors for PReLU activation function
    tensor1 = ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    weight = 0.25

    # Apply PReLU activation function with scalar weight
    output = ttnn.prelu(tensor1, weight)
    logger.info(f"PReLU: {output}")


def test_remainder(device):
    # Create tensors for remainder operation
    tensor1 = ttnn.from_torch(
        torch.tensor([[5, 7], [9, 11]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the remainder of division
    output = ttnn.remainder(tensor1, tensor2)
    logger.info(f"Remainder: {output}")


def test_hardmish(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[-2.0, -1.0], [1.0, 2.0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply Hard Mish activation function
    output = ttnn.hardmish(tensor)
    logger.info(f"Hard Mish: {output}")


def test_i1(device):
    # Create a tensor with specific values
    tensor = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Compute the modified Bessel function of the first kind of order 1
    output = ttnn.i1(tensor)
    logger.info(f"Bessel i1: {output}")


def test_var_hw(device):
    # Create a 4D tensor
    tensor = ttnn.from_torch(torch.randn(1, 2, 64, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    # Compute variance across height and width dimensions
    output = ttnn.var_hw(tensor)
    logger.info(f"Variance HW: {output}")


def test_std_hw(device):
    # Create a 4D tensor
    tensor = ttnn.from_torch(torch.randn(1, 2, 64, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    # Compute standard deviation across height and width dimensions
    output = ttnn.std_hw(tensor)
    logger.info(f"Standard Deviation HW: {output}")


def test_logical_left_shift(device):
    # Create a tensor with specific integer values
    tensor = ttnn.from_torch(torch.tensor([[1, 2], [4, 8]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Perform logical left shift by 2 bits
    output = ttnn.logical_left_shift(tensor, 2)
    logger.info(f"Logical left shift: {output}")


def test_logical_right_shift(device):
    # Create tensors for logical right shift
    tensor = ttnn.from_torch(
        torch.tensor([[128, 256], [512, 1024]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device
    )
    shift_amt = ttnn.from_torch(torch.full((2, 2), 3, dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)

    # Perform logical right shift by 3 bits
    output = ttnn.logical_right_shift(tensor, shift_amt)
    logger.info(f"Logical right shift: {output}")
