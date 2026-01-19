# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_clamp_bw(device):
    # Create sample tensors
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define min and max values for clamping
    min_val = 0.5
    max_val = 2.0

    # Call the clamp_bw function
    output = ttnn.clamp_bw(grad_tensor, input_tensor, min_val, max_val)
    logger.info(f"Clamped Output Backward: {output}")


def test_clip_bw(device):
    # Create sample tensors for backward clip operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define min and max values for clipping
    min_val = 0.5
    max_val = 2.0

    # Call the clip_bw function
    output = ttnn.clip_bw(grad_tensor, input_tensor, min_val, max_val)
    logger.info(f"Clip Backward: {output}")


def test_hardtanh_bw(device):
    # Create sample tensors for backward hard tanh operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the hardtanh_bw function with min and max values
    output = ttnn.hardtanh_bw(grad_tensor, input_tensor, min=-1.0, max=1.0)
    logger.info(f"Hard Tanh Backward: {output}")


def test_hardshrink_bw(device):
    # Create sample tensors for backward hard shrink operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the hardshrink_bw function with lambda parameter
    output = ttnn.hardshrink_bw(grad_tensor, input_tensor, lambd=0.5)
    logger.info(f"Hard Shrink Backward: {output}")


def test_softshrink_bw(device):
    # Create sample tensors for backward soft shrink operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the softshrink_bw function with lambda parameter
    output = ttnn.softshrink_bw(grad_tensor, input_tensor, lambd=0.5)
    logger.info(f"Soft Shrink Backward: {output}")


def test_leaky_relu_bw(device):
    # Create sample tensors for backward leaky relu operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the leaky_relu_bw function with negative slope parameter
    output = ttnn.leaky_relu_bw(grad_tensor, input_tensor, negative_slope=0.01)
    logger.info(f"Leaky ReLU Backward: {output}")


def test_elu_bw(device):
    # Create sample tensors for backward ELU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the elu_bw function with alpha parameter
    output = ttnn.elu_bw(grad_tensor, input_tensor, alpha=1.0)
    logger.info(f"ELU Backward: {output}")


def test_celu_bw(device):
    # Create sample tensors for backward CELU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the celu_bw function with alpha parameter
    output = ttnn.celu_bw(grad_tensor, input_tensor, alpha=1.0)
    logger.info(f"CELU Backward: {output}")


def test_logiteps_bw(device):
    # Create sample tensors for backward logit with epsilon operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the logiteps_bw function with epsilon parameter
    output = ttnn.logiteps_bw(grad_tensor, input_tensor, eps=0.0)
    logger.info(f"Logit Eps Backward: {output}")


def test_threshold_bw(device):
    # Create sample tensors for backward threshold operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define threshold and value parameters
    threshold = 1.0
    value = 1.0

    # Call the threshold_bw function
    output = ttnn.threshold_bw(grad_tensor, input_tensor, threshold, value)
    logger.info(f"Threshold Backward: {output}")


def test_softplus_bw(device):
    # Create sample tensors for backward softplus operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the softplus_bw function with beta and threshold parameters
    output = ttnn.softplus_bw(grad_tensor, input_tensor, beta=1.0, threshold=20.0)
    logger.info(f"Softplus Backward: {output}")


def test_rdiv_bw(device):
    # Create sample tensors for backward reverse division operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define scalar value for reverse division
    scalar = 0.5

    # Call the rdiv_bw function with scalar and round mode
    output = ttnn.rdiv_bw(grad_tensor, input_tensor, scalar, rounding_mode=None)
    logger.info(f"Reverse Division Backward: {output}")


def test_repeat_bw(device):
    # Create sample tensors for backward repeat operation
    # Input tensor that will be repeated
    input_tensor = ttnn.from_torch(
        torch.rand([1, 1, 32, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Grad tensor matching the shape after repeat operation (2x repeat in dim 0)
    grad_tensor = ttnn.from_torch(
        torch.rand([2, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define the shape for repeat operation
    shape = [2, 1, 1, 1]

    # Call the repeat_bw function
    output = ttnn.repeat_bw(grad_tensor, input_tensor, shape)
    logger.info(f"Repeat Backward: {output}")


def test_gelu_bw(device):
    # Create sample tensors for backward GELU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the gelu_bw function with approximation method
    output = ttnn.gelu_bw(grad_tensor, input_tensor, approximate="none")
    logger.info(f"GELU Backward: {output}")


def test_pow_bw(device):
    # Create sample tensors for backward power operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define exponent for power operation
    exponent = 2.0

    # Call the pow_bw function
    output = ttnn.pow_bw(grad_tensor, input_tensor, exponent)
    logger.info(f"Power Backward: {output}")


def test_exp_bw(device):
    # Create sample tensors for backward exponential operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the exp_bw function
    output = ttnn.exp_bw(grad_tensor, input_tensor)
    logger.info(f"Exponential Backward: {output}")


def test_tanh_bw(device):
    # Create sample tensors for backward hyperbolic tangent operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the tanh_bw function
    output = ttnn.tanh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Tangent Backward: {output}")


def test_sqrt_bw(device):
    # Create sample tensors for backward square root operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the sqrt_bw function
    output = ttnn.sqrt_bw(grad_tensor, input_tensor)
    logger.info(f"Square Root Backward: {output}")


def test_multigammaln_bw(device):
    # Create sample tensors for backward multivariate log gamma operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the multigammaln_bw function
    output = ttnn.multigammaln_bw(grad_tensor, input_tensor)
    logger.info(f"Multivariate Log Gamma Backward: {output}")


def test_lgamma_bw(device):
    # Create sample tensors for backward log gamma operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the lgamma_bw function
    output = ttnn.lgamma_bw(grad_tensor, input_tensor)
    logger.info(f"Log Gamma Backward: {output}")


def test_fill_bw(device):
    # Create sample tensors for backward fill operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the fill_bw function
    output = ttnn.fill_bw(grad_tensor, input_tensor)
    logger.info(f"Fill Backward: {output}")


def test_prod_bw(device):
    # Create sample tensors for backward product operation
    grad_tensor = ttnn.from_torch(
        torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.rand([1, 1, 32, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define dimension for product operation
    dim = 0

    # Call the prod_bw function with specific dimension
    output = ttnn.prod_bw(grad_tensor, input_tensor, dim=dim)
    logger.info(f"Prod Backward (dim={dim}): {output}")

    # Call the prod_bw function for all dimensions
    all_dims_output = ttnn.prod_bw(grad_tensor, input_tensor)
    logger.info(f"Prod Backward (all dims): {all_dims_output}")


def test_hardsigmoid_bw(device):
    # Create sample tensors for backward hard sigmoid operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the hardsigmoid_bw function
    output = ttnn.hardsigmoid_bw(grad_tensor, input_tensor)
    logger.info(f"Hard Sigmoid Backward: {output}")


def test_cos_bw(device):
    # Create sample tensors for backward cosine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the cos_bw function
    output = ttnn.cos_bw(grad_tensor, input_tensor)
    logger.info(f"Cosine Backward: {output}")


def test_acosh_bw(device):
    # Create sample tensors for backward hyperbolic arccosine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the acosh_bw function
    output = ttnn.acosh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Arccosine Backward: {output}")


def test_acos_bw(device):
    # Create sample tensors for backward arccosine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the acos_bw function
    output = ttnn.acos_bw(grad_tensor, input_tensor)
    logger.info(f"Arccosine Backward: {output}")


def test_atan_bw(device):
    # Create sample tensors for backward arctangent operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the atan_bw function
    output = ttnn.atan_bw(grad_tensor, input_tensor)
    logger.info(f"Arctangent Backward: {output}")


def test_rad2deg_bw(device):
    # Create sample tensors for backward radians to degrees operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the rad2deg_bw function
    output = ttnn.rad2deg_bw(grad_tensor, input_tensor)
    logger.info(f"Radians to Degrees Backward: {output}")


def test_frac_bw(device):
    # Create sample tensors for backward fractional part operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the frac_bw function
    output = ttnn.frac_bw(grad_tensor, input_tensor)
    logger.info(f"Fractional Part Backward: {output}")


def test_trunc_bw(device):
    # Create sample tensors for backward truncation operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the trunc_bw function
    output = ttnn.trunc_bw(grad_tensor, input_tensor)
    logger.info(f"Truncation Backward: {output}")


def test_log_sigmoid_bw(device):
    # Create sample tensors for backward log sigmoid operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the log_sigmoid_bw function
    output = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)
    logger.info(f"Log Sigmoid Backward: {output}")


def test_fill_zero_bw(device):
    # Create sample tensors for backward fill zero operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the fill_zero_bw function
    output = ttnn.fill_zero_bw(grad_tensor, input_tensor)
    logger.info(f"Fill Zero Backward: {output}")


def test_i0_bw(device):
    # Create sample tensors for backward Bessel I0 operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the i0_bw function
    output = ttnn.i0_bw(grad_tensor, input_tensor)
    logger.info(f"Bessel I0 Backward: {output}")


def test_tan_bw(device):
    # Create sample tensors for backward tangent operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the tan_bw function
    output = ttnn.tan_bw(grad_tensor, input_tensor)
    logger.info(f"Tangent Backward: {output}")


def test_sigmoid_bw(device):
    # Create sample tensors for backward sigmoid operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the sigmoid_bw function
    output = ttnn.sigmoid_bw(grad_tensor, input_tensor)
    logger.info(f"Sigmoid Backward: {output}")


def test_rsqrt_bw(device):
    # Create sample tensors for backward reciprocal square root operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the rsqrt_bw function
    output = ttnn.rsqrt_bw(grad_tensor, input_tensor)
    logger.info(f"Reciprocal Square Root Backward: {output}")


def test_neg_bw(device):
    # Create sample tensors for backward negation operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the neg_bw function
    output = ttnn.neg_bw(grad_tensor, input_tensor)
    logger.info(f"Negation Backward: {output}")


def test_relu_bw(device):
    # Create sample tensors for backward ReLU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the relu_bw function
    output = ttnn.relu_bw(grad_tensor, input_tensor)
    logger.info(f"ReLU Backward: {output}")


def test_logit_bw(device):
    # Create sample tensors for backward logit operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the logit_bw function
    output = ttnn.logit_bw(grad_tensor, input_tensor)
    logger.info(f"Logit Backward: {output}")


def test_floor_bw(device):
    # Create sample tensors for backward floor operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the floor_bw function
    output = ttnn.floor_bw(grad_tensor, input_tensor)
    logger.info(f"Floor Backward: {output}")


def test_rpow_bw(device):
    # Create sample tensors for backward reverse power operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    # Define exponent for reverse power operation
    exponent = 2.0

    # Call the rpow_bw function
    output = ttnn.rpow_bw(grad_tensor, input_tensor, exponent)
    logger.info(f"Reverse Power Backward: {output}")


def test_round_bw(device):
    # Create sample tensors for backward round operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the round_bw function
    output = ttnn.round_bw(grad_tensor, input_tensor)
    logger.info(f"Round Backward: {output}")


def test_log_bw(device):
    # Create sample tensors for backward natural logarithm operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the log_bw function
    output = ttnn.log_bw(grad_tensor, input_tensor)
    logger.info(f"Natural Logarithm Backward: {output}")


def test_relu6_bw(device):
    # Create sample tensors for backward ReLU6 operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the relu6_bw function
    output = ttnn.relu6_bw(grad_tensor, input_tensor)
    logger.info(f"ReLU6 Backward: {output}")


def test_abs_bw(device):
    # Create sample tensors for backward absolute value operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the abs_bw function
    output = ttnn.abs_bw(grad_tensor, input_tensor)
    logger.info(f"Absolute Value Backward: {output}")


def test_silu_bw(device):
    # Create sample tensors for backward SiLU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the silu_bw function
    output = ttnn.silu_bw(grad_tensor, input_tensor)
    logger.info(f"SiLU Backward: {output}")


def test_selu_bw(device):
    # Create sample tensors for backward SELU operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the selu_bw function
    output = ttnn.selu_bw(grad_tensor, input_tensor)
    logger.info(f"SELU Backward: {output}")


def test_square_bw(device):
    # Create sample tensors for backward square operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the square_bw function
    output = ttnn.square_bw(grad_tensor, input_tensor)
    logger.info(f"Square Backward: {output}")


def test_hardswish_bw(device):
    # Create sample tensors for backward hard swish operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the hardswish_bw function
    output = ttnn.hardswish_bw(grad_tensor, input_tensor)
    logger.info(f"Hard Swish Backward: {output}")


def test_tanhshrink_bw(device):
    # Create sample tensors for backward tanh shrink operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the tanhshrink_bw function
    output = ttnn.tanhshrink_bw(grad_tensor, input_tensor)
    logger.info(f"Tanh Shrink Backward: {output}")


def test_atanh_bw(device):
    # Create sample tensors for backward hyperbolic arctangent operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the atanh_bw function
    output = ttnn.atanh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Arctangent Backward: {output}")


def test_asin_bw(device):
    # Create sample tensors for backward arcsine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the asin_bw function
    output = ttnn.asin_bw(grad_tensor, input_tensor)
    logger.info(f"Arcsine Backward: {output}")


def test_asinh_bw(device):
    # Create sample tensors for backward hyperbolic arcsine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the asinh_bw function
    output = ttnn.asinh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Arcsine Backward: {output}")


def test_sin_bw(device):
    # Create sample tensors for backward sine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the sin_bw function
    output = ttnn.sin_bw(grad_tensor, input_tensor)
    logger.info(f"Sine Backward: {output}")


def test_sinh_bw(device):
    # Create sample tensors for backward hyperbolic sine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the sinh_bw function
    output = ttnn.sinh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Sine Backward: {output}")


def test_log10_bw(device):
    # Create sample tensors for backward base-10 logarithm operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the log10_bw function
    output = ttnn.log10_bw(grad_tensor, input_tensor)
    logger.info(f"Base-10 Logarithm Backward: {output}")


def test_log1p_bw(device):
    # Create sample tensors for backward log(1+x) operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the log1p_bw function
    output = ttnn.log1p_bw(grad_tensor, input_tensor)
    logger.info(f"Log(1+x) Backward: {output}")


def test_erfc_bw(device):
    # Create sample tensors for backward complementary error function operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the erfc_bw function
    output = ttnn.erfc_bw(grad_tensor, input_tensor)
    logger.info(f"Complementary Error Function Backward: {output}")


def test_ceil_bw(device):
    # Create sample tensors for backward ceiling operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the ceil_bw function
    output = ttnn.ceil_bw(grad_tensor, input_tensor)
    logger.info(f"Ceiling Backward: {output}")


def test_softsign_bw(device):
    # Create sample tensors for backward softsign operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the softsign_bw function
    output = ttnn.softsign_bw(grad_tensor, input_tensor)
    logger.info(f"Softsign Backward: {output}")


def test_cosh_bw(device):
    # Create sample tensors for backward hyperbolic cosine operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the cosh_bw function
    output = ttnn.cosh_bw(grad_tensor, input_tensor)
    logger.info(f"Hyperbolic Cosine Backward: {output}")


def test_log2_bw(device):
    # Create sample tensors for backward base-2 logarithm operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the log2_bw function
    output = ttnn.log2_bw(grad_tensor, input_tensor)
    logger.info(f"Base-2 Logarithm Backward: {output}")


def test_sign_bw(device):
    # Create sample tensors for backward sign operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the sign_bw function
    output = ttnn.sign_bw(grad_tensor, input_tensor)
    logger.info(f"Sign Backward: {output}")


def test_div_no_nan_bw(device):
    # Create sample tensors for backward division without NaN operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    scalar = 2.0

    # Call the div_no_nan_bw function
    output = ttnn.div_no_nan_bw(grad_tensor, input_tensor, scalar)
    logger.info(f"Division No NaN Backward: {output}")


def test_exp2_bw(device):
    # Create sample tensors for backward base-2 exponential operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the exp2_bw function
    output = ttnn.exp2_bw(grad_tensor, input_tensor)
    logger.info(f"Base-2 Exponential Backward: {output}")


def test_expm1_bw(device):
    # Create sample tensors for backward exp(x)-1 operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the expm1_bw function
    output = ttnn.expm1_bw(grad_tensor, input_tensor)
    logger.info(f"Exp(x)-1 Backward: {output}")


def test_reciprocal_bw(device):
    # Create sample tensors for backward reciprocal operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the reciprocal_bw function
    output = ttnn.reciprocal_bw(grad_tensor, input_tensor)
    logger.info(f"Reciprocal Backward: {output}")


def test_digamma_bw(device):
    # Create sample tensors for backward digamma operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the digamma_bw function
    output = ttnn.digamma_bw(grad_tensor, input_tensor)
    logger.info(f"Digamma Backward: {output}")


def test_erfinv_bw(device):
    # Create sample tensors for backward inverse error function operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the erfinv_bw function
    output = ttnn.erfinv_bw(grad_tensor, input_tensor)
    logger.info(f"Inverse Error Function Backward: {output}")


def test_erf_bw(device):
    # Create sample tensors for backward error function operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the erf_bw function
    output = ttnn.erf_bw(grad_tensor, input_tensor)
    logger.info(f"Error Function Backward: {output}")


def test_deg2rad_bw(device):
    # Create sample tensors for backward degrees to radians conversion operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Call the deg2rad_bw function
    output = ttnn.deg2rad_bw(grad_tensor, input_tensor)
    logger.info(f"Degrees to Radians Backward: {output}")


def test_polygamma_bw(device):
    # Create sample tensors for backward polygamma operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    input_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    n = 1

    # Call the polygamma_bw function
    output = ttnn.polygamma_bw(grad_tensor, input_tensor, n)
    logger.info(f"Polygamma Backward: {output}")


# Complex backward operations
def test_polar_bw(device):
    # Create sample tensors for backward polar coordinate operation
    # Create complex input tensor
    input_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(input_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(input_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Create complex gradient tensor
    grad_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_tensor = ttnn.complex_tensor(
        ttnn.Tensor(grad_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(grad_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Call the polar_bw function
    output = ttnn.polar_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Polar Backward: {output}")


def test_imag_bw(device):
    # Create sample tensors for backward imaginary part operation
    # Create a complex input tensor
    input_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(input_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(input_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Gradient tensor for the imaginary part (regular tensor, not complex)
    grad_data = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # Call the imag_bw function
    output = ttnn.imag_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Imaginary Part Backward: {output}")


def test_real_bw(device):
    # Create sample tensors for backward real part operation
    # Create a complex input tensor
    input_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(input_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(input_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Gradient tensor for the real part (regular tensor, not complex)
    grad_data = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # Call the real_bw function
    output = ttnn.real_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Real Part Backward: {output}")


def test_angle_bw(device):
    # Create sample tensors for backward angle operation
    # Create a complex input tensor
    input_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(input_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(input_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Gradient tensor for the angle (regular tensor, not complex)
    grad_data = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # Call the angle_bw function
    output = ttnn.angle_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Angle Backward: {output}")


def test_conj_bw(device):
    # Create sample tensors for backward complex conjugate operation
    # Create a complex input tensor
    input_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(input_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(input_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Create complex gradient tensor
    grad_real = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_imag = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    grad_tensor = ttnn.complex_tensor(
        ttnn.Tensor(grad_real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(grad_imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Call the conj_bw function
    output = ttnn.conj_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Complex Conjugate Backward: {output}")
