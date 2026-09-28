# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
import torch

import ttnn


@contextlib.contextmanager
def comparison_mode(pcc=0.99, raise_on_failure=True):
    with (
        ttnn.manage_config("enable_comparison_mode", True),
        ttnn.manage_config("comparison_mode_pcc", pcc),
        ttnn.manage_config("comparison_mode_should_raise_exception", raise_on_failure),
    ):
        yield


@pytest.mark.requires_fast_runtime_mode_off
def test_leaky_relu_with_positional_negative_slope_in_comparison_mode(device):
    torch_input = torch.full((1, 1, 32, 32), -2.0, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.leaky_relu.md: leaky-relu-golden-ignores-positional-slope
    with comparison_mode():
        output_tensor = ttnn.leaky_relu(input_tensor, 0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_sum_with_scalar_in_comparison_mode(device):
    torch_input = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.sum.md: sum-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.sum(input_tensor, dim=-1, keepdim=True, scalar=0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_sum_int32_with_fractional_scalar_in_comparison_mode(device):
    torch_input = torch.ones((1, 1, 32, 32), dtype=torch.int32)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # generated/ttnn.sum.md: sum-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.sum(input_tensor, dim=-1, keepdim=True, scalar=0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_mean_with_zero_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.mean.md: mean-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=True, scalar=0.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_max_with_negative_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.max.md: max-scalar-not-modeled
    with comparison_mode():
        output_tensor = ttnn.max(input_tensor, dim=-1, keepdim=True, scalar=-2.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_min_with_negative_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.min.md: inherits mean-scalar-ignored from ttnn.mean
    with comparison_mode():
        output_tensor = ttnn.min(input_tensor, dim=-1, keepdim=True, scalar=-2.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_var_with_zero_scalar_in_comparison_mode(device):
    columns = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    row_scales = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1)
    torch_input = (row_scales * columns).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.var.md: var-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.var(input_tensor, dim=-1, keepdim=True, scalar=0.0, correction=False)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_std_with_zero_scalar_in_comparison_mode(device):
    columns = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    row_scales = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1)
    torch_input = (row_scales * columns).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.std.md: std-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.std(input_tensor, dim=-1, keepdim=True, scalar=0.0, correction=False)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_selu_with_non_default_parameters_in_comparison_mode(device):
    torch_input = torch.full((1, 1, 32, 32), -1.0, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.selu.md: selu-golden-ignores-parameters
    with comparison_mode():
        output_tensor = ttnn.selu(input_tensor, scale=0.9, alpha=1.2)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_softmax_over_dimension_zero_in_comparison_mode(device):
    torch_input = torch.zeros((2, 32, 32), dtype=torch.bfloat16)
    torch_input[1] = 4.0
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.softmax.md: softmax-zero-dim
    with comparison_mode():
        output_tensor = ttnn.softmax(input_tensor, dim=0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_with_bias_in_comparison_mode(device):
    torch_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    torch_weight = torch.ones((1, 32), dtype=torch.bfloat16)
    torch_bias = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).reshape(1, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # generated/ttnn.rms_norm.md: rms-norm-missing-bias
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, weight=weight, bias=bias)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_with_residual_in_comparison_mode(device):
    torch_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    torch_residual = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).repeat(32, 1)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.rms_norm.md: rms-norm-missing-residual
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, residual_input_tensor=residual)

    assert isinstance(output_tensor, ttnn.Tensor)
