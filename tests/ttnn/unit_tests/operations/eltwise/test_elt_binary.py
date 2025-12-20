# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import pi
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random


def run_elt_binary_test_range(device, h, w, ttnn_function, low, high, pcc=0.9999):
    torch.manual_seed(0)
    low = low
    high = high
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ldexp(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.ldexp, -60, 60, pcc=0.9995)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp, -80, 80)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp2(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp2, -60, 100, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_and(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.logical_and, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_or(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.logical_or, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_xor(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.logical_xor, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_xlogy(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.xlogy, 1e-6, 1e6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_bias_gelu(device_module, h, w):
    device = device_module
    run_elt_binary_test_range(device, h, w, ttnn.bias_gelu, -100, 100)


def run_elt_binary_test_min_max(device, h, w, ttnn_function, low, high, pcc=0.9999):
    torch.manual_seed(0)
    low = low
    high = high
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_maximum(device_module, h, w):
    device = device_module
    run_elt_binary_test_min_max(device, h, w, ttnn.maximum, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_minimum(device_module, h, w):
    device = device_module
    run_elt_binary_test_min_max(device, h, w, ttnn.minimum, -100, 100)


def test_arithmetic_operators(device_module):
    device = device_module
    """Test basic arithmetic operators (+, -, *, /) on ttnn tensors"""

    # Create test tensors with different values
    a_torch = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
    b_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)

    # Convert to ttnn tensors on device
    a = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)

    # Test operations
    c = a + b  # Addition: 4 + 2 = 6
    d = a - b  # Subtraction: 4 - 2 = 2
    e = a * b  # Multiplication: 4 * 2 = 8
    f = a / b  # Division: 4 / 2 = 2
    g = a / 2  # Tensor / scalar: 4 / 2 = 2
    h = 8 / a  # Scalar / tensor: 8 / 4 = 2

    # Verify results
    c_torch = ttnn.to_torch(c)
    expected_add = torch.full((32, 32), 6.0, dtype=torch.bfloat16)
    assert torch.equal(c_torch, expected_add), "Addition result incorrect"

    d_torch = ttnn.to_torch(d)
    expected_sub = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(d_torch, expected_sub), "Subtraction result incorrect"

    e_torch = ttnn.to_torch(e)
    expected_mul = torch.full((32, 32), 8.0, dtype=torch.bfloat16)
    assert torch.equal(e_torch, expected_mul), "Multiplication result incorrect"

    f_torch = ttnn.to_torch(f)
    expected_div = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(f_torch, expected_div), "Division result incorrect"

    g_torch = ttnn.to_torch(g)
    expected_tensor_div_scalar = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(g_torch, expected_tensor_div_scalar), "Tensor / scalar result incorrect"

    h_torch = ttnn.to_torch(h)
    expected_scalar_div_tensor = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(h_torch, expected_scalar_div_tensor), "Scalar / tensor result incorrect"
