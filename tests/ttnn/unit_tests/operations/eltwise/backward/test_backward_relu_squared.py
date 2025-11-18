# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 64, 128])),
    ),
)
def test_bw_relu_squared(input_shapes, device):
    """Test relu_squared backward operation with various shapes."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def test_bw_relu_squared_edge_cases(device):
    """Test relu_squared backward with edge cases."""
    # Use a shape that matches the number of edge case values
    edge_values = torch.tensor([-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0], dtype=torch.bfloat16, requires_grad=True)
    # Reshape to a valid 4D shape: [1, 1, 7, 1] = 7 elements (or pad to [1, 1, 3, 3] = 9)
    # Let's use [1, 1, 7, 1] to match exactly 7 values
    edge_cases = edge_values.reshape([1, 1, 7, 1])
    input_shapes = edge_cases.shape

    grad_data = torch.randn(input_shapes, dtype=torch.bfloat16) * 10
    grad_tensor = ttnn.from_torch(grad_data, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.from_torch(edge_cases, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared_bw)
    golden_tensor = golden_function(grad_data, edge_cases)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def test_bw_relu_squared_negative_inputs(device):
    """Test that negative inputs produce zero gradients."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)
    output = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # All gradients should be zero for negative inputs
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)


def test_bw_relu_squared_positive_inputs(device):
    """Test gradient computation for positive inputs."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def test_bw_relu_squared_mathematical_correctness(device):
    """Verify backward matches PyTorch autograd computation."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)
    tt_output = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # PyTorch autograd computation
    pyt_input = in_data.clone().detach().requires_grad_(True)
    pyt_output = torch.square(torch.relu(pyt_input))
    pyt_output.backward(gradient=grad_data)
    pyt_grad = pyt_input.grad

    # Verify tt_output matches PyTorch gradient
    assert torch.allclose(
        tt_output, pyt_grad, atol=1e-3, rtol=1e-2
    ), f"tt_output does not match PyTorch gradient. Max diff: {torch.max(torch.abs(tt_output - pyt_grad)).item():.6f}"

    comp_pass = compare_pcc(tt_output_tensor_on_device, [pyt_grad])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "grad_range",
    [
        (-100, 100),
        (-10, 10),
        (-1, 1),
        (0, 10),  # Only positive gradients
        (-10, 0),  # Only negative gradients
    ],
)
def test_bw_relu_squared_gradient_ranges(input_shapes, grad_range, device):
    """Test with various gradient value ranges."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, grad_range[0], grad_range[1], device)

    tt_output_tensor_on_device = ttnn.relu_squared_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
