# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
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
def test_relu_squared(input_shapes, device):
    """Test relu_squared forward operation with various shapes."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    output_tensor = ttnn.relu_squared(input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared)
    golden_tensor = golden_function(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


def test_relu_squared_edge_cases(device):
    """Test relu_squared with edge cases: zeros, negatives, positives."""
    # Use a shape that matches the number of edge case values
    edge_values = torch.tensor([-10.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 10.0], dtype=torch.bfloat16)
    # Reshape to a valid 4D shape: [1, 1, 3, 3] = 9 elements
    edge_cases = edge_values.reshape([1, 1, 3, 3])

    input_tensor = ttnn.from_torch(edge_cases, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu_squared(input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared)
    golden_tensor = golden_function(edge_cases)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


def test_relu_squared_negative_values(device):
    """Test that negative values produce zero output."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device)

    output_tensor = ttnn.relu_squared(input_tensor)
    output = ttnn.to_torch(output_tensor)

    # All outputs should be zero for negative inputs
    assert torch.all(output == 0.0)


def test_relu_squared_positive_values(device):
    """Test that positive values are squared correctly."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device)

    output_tensor = ttnn.relu_squared(input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_squared)
    golden_tensor = golden_function(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


def test_relu_squared_mathematical_correctness(device):
    """Verify relu_squared matches manual computation."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    output_tensor = ttnn.relu_squared(input_tensor)
    output = ttnn.to_torch(output_tensor)

    # Manual computation: square(relu(x))
    expected = torch.square(torch.relu(in_data))

    assert_with_pcc(output, expected, pcc=0.9999)
