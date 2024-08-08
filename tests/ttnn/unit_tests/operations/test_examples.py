# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.prim.example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_example_compute_output_shapes(height, width):
    input_tensor = ttnn.zeros((height, width), layout=ttnn.TILE_LAYOUT)
    output_shape = ttnn.prim.example.compute_output_shapes(input_tensor)
    assert output_shape == input_tensor.shape


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_example_create_output_tensors(device, height, width):
    input_tensor = ttnn.zeros((height, width), layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.prim.example.create_output_tensors(input_tensor)
    assert output_tensor.shape == input_tensor.shape


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_composite_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.composite_example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
