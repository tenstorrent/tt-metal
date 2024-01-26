# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2, -3])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_softmax(device, batch_size, h, w, dim, input_layout):
    if dim != -1 and input_layout != ttnn.TILE_LAYOUT:
        pytest.skip("Not supported yet")

    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, input_layout)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@skip_for_wormhole_b0()
def test_softmax_with_3D(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 1500, 1500), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@skip_for_wormhole_b0()
def test_softmax_with_padded_tile_layout(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 2, 2), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.skip(reason="#4629: softmax pcc at 0.948 when comparing to torch")
@skip_for_wormhole_b0()
def test_specific_tensor_combination(device):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_file = os.path.join(current_dir, "softmax_weights.pt")
    torch_input_tensor = torch.load(tensor_file)

    torch_output_tensor = torch.softmax(torch_input_tensor, -1)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)

    output = ttnn.softmax(input_tensor, -1)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9999)
