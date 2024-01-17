# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [20])
@pytest.mark.parametrize("w", [4])
def test_add(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=0)
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    # input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    # TODO : research why reversing these two causes a hang
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    # input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.TILE_LAYOUT)
    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=0)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
