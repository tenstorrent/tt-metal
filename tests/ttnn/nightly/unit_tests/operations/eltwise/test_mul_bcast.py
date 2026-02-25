# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_mul_channel_bcast_repeat(device, h, w):
    torch_input_tensor_a = torch.rand((16, 16, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((16, 1, h, w), dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_mul_batch_bcast_repeat(device, h, w):
    torch_input_tensor_a = torch.rand((1, 16, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((16, 16, h, w), dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    assert_with_pcc(torch_output_tensor, output, 0.9999)
