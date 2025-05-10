# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2, (2, 1)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
def test_sum_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [1, 9])
@pytest.mark.parametrize("c", [1, 9])
@pytest.mark.parametrize("h", [9, 37])
@pytest.mark.parametrize("w", [9, 63])
@pytest.mark.parametrize("dim", [None, 0, 1, 2, 3])
def test_sum_4d(device, n, c, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.float32)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
