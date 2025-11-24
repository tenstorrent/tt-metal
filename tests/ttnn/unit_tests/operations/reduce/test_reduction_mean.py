# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose
from models.common.utility_functions import torch_random, comp_allclose


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_scaling(device, shape, dim, keepdim):
    """Use assert_allclose with ones() to test that mean's scaling factor is
    computed correctly.
    """
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)
    torch_output_tensor = torch_output_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_allclose(torch_output_tensor, output_tensor, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("scalar", [2.0])
def test_mean_scaling_factor(device, shape, dim, scalar):
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch.bfloat16)
    torch_output_tensor = torch_output_tensor * scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_allclose(torch_output_tensor, output_tensor, rtol=1e-2, atol=1e-2)
