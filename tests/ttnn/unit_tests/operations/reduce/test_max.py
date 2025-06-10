# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, is_grayskull, skip_for_grayskull


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
def test_max(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_grayskull("May fail on GS if run all the tests in this file. #17084")
@pytest.mark.parametrize("batch_size1", [2])
@pytest.mark.parametrize("batch_size2", [32])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dim", [-3])
def test_max_4d(device, batch_size1, batch_size2, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size1, batch_size2, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dim", [-2, -1, 0, 1])
def test_max_2d(device, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
def test_max_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.max(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((32, 32, 32, 64), -4),
        ((2, 32, 32, 64), -3),
        ((32, 32, 64), -3),
        ((1, 2, 3, 4), -1),
        ((2, 22, 37, 41), -4),
        ((2, 32, 64, 64), -3),
        ((2, 22, 37, 41), -3),
        ((2, 32, 64, 64), -2),
        ((2, 22, 37, 41), -1),
        ((2, 32, 64, 64), -1),
        ((2, 22, 37), -3),
        ((2, 22, 37), -2),
        ((2, 22, 37), -1),
        ((1, 6, 7), -3),
        ((32, 6, 7), -3),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_max_dim(device, input_shape_and_dim, keepdim):
    input_shape, max_dim = input_shape_and_dim
    if is_grayskull() and (
        input_shape[-1] % 32 != 0 or input_shape[-2] % 32 != 0 or input_shape[max_dim] % 32 != 0 or max_dim <= -2
    ):
        pytest.skip("May fail on GS if run all the tests in this file. #17084")

    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=max_dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor, dim=max_dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
