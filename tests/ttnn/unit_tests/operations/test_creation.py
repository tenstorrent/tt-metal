# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, divup


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_zeros_like(device, input_shape):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.zeros_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_ones_like(device, input_shape):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.ones_like(input_tensor)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5, 3, 15, 25],
)
def test_full_like(device, input_shape, fill_value):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.full_like(input_tensor, fill_value=fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_ones(device, input_shape):
    torch_tensor = torch.ones(input_shape, dtype=torch.bfloat16)

    tensor = ttnn.ones(input_shape, device=device)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_zeros(device, input_shape):
    torch_tensor = torch.zeros(input_shape, dtype=torch.bfloat16)

    tensor = ttnn.zeros(input_shape, device=device)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
def test_full(device, input_shape, fill_value):
    torch_tensor = torch.full(input_shape, dtype=torch.bfloat16, fill_value=fill_value)

    tensor = ttnn.full(input_shape, device=device, fill_value=fill_value)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "start",
    [4, 8, 16, 32],
)
@pytest.mark.parametrize(
    "end",
    [100, 200, 300],
)
@pytest.mark.parametrize(
    "step",
    [1, 2, 3, 4, 5],
)
def test_arange(device, start, end, step):
    torch_input_tensor = torch.rand((start, end, step), dtype=torch.bfloat16)
    torch_output_tensor = torch.arange(start, end, step)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.arange(
        input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], ttnn.bfloat16, device
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor[-1, -1, -1, :]
    if divup((end - start), step) % 2 != 0:
        output_tensor = output_tensor[:-1]

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1, 4, 4],  # 256x256
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
    ],
)
def test_empty(device, input_shapes):
    torch_input_tensor = torch.rand((input_shapes), dtype=torch.bfloat16)
    torch_output_tensor = torch.empty(torch_input_tensor.shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.empty(input_tensor.shape, device=device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert list(torch_output_tensor.shape) == list(output_tensor.shape)
