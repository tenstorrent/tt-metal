# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sub_scalar(device, s, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor - s

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = input_tensor - s
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sub_scalar_float(device, s, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.float)
    torch_output_tensor = torch_input_tensor - s

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = input_tensor - s
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsub_scalar(device, s, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = s - torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # TODO : add Tensor.__rsub__ eventually
    output_tensor = ttnn.mul(input_tensor, -1.0)
    output_tensor = ttnn.add(output_tensor, s)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsub_scalar_float(device, s, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.float)
    torch_output_tensor = s - torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # TODO : add Tensor.__rsub__ eventually
    output_tensor = ttnn.mul(input_tensor, -1.0)
    output_tensor = ttnn.add(output_tensor, s)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("scalar_input_tensor_b", [0.5])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_sub_scalar_and_alpha(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sub(torch_input_tensor, scalar_input_tensor_b)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.sub(input_tensor, scalar_input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_sub(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sub(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.sub(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [128])
def test_sub_4D(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sub(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.sub(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
