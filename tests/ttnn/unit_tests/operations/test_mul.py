# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


# fmt: off
@pytest.mark.parametrize("scalar", [3.0])
# fmt: on
def test_multiply_not_4D(device, scalar):
    torch_input_tensor_a = torch.arange(32).to(dtype=torch.bfloat16)
    torch_input_tensor_b = torch.arange(32).to(dtype=torch.bfloat16)

    torch_output_tensor = torch_input_tensor_a * torch_input_tensor_b * scalar

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = input_tensor_a * input_tensor_b * scalar
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_mul_4D(device, h, w):
    torch_input_tensor_a = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# fmt: off
@pytest.mark.parametrize("scalar", [3.0, 0.125])
# fmt: on
def test_multiply_with_scalar_and_tile_layout(device, scalar):
    torch_input_tensor_a = torch.arange(32 * 32).reshape(32, 32).to(dtype=torch.bfloat16)
    torch_output_tensor = scalar * torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    output = scalar * input_tensor_a
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.skip(reason="Unable to multiply scalar to tensor with int")
# fmt: off
@pytest.mark.parametrize("input_a,scalar", [
        ([13, 16, 42, 42], 0.125)
    ])
# fmt: on
def test_multiply_int32_with_scalar(device, input_a, scalar):
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.int32)
    torch_output_tensor = scalar * torch_input_tensor_a
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    output = scalar * input_tensor_a
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
