# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


# fmt: off
@pytest.mark.parametrize("input_a,input_b,scalar", [
        ([1.0,2.0,3.0],[1.0,2.0,3.0],3.0)
    ])
# fmt: on
def test_multiply_not_4D(device, input_a, input_b, scalar):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    input_b += [0.0] * (32 - len(input_b))
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, len(input_a)))
    torch_input_tensor_b = torch.as_tensor(input_b, dtype=torch.bfloat16).reshape((1, len(input_b)))

    torch_output = torch_input_tensor_a * torch_input_tensor_b * scalar

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    tt_output = input_tensor_a * input_tensor_b * scalar
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_mul_4D(device, h, w):
    torch_a = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_output = torch.mul(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.to_device(b, device)
    tt_output = ttnn.mul(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


# fmt: off
@pytest.mark.parametrize("input_a,scalar", [
        ([0, 1.0,2.0,3.0],3.0),
        ([13, 16, 42, 42], 0.125)
    ])
# fmt: on
def test_multiply_with_scalar_and_tile_layout(device, input_a, scalar):
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16)
    torch_input_tensor_a = torch_input_tensor_a.reshape(1, 1, 1, 4)
    padding_needed = (32 - (torch_input_tensor_a.shape[-1] % 32)) % 32
    torch_input_tensor_a = F.pad(torch_input_tensor_a, (0, padding_needed, 0, 31, 0, 0, 0, 0))
    torch_output = scalar * torch_input_tensor_a
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    tt_output = scalar * input_tensor_a
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.skip(reason="Unable to multiply scalar to tensor with int")
# fmt: off
@pytest.mark.parametrize("input_a,scalar", [
        ([13, 16, 42, 42], 0.125)
    ])
# fmt: on
def test_multiply_int32_with_scalar(device, input_a, scalar):
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.int32)
    torch_output = scalar * torch_input_tensor_a
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    tt_output = scalar * input_tensor_a
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)
