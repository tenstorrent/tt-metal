# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_sub_scalar(device, s, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor - s

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = input_tensor - s
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize("alpha", [0.42])
@pytest.mark.parametrize("scalar_input_tensor_b", [0.5])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_sub_scalar_and_alpha(device, alpha, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sub(torch_input_tensor, scalar_input_tensor_b, alpha=alpha)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.sub(input_tensor, scalar_input_tensor_b, alpha=alpha)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def test_subtract(device):
    a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    b = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    output = ttnn.sub(a, b, alpha=2)
    assert_with_pcc(torch.tensor((1, 0)), ttnn.to_torch(ttnn.from_device(output)), 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_sub(device, h, w):
    torch_a = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch.sub(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)
    a = ttnn.to_device(a, device)
    b = ttnn.to_device(b, device)
    tt_output = ttnn.sub(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("n", [32])
@pytest.mark.parametrize("c", [2 * 32])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_sub_4D(device, n, c, h, w):
    torch_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output = torch.sub(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)
    a = ttnn.to_device(a, device)
    b = ttnn.to_device(b, device)

    tt_output = ttnn.sub(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
