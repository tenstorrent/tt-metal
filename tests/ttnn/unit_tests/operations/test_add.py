# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [2 * 32])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == (size,)


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_add_scalar(device, s, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + s

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = input_tensor + s
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("alpha", [0.42])
@pytest.mark.parametrize("scalar_input_tensor_b", [0.5])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_add_scalar_and_alpha(device, alpha, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor, scalar_input_tensor_b, alpha=alpha)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.add(input_tensor, scalar_input_tensor_b, alpha=alpha)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_add(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_add_4D(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
@pytest.mark.parametrize("scalar", [0.42])
def test_add_scalar(device, h, w, scalar):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    output = scalar + input_tensor_a
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_add_4D(device, h, w):
    torch_input_tensor_a = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_add_broadcasts(device, h, w):
    torch_a = torch.rand((2, 16, 1, w), dtype=torch.bfloat16)
    torch_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output = torch.add(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.to_device(b, device)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output = torch.add(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.to_device(b, device)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.skip(reason="4005: Unable to broadcast on batch or seq dimension")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_add_broadcasts_on_batch(device, h, w):
    torch_a = torch.rand((1, 16, 1, w), dtype=torch.bfloat16)
    torch_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output = torch.add(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.to_device(b, device)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
