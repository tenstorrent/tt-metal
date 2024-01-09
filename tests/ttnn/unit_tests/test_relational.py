# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# gt
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_gt_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.gt(torch_input_tensor, scalar_input_tensor_b)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.gt(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_gt(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.gt(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.gt(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# gte
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_gte_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor >= scalar_input_tensor_b

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.gte(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_gte(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a >= torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.gte(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# eq
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_eq_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.eq(torch_input_tensor, scalar_input_tensor_b)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.eq(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_eq(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.eq(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.eq(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# lt
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_lt_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.lt(torch_input_tensor, scalar_input_tensor_b)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.lt(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_lt(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.lt(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.lt(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# ne
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_ne_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.ne(torch_input_tensor, scalar_input_tensor_b)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.ne(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_ne(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.ne(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.ne(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# lte
@pytest.mark.parametrize("scalar_input_tensor_b", [-6.0])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_lte_scalar(device, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.randint(-10, 10, size=(h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor <= scalar_input_tensor_b

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.lte(scalar_input_tensor_b, input_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_lte(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a <= torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output = ttnn.lte(input_tensor_a, input_tensor_b)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
