# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# fmt: off
@pytest.mark.parametrize("m,k,n", [
    (1, 2, 2),
    (1, 2, 4),
    (1, 4, 2),
    (1, 4, 4),
    (3, 2, 2),
    (3, 2, 4),
    (3, 4, 2),
    (3, 4, 4),
])
# fmt: on
def test_matmul_with_matched_width_height(device, m, k, n):
    torch_input_tensor_a = torch.rand((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k, n), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("k, n", [
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    ])
# fmt: on
def test_matmul_with_matched_width_height_from_1D(device, k, n):
    torch_input_tensor_a = torch.rand((k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k, n), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("w", [(4), (2)])
def test_matmul_does_dot_product(device, w):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_input_tensor_b = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.from_device(tt_output)

    tt_output = ttnn.to_torch(tt_output)

    assert torch_output.shape == ()
    assert tt_output.shape == (32,)
    assert torch.allclose(torch_output, tt_output[0], atol=1e-2)


# fmt: off
@pytest.mark.parametrize("n,c,h,w", [
    (1, 1, 2, 4),
    (1, 1, 4, 2),
    (3, 3, 2, 4),
    (3, 3, 4, 2),
    (1, 3, 2, 4),
    (3, 1, 4, 2),
    ])
# fmt: on
def test_matmul_with_matched_width_height_4D(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, w, h), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("n,c,h,w", [
    (1, 1, 2, 2),
    (1, 1, 4, 4),
    (3, 3, 4, 4),
    (3, 1, 4, 4),
    (1, 3, 4, 4)
    ])
# fmt: on
def test_matmul_same_shape_and_valid(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("input_a,input_b", [
        ([1.0,2.0,3.0],[3.0,4.0,5.0])
    ])
# fmt: on
def test_matmul_same_shape_but_invalid(device, input_a, input_b):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    input_b += [0.0] * (32 - len(input_b))

    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_a)))
    torch_input_tensor_b = torch.as_tensor(input_b, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_b)))

    with pytest.raises(RuntimeError) as exception:
        torch.matmul(torch_input_tensor_a, torch_input_tensor_b)
    assert "Expected size for first two dimensions of batch2 tensor to be: [1, 32] but got: [1, 1]." in str(
        exception.value
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    with pytest.raises(RuntimeError) as exception:
        ttnn.matmul(input_tensor_a, input_tensor_b)
    assert "The width of the first tensor must be equal to the height of the second tensor" in str(exception.value)


def test_tutorial_matmul(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output = torch_a @ torch_b

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device)
    b = ttnn.to_device(b, device)

    output = a @ b
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.999)


def test_tutorial_matmul_with_tilized_inputs(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output = torch_a @ torch_b

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device)
    b = ttnn.to_device(b, device)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output = a @ b
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.999)


def test_tutorial_matmul_with_tilized_inputs_in_l1_memory(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output = torch_a @ torch_b

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.999)


def test_tutorial_matmul_with_tilized_input_in_l1_memory_and_user_specified_core_grid(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output = torch_a @ torch_b

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=(8, 8))
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.999)
