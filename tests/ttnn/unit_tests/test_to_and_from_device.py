# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [8])  # must be even to be put on device
def test_to_and_from_4D(device, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    tensor = ttnn.to_device(tensor, device)
    tensor = ttnn.from_device(tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [4])
def test_to_and_from_2D(device, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    # when w=3->size 40 & page size 6 (bad)
    # when w=4-> size 56 & page size 8  (ok)
    tensor = ttnn.to_device(tensor, device)
    tensor = ttnn.from_device(tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


# fmt: off
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.uint32), (torch.float32, ttnn.bfloat16)])
# fmt: on
def test_to_and_from_device(device, torch_dtype, ttnn_dtype):
    # Notice that the width of these tensors are even!
    torch_input_tensor = torch.as_tensor([0, 1, 2, 3], dtype=torch_dtype)
    tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype)
    tensor = ttnn.to_device(tensor, device)
    tensor = ttnn.from_device(tensor)
    torch_output_tensor = ttnn.to_torch(tensor).to(torch_dtype)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.skip(reason="4359: from_device is hanging")
@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("c", [8])
@pytest.mark.parametrize("h", [1500])
@pytest.mark.parametrize("w", [64])
def test_from_device_hang(device, b, c, h, w):
    torch_input_tensor = torch.rand((b, c, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.from_device(output_tensor)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("c", [8])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_to_and_from_multiple_times(device, b, c, h, w):
    tensor = torch.rand((b, c, h, w), dtype=torch.bfloat16)
    original_tensor = tensor
    for i in range(0, 50):
        tensor = ttnn.from_torch(tensor)
        tensor = ttnn.to_device(tensor, device)
        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        tensor = ttnn.from_device(tensor)
        tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        tensor = ttnn.to_torch(tensor)

    assert torch.allclose(original_tensor, tensor)
