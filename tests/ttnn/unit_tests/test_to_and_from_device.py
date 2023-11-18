# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [8])  # must be even to be put on device
def test_to_and_from_4D(device, h, w):
    torch_input = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    tt_output = ttnn.from_torch(torch_input)
    tt_output = ttnn.to_device(tt_output, device)
    tt_output = ttnn.from_device(tt_output)
    torch_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, torch_input)


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [4])
def test_to_and_from_2D(device, h, w):
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    tt_output = ttnn.from_torch(torch_input)
    # when w=3->size 40 & page size 6 (bad)
    # when w=4-> size 56 & page size 8  (ok)
    tt_output = ttnn.to_device(tt_output, device)
    tt_output = ttnn.from_device(tt_output)
    torch_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, torch_input)


# fmt: off
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.uint32), (torch.float32, ttnn.bfloat16)])
# fmt: on
def test_to_and_from_device(device, torch_dtype, ttnn_dtype):
    # Notice that the width of these tensors are even!
    torch_input = torch.as_tensor([0, 1, 2, 3], dtype=torch_dtype)
    tt_output = ttnn.from_torch(torch_input, dtype=ttnn_dtype)
    tt_output = ttnn.to_device(tt_output, device)
    tt_output = ttnn.from_device(tt_output)
    torch_output = ttnn.to_torch(tt_output).to(torch_dtype)
    assert torch.allclose(torch_output, torch_input)
