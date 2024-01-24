# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, update_process_id

import torch.nn.functional as F


@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [2 * 33])
def test_unpad_2D(h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.pad_to_tile(input_tensor)
    output_tensor = ttnn.unpad_from_tile(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_input_tensor, output_tensor)


@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [2 * 33])
def test_unpad_2D_from_device(device, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.pad_to_tile(input_tensor)
    output_tensor = ttnn.unpad_from_tile(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_input_tensor, output_tensor)
