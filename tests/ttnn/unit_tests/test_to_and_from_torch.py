# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_to_and_from_4D(height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_to_and_from_2D(height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)
