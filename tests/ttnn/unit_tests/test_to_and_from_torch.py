# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [3])
def test_to_and_from_4D(h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("h", [7])
@pytest.mark.parametrize("w", [3])
def test_to_and_from_2D(h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)
