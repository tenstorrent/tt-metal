# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 1, 256), 2),
        ((1, 1, 256), -2),
        ((1, 256), 1),
        ((1, 1, 30), 2),
        ((1, 1, 30), -2),
        ((1, 30), 1),
    ],
)
def test_unsqueeze(device, input_shape, dim):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_unsqueeze_tensor = torch.unsqueeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.unsqueeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_unsqueeze_tensor)
