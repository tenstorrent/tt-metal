# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 1, 1, 256), 2),
        ((1, 1, 1, 256), -1),
        ((1, 1, 1, 30), 2),
        ((1, 1, 1, 30), -1),
        ((1, 32, 16), 0),
        ((1, 1, 24576), 0),
        ((1, 19), 0),
        ((1, 1, 480, 640), 1),
        ((3, 1370, 1, 1, 1280), -2),
        ((3, 197, 1, 1, 1024), -2),
        ((3, 197, 1, 1, 768), -2),
        ((3, 50, 1, 1, 1024), -2),
        ((3, 50, 1, 1, 768), -2),
    ],
)
def test_squeeze(device, input_shape, dim):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.squeeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_squeeze_tensor)
