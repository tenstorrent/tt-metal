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
    ],
)
def test_squeeze_as_reshape(device, input_shape, dim):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.squeeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_squeeze_tensor)
