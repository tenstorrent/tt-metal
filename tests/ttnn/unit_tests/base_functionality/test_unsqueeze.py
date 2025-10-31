# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ((1, 1, 253), 2, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -2, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 253), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -2, ttnn.TILE_LAYOUT),
        ((1, 253), 1, ttnn.TILE_LAYOUT),
        ((57, 83), 1, ttnn.TILE_LAYOUT),
        ((123, 259), -2, ttnn.TILE_LAYOUT),
        ((57, 83), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((123, 259), -2, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), -1, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), 0, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_unsqueeze(device, input_shape, dim, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_unsqueeze_tensor = torch.unsqueeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    ttnn_output = ttnn.unsqueeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_unsqueeze_tensor)


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ((1, 1, 253), 4, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -5, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_invalid_cases(device, input_shape, dim, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    with pytest.raises(RuntimeError):
        ttnn.unsqueeze(input_tensor, dim)
