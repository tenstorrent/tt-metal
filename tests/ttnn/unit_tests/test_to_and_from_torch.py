# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_to_and_from_4D(height, width):
    torch_input_tensor = torch.rand((1, 1, height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_and_from_2D(height, width, dtype, layout):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b and ROW_MAJOR_LAYOUT not supported")

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    torch_output_tensor = ttnn.to_torch(tensor).to(torch_input_tensor.dtype)

    allclose_kwargs = {}
    if dtype == ttnn.bfloat8_b:
        allclose_kwargs["atol"] = 1e-2
    assert torch.allclose(torch_input_tensor, torch_output_tensor, **allclose_kwargs)
