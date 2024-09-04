# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (2,),
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6, 7),
        (2, 3, 4, 5, 6, 7, 8),
        (2, 3, 4, 5, 6, 7, 8, 9),
    ],
)
def test_to_and_from(shape):
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6, 7),
        (2, 3, 4, 5, 6, 7, 8),
        (2, 3, 4, 5, 6, 7, 8, 9),
    ],
)
def test_to_and_from_using_tile_layout(shape):
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.int32, ttnn.uint8])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_and_from_2D(height, width, dtype, layout):
    if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("ROW_MAJOR_LAYOUT not supported for bfloat8_b and bfloat4_b")

    if dtype == ttnn.uint8:
        torch_input_tensor = torch.randint(0, 255, (height, width)).to(torch.uint8)
    elif dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-100, 100, (height, width)).to(torch.int32)
    else:
        torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    torch_output_tensor = ttnn.to_torch(tensor).to(torch_input_tensor.dtype)

    allclose_kwargs = {}
    if dtype == ttnn.bfloat4_b:
        allclose_kwargs["atol"] = 0.01
        allclose_kwargs["rtol"] = 0.3
        assert_with_pcc(torch_input_tensor, torch_output_tensor, pcc=0.9)
    else:
        if dtype == ttnn.bfloat8_b:
            allclose_kwargs["atol"] = 1e-2
        assert torch.allclose(torch_input_tensor, torch_output_tensor, **allclose_kwargs)
