# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("from_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("to_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_to_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    input_tensor = ttnn.to_dtype(input_tensor, from_dtype)
    assert input_tensor.dtype == from_dtype
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(input_tensor.shape) == (height, width)

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)
    assert output_tensor.dtype == to_dtype
    if to_dtype == ttnn.bfloat8_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(output_tensor.shape) == (height, width)

    output_tensor = ttnn.to_torch(output_tensor).to(torch_input_tensor.dtype)

    assert_with_pcc(torch_input_tensor, output_tensor)
