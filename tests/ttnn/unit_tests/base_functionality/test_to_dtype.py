# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype
from tests.ttnn.utils_for_testing import assert_with_pcc

bfloat4_pcc = 0.960
torch.manual_seed(0)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
FLOAT_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if is_ttnn_float_type(dtype)]


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", FLOAT_TYPES)
@pytest.mark.parametrize("from_dtype", FLOAT_TYPES)
def test_to_float_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.rand((height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [36])
@pytest.mark.parametrize("width", [36])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_unaligned_shape(height, width, from_dtype, to_dtype):
    if (
        from_dtype == ttnn.bfloat4_b
        or from_dtype == ttnn.bfloat8_b
        or to_dtype == ttnn.bfloat4_b
        or to_dtype == ttnn.bfloat8_b
    ):
        pytest.skip("bfloat4_b and bfloat8_b require align shape divisible by tile")

    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_with_tile_layout(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    assert input_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)
