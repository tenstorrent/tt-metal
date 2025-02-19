# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from math import prod

import pytest
import torch
import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

layouts = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]

dtypes = [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)]
shapes = [(1,), (2,), (2, 3), (4, 16, 3, 1), (4, 3, 1, 2, 2)]
repeat_shapes = [
    (1,),
    (1, 2),
    (4, 3, 2, 1),
    (2, 3, 4, 5, 2),
    (2048,),
]


def _get_size(larger, smaller) -> int:
    return prod([a * b for a, b in zip(((1,) * (len(larger) - len(smaller)) + smaller), larger)])


def _get_final_size(shape, reshape):
    if len(shape) > len(reshape):
        return _get_size(shape, reshape)
    else:
        return _get_size(reshape, shape)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_repeat(device, layout, dtype, shape, repeat_shape):
    torch_dtype, ttnn_dtype = dtype

    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")

    if layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal config")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=torch_dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=ttnn_dtype)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)
    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_pc_repeat(device, layout, shape, repeat_shape):
    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")
    device.enable_program_cache()
    num_iters = 3
    input_tensors = []
    torch_results = []
    for i in range(3):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch_tensor.repeat(repeat_shape))
        input_tensors.append(ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16))
    for i in range(3):
        output = ttnn.repeat(input_tensors[i], ttnn.Shape(repeat_shape))
        output = ttnn.to_torch(output)
        assert (
            output.shape == torch_results[i].shape
        ), f"Output shape {output.shape} does not match torch shape {torch_results[i].shape}"

        assert_with_pcc(torch_results[i], output, 0.9999)
