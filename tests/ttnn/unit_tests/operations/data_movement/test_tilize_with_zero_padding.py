# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests.ttnn_pytorch_ops import tilize_with_zero_padding
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input

shapes = [[[1, 1, 30, 32]], [[3, 1, 315, 384]], [[1, 1, 100, 7104]]]


@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize(
    "tilize_with_zero_padding_args",
    (
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    ),
)
def test_tilize_with_zero_padding_test(input_shapes, tilize_with_zero_padding_args, device, function_level_defaults):
    shape = input_shapes[0]
    torch_input = (torch.rand(shape) * 200 - 100).to(torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=tilize_with_zero_padding_args["dtype"][0],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=tilize_with_zero_padding_args["input_mem_config"][0],
    )
    tt_output = ttnn.tilize_with_zero_padding(
        tt_input, memory_config=tilize_with_zero_padding_args["output_mem_config"]
    )
    torch_output = tt_output.cpu().to_torch_with_padded_shape()

    torch_golden = tilize_with_zero_padding(torch_input)
    assert_equal(torch_golden, torch_output)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.tilize_with_zero_padding))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.tilize_with_zero_padding))
@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 3, 50, 70),  # pads both of the last two dimensions
        (2, 50, 64),  # pads only the second-to-last dimension
        (1, 1, 40, 2000),  # a last dimension of more than 1024
        (2, 2, 2, 50, 50),  # rank 5
        (100,),  # rank 1
    ],
    ids=["height_and_width", "height", "wide", "rank5", "rank1"],
)
def test_tilize_with_zero_padding_docstring_supported_inputs(input_shape, dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.tilize_with_zero_padding docstring lists.

    The data must come back unchanged. For inputs of rank 2 or more, the padded shape must be the input's with the last
    two dimensions rounded up to multiples of 32, and the padding must hold zeros. BFLOAT16 inputs leave out NaN, as the
    docstring requires, and compare zeros by value.
    """
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    input_tensor = ttnn.from_torch(
        full_range_input(input_shape, dtype, nan=dtype != ttnn.bfloat16),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    output = ttnn.tilize_with_zero_padding(input_tensor)

    assert output.layout == ttnn.TILE_LAYOUT
    assert output.dtype == dtype
    assert output.memory_config() == memory_config
    assert list(output.shape) == list(input_shape)
    input_values = ttnn.to_torch(input_tensor).to(torch.float64)
    signed_zeros = dtype != ttnn.bfloat16
    if len(input_shape) < 2:
        # The docstring states the padded shape only for inputs of rank 2 or more.
        assert_exact(input_values, ttnn.to_torch(output), signed_zeros=signed_zeros)
        return
    padded_shape = [*input_shape[:-2], *((size + 31) // 32 * 32 for size in input_shape[-2:])]
    assert list(output.padded_shape) == padded_shape
    expected = torch.zeros(padded_shape, dtype=torch.float64)
    expected[tuple(slice(0, size) for size in input_shape)] = input_values
    assert_exact(expected, ttnn.from_device(output).to_torch_with_padded_shape(), signed_zeros=signed_zeros)
