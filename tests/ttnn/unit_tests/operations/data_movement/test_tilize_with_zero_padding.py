# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests.ttnn_pytorch_ops import tilize_with_zero_padding
from tests.ttnn.utils_for_testing import assert_equal

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
