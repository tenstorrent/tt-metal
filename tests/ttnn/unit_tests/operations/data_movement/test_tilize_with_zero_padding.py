# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

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
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "tilize_with_zero_padding",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        tilize_with_zero_padding_args,
    )
