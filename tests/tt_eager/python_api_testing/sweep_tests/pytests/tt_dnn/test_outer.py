# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test

shapes = [
    [[1, 1, 1, 32], [1, 1, 1, 32]],
    [[1, 1, 1, 2048], [1, 1, 1, 64]],  # LLaMa dimensions
]


@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize("dtype", (ttnn.bfloat16,))
def test_run_outer_test(input_shapes, device, dtype, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "outer",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
            "input_mem_config": [
                ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    )
