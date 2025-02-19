# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
import ttnn
from models.utility_functions import is_blackhole


def create_grid(x, y):
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))
    return ttnn.CoreRangeSet({core_range})


params = [
    pytest.param(
        [[5, 5, 32, 32]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_end": [4, 4, 31, 28],
        },
    )
]

params += [
    pytest.param(
        [[5, 5, 64, 96]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_end": [4, 4, 60, 90],
        },
    )
]

params += [
    pytest.param(
        [[5, 5, 64, 96]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_end": [4, 4, 60, 90],
        },
    )
]


params += [
    pytest.param(
        [[1, 1, 128, 7328]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_end": [0, 0, 119, 7299],
        },
    )
]

input_x = 64 if is_blackhole() else 32

params += [
    pytest.param(
        [[1, 1, 128, input_x]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(create_grid(1, 2), [64, input_x], ttnn.ShardOrientation.ROW_MAJOR),
                )
            ],
            "output_mem_config": ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(create_grid(1, 2), [64, input_x // 2], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            "output_tensor_start": [0, 0, 0, 0],
            "output_tensor_end": [0, 0, 127, (input_x // 2) - 1],
        },
    )
]


# @skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("input_shapes, untilize_with_unpadding_args", params)
def test_run_untilize_with_unpadding_test(input_shapes, untilize_with_unpadding_args, device):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16, True
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "untilize_with_unpadding",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        untilize_with_unpadding_args,
    )
