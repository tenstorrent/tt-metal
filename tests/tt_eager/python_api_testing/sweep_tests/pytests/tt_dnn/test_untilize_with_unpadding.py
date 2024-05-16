# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
import tt_lib as ttl


def create_grid(x, y):
    core_range = ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(x - 1, y - 1))
    return ttl.tensor.CoreRangeSet({core_range})


params = [
    pytest.param([[5, 5, 32, 32]], untilize_with_unpadding_args)
    for untilize_with_unpadding_args in generation_funcs.gen_untilize_with_unpadding_args([[5, 5, 32, 32]])
]
params += [
    pytest.param([[5, 5, 64, 96]], untilize_with_unpadding_args)
    for untilize_with_unpadding_args in generation_funcs.gen_untilize_with_unpadding_args([[5, 5, 64, 96]])
]

params += [
    pytest.param(
        [[1, 1, 128, 7328]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
            ],
            "output_mem_config": ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            ),
            "output_tensor_end": [0, 0, 119, 7299],
        },
    )
]


params += [
    pytest.param(
        [[1, 1, 128, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [
                ttl.tensor.MemoryConfig(
                    ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttl.tensor.BufferType.L1,
                    ttl.tensor.ShardSpec(create_grid(1, 2), [64, 32], ttl.tensor.ShardOrientation.ROW_MAJOR, False),
                )
            ],
            "output_mem_config": ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(create_grid(1, 2), [64, 16], ttl.tensor.ShardOrientation.ROW_MAJOR, False),
            ),
            "output_tensor_start": [0, 0, 0, 0],
            "output_tensor_end": [0, 0, 127, 15],
        },
    )
]


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
