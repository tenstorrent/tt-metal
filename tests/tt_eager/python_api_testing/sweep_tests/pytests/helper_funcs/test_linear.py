# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize(
    "input_shapes",
    ([[1, 1, 32, 64], [1, 1, 256, 64]], [[1, 1, 64, 128], [1, 1, 32, 128]]),
)
def test_linear_no_bias(input_shapes, device):
    comparison_func = partial(comparison_funcs.comp_pcc)

    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32)
    ]
    weight = torch.randn(input_shapes[1])
    run_single_pytorch_test(
        "linear",
        [
            input_shapes[0],
        ],
        datagen_func,
        comparison_func,
        device,
        {
            "weight": weight,
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            "dtype": [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
            "input_mem_config": [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
            ]
            * 2,
            "output_mem_config": ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            ),
            "bias": None,
        },
    )


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[1, 1, 32, 64], [1, 1, 256, 64], [1, 1, 1, 256]],
        [[1, 1, 64, 128], [1, 1, 32, 128], [1, 1, 1, 32]],
    ),
)
def test_linear_with_bias(input_shapes, device):
    comparison_func = partial(comparison_funcs.comp_pcc)

    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32)
    ]
    weight = torch.randn(input_shapes[1])
    bias = torch.randn(input_shapes[2])
    run_single_pytorch_test(
        "linear",
        [
            input_shapes[0],
        ],
        datagen_func,
        comparison_func,
        device,
        {
            "weight": weight,
            "bias": bias,
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.ROW_MAJOR],
            "dtype": [ttl.tensor.DataType.BFLOAT16] * 3,
            "input_mem_config": [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
            ]
            * 3,
            "output_mem_config": ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            ),
        },
    )
