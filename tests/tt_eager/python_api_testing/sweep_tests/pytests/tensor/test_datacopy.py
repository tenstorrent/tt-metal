# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn.deprecated as ttl
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize("input_shapes", ([[1, 1, 32, 32]], [[1, 1, 256, 256]]))
@pytest.mark.parametrize(
    "dtype", (ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.BFLOAT8_B)
)
@pytest.mark.parametrize(
    "memory_config",
    (
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        ),
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
        ),
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.SINGLE_BANK, ttnn.experimental.tensor.BufferType.DRAM
        ),
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.SINGLE_BANK, ttnn.experimental.tensor.BufferType.L1
        ),
    ),
)
def test_run_datacopy_test(input_shapes, device, dtype, memory_config, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]

    if dtype == ttnn.experimental.tensor.DataType.BFLOAT8_B:
        comparison_func = partial(comparison_funcs.comp_allclose, atol=1.0)
    else:
        comparison_func = partial(comparison_funcs.comp_equal)

    run_single_pytorch_test(
        "datacopy",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype],
            "layout": [ttnn.experimental.tensor.Layout.TILE],
            "input_mem_config": [
                ttnn.experimental.tensor.MemoryConfig(
                    ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
                )
            ],
            "output_mem_config": memory_config,
        },
    )
