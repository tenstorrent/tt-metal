# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize("input_shapes", ([[1, 1, 32, 32]], [[1, 1, 256, 256]]))
@pytest.mark.parametrize("dtype", (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B))
@pytest.mark.parametrize(
    "memory_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.SINGLE_BANK, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.SINGLE_BANK, ttl.tensor.BufferType.L1),
    ),
)
def test_run_datacopy_test(input_shapes, device, dtype, memory_config, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
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
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
            ],
            "output_mem_config": memory_config,
        },
    )
