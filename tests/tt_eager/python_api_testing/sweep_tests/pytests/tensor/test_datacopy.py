# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize("input_shapes", ([[1, 1, 32, 32]], [[1, 1, 256, 256]]))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
@pytest.mark.parametrize(
    "memory_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.SINGLE_BANK, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.SINGLE_BANK, ttnn.BufferType.L1),
    ),
)
def test_run_datacopy_test(input_shapes, device, dtype, memory_config, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]

    if dtype == ttnn.bfloat8_b:
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
            "layout": [ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": memory_config,
        },
    )
