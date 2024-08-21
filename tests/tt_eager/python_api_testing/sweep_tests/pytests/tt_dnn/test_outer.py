# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test

shapes = [
    # Single core (won't be hit after padding is added for multicast)
    [[1, 1, 32, 1], [1, 1, 1, 32]],  # no reshape
    [[1, 1, 2048, 1], [1, 1, 1, 64]],  # LLaMa dimensions (no reshape)
    # FIXME: padding required for reshape on device
    # [[1, 1, 1, 32], [1, 1, 1, 32]],    #invoke reshape on lhs
    # [[1, 1, 1, 32], [1, 1, 32, 1]],    #invoke reshape on both
    # [[1, 1, 32, 1], [1, 1, 32, 1]],    #invoke reshape on rhs
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
            "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [None, None],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    )
