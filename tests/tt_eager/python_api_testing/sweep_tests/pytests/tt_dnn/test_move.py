# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import ttnn


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test

shapes = [
    [[1, 1, 1, 30]],  # Single core
    [[1, 1, 300, 380]],  # multi core
    [[1, 3, 320, 380]],  # multi core
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384]],  # Multi core
]
mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize("input_mem_config", mem_configs, ids=["input_DRAM", "input_L1"])
@pytest.mark.parametrize("output_mem_config", mem_configs, ids=["output_DRAM", "output_L1"])
def test_run_move_op(
    input_shapes,
    input_mem_config,
    output_mem_config,
    device,
    function_level_defaults,
):
    if input_mem_config != ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1):
        pytest.skip("Only inputs on L1 for move op currently supported")

    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"input_mem_config": [input_mem_config], "output_mem_config": output_mem_config})
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "move",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
    )
