# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from models.utility_functions import is_wormhole_b0

shape1 = [
    [[1, 1, 32, 32], [1, 1, 1, 32]],  # Single core
    [[1, 1, 3840, 32], [1, 1, 1, 32]],  # Multi core h
    [[1, 3, 3840, 32], [1, 1, 1, 32]],  # Multi core h
    [[1, 3, 3840, 32], [1, 3, 1, 32]],  # Multi core h
]


@pytest.mark.parametrize(
    "input_shapes",
    shape1,
)
@pytest.mark.parametrize("bcast_op_type", ("add", "sub", "mul"))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
def test_run_bcast_h_test(input_shapes, bcast_op_type, dtype, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        f"bcast-{bcast_op_type}-h",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)] * 2,
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    )


shape2 = [
    [[1, 1, 32, 32], [1, 1, 32, 1]],  # Single core
    [[1, 1, 32, 3840], [1, 1, 32, 1]],  # Multi core w
    [[1, 3, 32, 3840], [1, 1, 32, 1]],  # Multi core w
    [[1, 3, 32, 3840], [1, 3, 32, 1]],  # Multi core w
]
if is_wormhole_b0():
    del shape2[1:]


@pytest.mark.parametrize("input_shapes", shape2)
@pytest.mark.parametrize("bcast_op_type", ("add", "sub", "mul"))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
def test_run_bcast_w_test(input_shapes, bcast_op_type, dtype, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        f"bcast-{bcast_op_type}-w",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)] * 2,
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    )


shapes3 = [
    [[1, 1, 32, 32], [1, 1, 1, 1]],  # Single core
    [[1, 1, 320, 384], [1, 1, 1, 1]],  # Multi core hw
    [[1, 3, 320, 384], [1, 1, 1, 1]],  # Multi core hw
    [[1, 3, 320, 384], [1, 3, 1, 1]],  # Multi core hw
]
if is_wormhole_b0():
    del shapes3[1:]


@pytest.mark.parametrize("input_shapes", shapes3)
@pytest.mark.parametrize("bcast_op_type", ("add", "sub", "mul"))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
def test_run_bcast_hw_test(input_shapes, bcast_op_type, dtype, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        f"bcast-{bcast_op_type}-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)] * 2,
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        },
    )
