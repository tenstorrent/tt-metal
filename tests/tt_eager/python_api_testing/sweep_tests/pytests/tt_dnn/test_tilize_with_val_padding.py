# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

params = [
    pytest.param([[5, 5, 50, 50]], tilize_with_val_padding_args)
    for tilize_with_val_padding_args in generation_funcs.gen_tilize_with_val_padding_args([[5, 5, 50, 50]])
]
params += [
    pytest.param([[5, 5, 64, 96]], tilize_with_val_padding_args)
    for tilize_with_val_padding_args in generation_funcs.gen_tilize_with_val_padding_args([[5, 5, 64, 96]])
]
params += [
    pytest.param(
        [[1, 1, 120, 7300]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 128, 7328],
            "pad_value": 10,
        },
    )
]


@pytest.mark.parametrize("input_shapes, tilize_with_val_padding_args", params)
def test_run_tilize_with_val_padding_test(input_shapes, tilize_with_val_padding_args, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "tilize_with_val_padding",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        tilize_with_val_padding_args,
    )
