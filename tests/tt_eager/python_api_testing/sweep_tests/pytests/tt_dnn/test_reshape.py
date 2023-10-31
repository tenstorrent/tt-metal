# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import torch
from pathlib import Path
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0

params = [
    pytest.param([[4, 4, 32, 32]], reshape_args)
    for reshape_args in generation_funcs.gen_reshape_args([[4, 4, 32, 32]], max_out_shapes=64)
]
params += [
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [-1, 2, 32, 32],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [-1, 2, 32, 64],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [2, -1, 32, 32],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [2, -1, 32, 64],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [4, 2, -1, 32],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [4, 2, -1, 64],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [4, 4, 32, -1],
        },
    ),
    pytest.param(
        [[4, 4, 32, 32]],
        {
            "dtype": [ttl.tensor.DataType.BFLOAT16],
            "layout": [ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)],
            "output_mem_config": ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
            "reshape_dims": [4, 2, 32, -1],
        },
    ),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize("input_shapes, reshape_args", params)
def test_run_reshape_test(
    input_shapes, reshape_args, device, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "reshape",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        reshape_args,
    )
