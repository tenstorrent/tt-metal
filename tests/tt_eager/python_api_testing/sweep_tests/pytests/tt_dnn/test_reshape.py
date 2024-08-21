# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
# TODO: migrate to new sweep infra
#
# import pytest
# import torch
# from functools import partial
#
#
# from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
# from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
# import ttnn
#
# params = [
#    pytest.param([[4, 4, 32, 32]], reshape_args)
#    for reshape_args in generation_funcs.gen_reshape_args([[4, 4, 32, 32]], max_out_shapes=64)
# ]
# params += [
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [-1, 2, 32, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [-1, 2, 32, 64],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [2, -1, 32, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [2, -1, 32, 64],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 2, -1, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 2, -1, 64],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 4, 32, -1],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.TILE_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 2, 32, -1],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [-1, 2, 32, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [2, -1, 32, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 1, -1, 32],
#        },
#    ),
#    pytest.param(
#        [[4, 4, 32, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [4, 2, 32, -1],
#        },
#    ),
#    pytest.param(
#        [[2, 3, 4, 4]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [2, 12, 1, 4],
#        },
#    ),
#    pytest.param(
#        [[1, 3, 6, 4]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [3, 1, 3, 8],
#        },
#    ),
#    pytest.param(
#        [[1, 1, 60, 768]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [1, 60, 12, 64],
#        },
#    ),
#    pytest.param(
#        [[2, 4, 128, 4]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [1, 2, 64, 32],
#        },
#    ),
#    pytest.param(
#        [[1, 11, 64, 2]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [1, 11, 1, 128],
#        },
#    ),
#    pytest.param(
#        [[1, 2, 64, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [2, 4, 128, 4],
#        },
#    ),
#    pytest.param(
#        [[3, 2, 2, 32]],
#        {
#            "dtype": [ttnn.bfloat16],
#            "layout": [ttnn.ROW_MAJOR_LAYOUT],
#            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
#            "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
#            "reshape_dims": [1, 3, 1, -1],
#        },
#    ),
# ]
#
#
# @pytest.mark.parametrize("input_shapes, reshape_args", params)
# def test_run_reshape_test(input_shapes, reshape_args, device, function_level_defaults):
#    datagen_func = [
#        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
#    ]
#    comparison_func = partial(comparison_funcs.comp_equal)
#    run_single_pytorch_test(
#        "reshape",
#        input_shapes,
#        datagen_func,
#        comparison_func,
#        device,
#        reshape_args,
#    )
#
