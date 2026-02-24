# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from tests.ttnn.utils_for_testing import assert_equal
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import (
    tilize_with_val_padding as pytorch_tilize_with_val_padding,
)

torch.manual_seed(0)

params = [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 10.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 10.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 50.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": -18.0,
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


@pytest.mark.parametrize("input_shape", [(32, 15916), (16, 5210112), (48, 5210112), (180, 5210116)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)


# nd_sharded_sweep_params = [
#     pytest.param(
#         [[1, 3, 50, 96]],
#         {
#             "dtype": [ttnn.bfloat16],
#             "layout": [ttnn.ROW_MAJOR_LAYOUT],
#             "input_mem_config": [
#                 ttnn.MemoryConfig(
#                     buffer_type=ttnn.BufferType.L1,
#                     nd_shard_spec=ttnn.NdShardSpec(
#                         shard_shape=[1, 1, 50, 96],
#                         grid=ttnn.CoreRangeSet(
#                             {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}
#                         ),
#                         orientation=ttnn.ShardOrientation.ROW_MAJOR,
#                     ),
#                 )
#             ],
#             "output_mem_config": ttnn.MemoryConfig(
#                 buffer_type=ttnn.BufferType.L1,
#                 nd_shard_spec=ttnn.NdShardSpec(
#                     shard_shape=[1, 1, 64, 96],
#                     grid=ttnn.CoreRangeSet(
#                         {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}
#                     ),
#                     orientation=ttnn.ShardOrientation.ROW_MAJOR,
#                 ),
#             ),
#             "output_tensor_shape": [1, 3, 64, 96],
#             "pad_value": 0.0,
#         },
#     ),
#     pytest.param(
#         [[3, 100, 158]],
#         {
#             "dtype": [ttnn.bfloat16],
#             "layout": [ttnn.ROW_MAJOR_LAYOUT],
#             "input_mem_config": [
#                 ttnn.MemoryConfig(
#                     buffer_type=ttnn.BufferType.L1,
#                     nd_shard_spec=ttnn.NdShardSpec(
#                         shard_shape=[2, 64, 96],
#                         grid=ttnn.CoreRangeSet(
#                             {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}
#                         ),
#                         orientation=ttnn.ShardOrientation.ROW_MAJOR,
#                     ),
#                 )
#             ],
#             "output_mem_config": ttnn.MemoryConfig(
#                 buffer_type=ttnn.BufferType.L1,
#                 nd_shard_spec=ttnn.NdShardSpec(
#                     shard_shape=[3, 96, 96],
#                     grid=ttnn.CoreRangeSet(
#                         {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}
#                     ),
#                     orientation=ttnn.ShardOrientation.ROW_MAJOR,
#                 ),
#             ),
#             "output_tensor_shape": [4, 128, 160],
#             "pad_value": 0.0,
#         },
#     ),
# ]


# @pytest.mark.parametrize("input_shapes, tilize_with_val_padding_args", nd_sharded_sweep_params)
# def test_run_tilize_with_val_padding_nd_sharded_sweep(
#     input_shapes, tilize_with_val_padding_args, device, function_level_defaults
# ):
#     datagen_func = [
#         generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
#     ]
#     comparison_func = comparison_funcs.comp_equal
#     run_single_pytorch_test(
#         "tilize_with_val_padding",
#         input_shapes,
#         datagen_func,
#         comparison_func,
#         device,
#         tilize_with_val_padding_args,
#    )


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, output_padded_shape, output_shard_shape, shard_core_grid",
    [
        (
            [3, 50, 96],
            [2, 50, 96],
            [3, 64, 96],
            [1, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [3, 100, 158],
            [2, 64, 96],
            [3, 128, 160],
            [2, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [4, 100, 160],
            [2, 100, 160],
            [4, 128, 160],
            [2, 128, 160],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [3, 100, 158],
            [2, 64, 96],
            [4, 128, 160],
            [3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [5, 3, 100, 158],
            [4, 2, 64, 96],
            [8, 4, 128, 160],
            [5, 3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2, 0.0])
def test_tilize_with_val_padding_nd_sharded(
    device, tensor_shape, input_shard_shape, output_padded_shape, output_shard_shape, shard_core_grid, pad_value
):
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor,
        output_padded_shape,
        pad_value,
        memory_config=output_memory_config,
    )

    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()  # ttnn.to_torch(ttnn_output_tensor)
    print(input_torch_tensor.shape)
    print(output_torch_tensor.shape)
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    print(expected_torch_tensor.shape)
    assert_equal(expected_torch_tensor, output_torch_tensor)
