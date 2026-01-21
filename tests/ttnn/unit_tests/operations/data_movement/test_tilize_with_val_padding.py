# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

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

# WIDTH_SHARDED test - existing functionality
params += [
    pytest.param(
        [[1, 1, 30, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(30, 64),
                    core_grid=ttnn.CoreGrid(y=1, x=2),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(32, 64),
                core_grid=ttnn.CoreGrid(y=1, x=2),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            "output_tensor_shape": [1, 1, 32, 64],
            "pad_value": 0.0,
        },
        id="width_sharded_height_padding",
    )
]

# HEIGHT_SHARDED tests - new functionality

# HEIGHT_SHARDED: Pad height (main use case) - 4 cores
params += [
    pytest.param(
        [[1, 1, 96, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(96, 64),  # Global shape
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 64),  # Global shape - padded height 96→128, shard=32×64
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 64],
            "pad_value": 10.0,  # Non-zero to verify padding correctness
        },
        id="height_sharded_pad_height_4cores",
    )
]

# HEIGHT_SHARDED: Pad width - 2 cores
params += [
    pytest.param(
        [[1, 1, 64, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(64, 30),  # Global shape
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(64, 32),  # Global shape - padded width 30→32
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 64, 32],
            "pad_value": -3.0,
        },
        id="height_sharded_pad_width_2cores",
    )
]

# HEIGHT_SHARDED: Pad both height and width - 2 cores, tile-aligned
params += [
    pytest.param(
        [[1, 1, 64, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(64, 50),  # Global shape - tile-aligned height
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(64, 64),  # Global shape - padded width 50→64
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 5.5,
        },
        id="height_sharded_pad_both_dims_2cores",
    )
]

# HEIGHT_SHARDED: Non-contiguous core grid with tile-aligned shards
params += [
    pytest.param(
        [[1, 1, 96, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(32, 30),  # Shard shape - 32 rows per core (tile-aligned)
                    core_grid=ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                            ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                        }
                    ),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(32, 32),  # Shard shape - padded width 30→32
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            "output_tensor_shape": [1, 1, 96, 32],
            "pad_value": -7.5,
        },
        id="height_sharded_noncontiguous_grid",
    )
]

# HEIGHT_SHARDED: 8 cores
params += [
    pytest.param(
        [[1, 1, 224, 32]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(224, 32),  # Global shape
                    core_grid=ttnn.CoreGrid(y=8, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(256, 32),  # Global shape - padded height 224→256, shard=32×32
                core_grid=ttnn.CoreGrid(y=8, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 256, 32],
            "pad_value": 0.0,
        },
        id="height_sharded_pad_height_8cores",
    )
]

# HEIGHT_SHARDED: COL_MAJOR orientation - tests mapping table traversal
params += [
    pytest.param(
        [[1, 1, 96, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(96, 30),  # Global shape
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.COL_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 32),  # Global shape - pad both: 96→128, 30→32, shard=32×32
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 32],
            "pad_value": 12.0,
        },
        id="height_sharded_col_major_orientation",
    )
]

# HEIGHT_SHARDED: 2D grid (2×2) - tests physical mapping across 2D grid
params += [
    pytest.param(
        [[1, 1, 128, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 50),  # Global shape
                    core_grid=ttnn.CoreGrid(y=2, x=2),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 64),  # Global shape - padded width 50→64
                core_grid=ttnn.CoreGrid(y=2, x=2),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 64],
            "pad_value": -1.0,
        },
        id="height_sharded_2d_grid",
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
