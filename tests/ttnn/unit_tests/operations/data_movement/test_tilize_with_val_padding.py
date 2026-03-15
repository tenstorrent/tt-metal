# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

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
# HEIGHT_SHARDED multicore tests
params += [
    pytest.param(
        [[1, 1, 96, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(96, 64),
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 64),
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 64],
            "pad_value": 3.25,
            "use_multicore": True,
        },
        id="height_sharded_multicore_pad_height_4cores",
    )
]
params += [
    pytest.param(
        [[2, 1, 64, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 30),
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 32),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [2, 1, 64, 32],
            "pad_value": -1.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_multi_batch",
    )
]

params += [
    pytest.param(
        [[1, 1, 96, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(96, 30),
                    core_grid=ttnn.CoreGrid(y=3, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.COL_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(96, 32),
                core_grid=ttnn.CoreGrid(y=3, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 96, 32],
            "pad_value": 8.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_col_major_3cores",
    )
]

params += [
    pytest.param(
        [[1, 1, 64, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(64, 64),
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(64, 64),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 0.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_no_padding",
    )
]

params += [
    pytest.param(
        [[1, 1, 128, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 30),
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 32),
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 32],
            "pad_value": 5.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_width_pad_only_4cores",
    )
]

params += [
    pytest.param(
        [[2, 1, 50, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(100, 64),
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 64),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [2, 1, 64, 64],
            "pad_value": 2.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_multi_batch_height_pad_only",
    )
]

params += [
    pytest.param(
        [[2, 1, 50, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(100, 30),
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 32),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [2, 1, 64, 32],
            "pad_value": -7.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_multi_batch_height_and_width_pad",
    )
]

params += [
    pytest.param(
        [[1, 1, 256, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(256, 64),
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(256, 64),
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 256, 64],
            "pad_value": 0.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_multiple_tile_rows_per_core",
    )
]

params += [
    pytest.param(
        [[1, 1, 128, 96]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 96),
                    core_grid=ttnn.CoreGrid(y=4, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 96),
                core_grid=ttnn.CoreGrid(y=4, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 96],
            "pad_value": 0.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_multiple_tiles_per_row",
    )
]

params += [
    pytest.param(
        [[1, 1, 64, 64]],
        {
            "dtype": [ttnn.float32],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(64, 64),
                    core_grid=ttnn.CoreGrid(y=2, x=1),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(64, 64),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 1.5,
            "use_multicore": True,
        },
        id="height_sharded_multicore_fp32",
    )
]

params += [
    pytest.param(
        [[1, 1, 128, 64]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 64),
                    core_grid=ttnn.CoreGrid(y=2, x=2),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 64),
                core_grid=ttnn.CoreGrid(y=2, x=2),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 64],
            "pad_value": 0.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_row_major_2x2_grid_no_pad",
    )
]

params += [
    pytest.param(
        [[1, 1, 128, 30]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [
                ttnn.create_sharded_memory_config(
                    shape=(128, 30),
                    core_grid=ttnn.CoreGrid(y=2, x=2),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
            ],
            "output_mem_config": ttnn.create_sharded_memory_config(
                shape=(128, 32),
                core_grid=ttnn.CoreGrid(y=2, x=2),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
            "output_tensor_shape": [1, 1, 128, 32],
            "pad_value": 6.0,
            "use_multicore": True,
        },
        id="height_sharded_multicore_row_major_2x2_grid_width_pad",
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
