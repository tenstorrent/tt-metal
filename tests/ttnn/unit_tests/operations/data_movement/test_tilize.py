# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

shapes = [[[1, 1, 32, 32]], [[3, 1, 320, 384]], [[1, 1, 128, 7328]]]


@pytest.mark.parametrize(
    "input_shapes",
    shapes,
)
@pytest.mark.parametrize(
    "tilize_args",
    (
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "use_multicore": False,
        },
    ),
)
def test_tilize_test(input_shapes, tilize_args, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test("tilize", input_shapes, datagen_func, comparison_func, device, tilize_args)


@pytest.mark.parametrize("shape", [(64, 128), (512, 512)])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_fp32_truncation(device, shape, use_multicore):
    torch.manual_seed(2005)
    input_a = torch.full(shape, 1.9908e-05, dtype=torch.float32)
    # Use the fixture-provided device directly
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.tilize(input_tensor, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(input_tensor)
    assert torch.allclose(input_a, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[32, 256 * 64]])
@pytest.mark.parametrize("shard_shape", [[32, 256]])
@pytest.mark.parametrize(
    "shard_core_grid", [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})]
)  # 64 cores in an 8x8 grid
def test_tilize_row_major_to_width_sharded(device, dtype, tensor_shape, shard_shape, shard_core_grid):
    torch.manual_seed(42)
    torch.set_printoptions(sci_mode=False)

    shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create test data with sequential values from 1 to n for debugging
    for _ in range(30):
        input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)

        # Convert to ttnn tensor with row major layout and width sharding
        input_ttnn_tensor = ttnn.from_torch(
            input_torch_tensor,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_memory_config,
        )
        ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config)
        output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

        assert torch.equal(input_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize("input_shape", [(32, 15936), (160, 5210112)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)
