# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

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
def test_tilize_row_major_to_width_sharded(device, dtype):
    """
    Test tilize operation for row major to width sharded tensors.

    Shape: [32, 256*64] = [32, 16384]
    Width sharded with shard shape: [32, 256]
    Distributed on 64 cores (8x8 grid), with 8 tiles per core.
    """
    # Define tensor shape: [32, 256*64]
    tensor_shape = [32, 256 * 64]  # [32, 16384]

    # Define width sharding configuration
    # Shard shape: [32, 256] (full height, width divided across 64 cores)
    shard_shape = [32, 256]

    # 64 cores in an 8x8 grid
    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    # Create ShardSpec for width sharding
    shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    # Input memory config: row major with width sharding
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Output memory config: tile layout with width sharding
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create test data
    torch.manual_seed(42)
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

    # Convert to ttnn tensor with row major layout and width sharding
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory_config
    )

    # Apply tilize operation
    ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config)

    # Convert back to torch and verify
    output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Verify the output matches the input
    assert torch.allclose(input_torch_tensor, output_torch_tensor, rtol=1e-2, atol=1e-2)
