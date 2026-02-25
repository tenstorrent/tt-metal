# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
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


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([4, 128, 128], [2, 64, 64]),
        ([3, 160, 160], [2, 64, 64]),
        ([5, 4, 160, 160], [2, 3, 64, 96]),
        ([23, 96, 160], [4, 64, 96]),  # Test with uneven sharding and cliff core
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("input_is_nd_sharded", [True, False])
@pytest.mark.parametrize("output_is_nd_sharded", [True, False])
@pytest.mark.parametrize("use_multicore", [True, False])
def test_tilize_nd_sharded(
    device, dtype, tensor_shape, shard_shape, shard_core_grid, input_is_nd_sharded, output_is_nd_sharded, use_multicore
):
    torch.manual_seed(42)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    if input_is_nd_sharded:
        input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)
        if not use_multicore:
            pytest.skip("Singlecore is not supported for sharded input")
    else:
        input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if output_is_nd_sharded:
        output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=use_multicore)
    output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert_equal(input_torch_tensor, output_torch_tensor)


# Legacy-sharded TILE output requires shard height and width to be multiples of 32 (tile size).
# Only (tensor_shape, grid) where height_shard, width_shard, and block_shard dims are all % 32 == 0.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        (
            [4, 128, 128],
            [2, 64, 64],
        ),  # 2D (512,128): height_shard (128,128), width_shard (512,32), block (256,64) all tile-aligned
        ([8, 64, 128], [2, 32, 64]),  # 2D (512,128): same as above
        (
            [7, 128, 128],
            [2, 64, 96],
        ),  # Uneven ND sharding: dim 0 has 7 % 2 != 0 -> cliff core; 2D (896,128) still tile-aligned
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_tilize_nd_sharded_to_legacy_sharded(
    device, dtype, tensor_shape, shard_shape, shard_core_grid, output_memory_layout
):
    """tilize: ND-sharded row-major input -> legacy-sharded (HEIGHT/WIDTH/BLOCK) tile output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    height_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )
    shard_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": height_shard_shape},
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": width_shard_shape},
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": block_shard_shape},
    }
    layout_info = shard_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        layout_info["shard_grid"], layout_info["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    #    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)
    out_shard_spec = ttnn_output_tensor.memory_config().shard_spec
    print("output shard shape:", out_shard_spec.shape)
    print("output shard grid:", out_shard_spec.grid)
    output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert_equal(input_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize("input_shape", [(32, 15936), (160, 5210112)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)
