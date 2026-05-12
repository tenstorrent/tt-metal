# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
import math

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

pytestmark = pytest.mark.use_module_device


# Test for int types
@pytest.mark.parametrize("shape", [[1, 1, 32, 256], [64, 64], [9, 32, 768], [128]])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_to_memory_config(shape, layout, dtype, device):
    torch.manual_seed(2005)
    torch_dtype = torch.int32

    input = torch.randint(1, 100, shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, dtype, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, dtype, layout=layout, device=device)

    ttnn.to_memory_config(input, input_b.memory_config(), output_tensor=input_b)
    assert input_b.shape == input.shape
    assert_equal(ttnn.to_torch(input), ttnn.to_torch(input_b))


# Test for block sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[128, 64]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_to_memory_config_block_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype, layout=layout)

    num_cores_x = 2
    num_cores_y = 2
    num_cores = num_cores_x * num_cores_y
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]
    tile_widths_per_core = math.ceil(dims_b4_last_dim / num_cores_x)
    shard_shape = (tile_widths_per_core, 32 * math.ceil((shape[-1] / 32 / num_cores_y)))
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.to_memory_config(input_tensor, output_mem_config, output_tensor=output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_equal(input_tensor, outout_tensor)


# Test for width sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[1, 2, 96, 128]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
def test_to_memory_config_width_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype, layout=layout)

    num_cores = 2
    shard_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 4)),
        ]
    )

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]

    shard_shape = (dims_b4_last_dim, 32 * math.ceil((math.ceil(shape[-1] / 32) / num_cores)))
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.to_memory_config(input_tensor, output_mem_config, output_tensor=output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_equal(input_tensor, outout_tensor)


# Test for height sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[512, 64]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
def test_to_memory_config_height_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype=dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype=dtype, layout=layout)

    num_cores = 8
    shard_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 5)),
        ]
    )

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]
    tile_widths_per_core = math.ceil(dims_b4_last_dim / num_cores)
    shard_shape = (tile_widths_per_core, shape[-1])

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.to_memory_config(input_tensor, output_mem_config, output_tensor=output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_equal(input_tensor, outout_tensor)


def test_to_memory_config_width_sharded_unaligned_shard_width(device):
    torch.manual_seed(1234)
    shape = [1, 1, 64, 100]
    input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR)
    ttnn_output = ttnn.from_torch(output_torch, ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR)

    num_cores = 4
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    shard_shape = (64, 25)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_tensor = ttnn.to_device(ttnn_input, device, memory_config=mem_config)
    output_tensor = ttnn.to_device(ttnn_output, device, memory_config=mem_config)

    ttnn.to_memory_config(input_tensor, output_tensor.memory_config(), output_tensor=output_tensor)
    input_result = ttnn.to_torch(input_tensor)
    output_result = ttnn.to_torch(output_tensor)
    assert_equal(input_result, output_result)


def check_mem_config(tensor, expected_memory_config, is_nd_sharded):
    out_mc = tensor.memory_config()
    assert out_mc.is_sharded(), "Output tensor is not sharded"
    if is_nd_sharded:
        expected_shard_spec = expected_memory_config.nd_shard_spec
        actual_shard_spec = out_mc.nd_shard_spec
        assert (
            actual_shard_spec.shard_shape == expected_shard_spec.shard_shape
        ), f"ND Shard shape mismatch: {actual_shard_spec.shard_shape} != {expected_shard_spec.shard_shape}"
        assert actual_shard_spec.shard_distribution_strategy == expected_shard_spec.shard_distribution_strategy, (
            f"Distribution strategy mismatch: "
            f"{actual_shard_spec.shard_distribution_strategy} != {expected_shard_spec.shard_distribution_strategy}"
        )
    else:
        expected_shard_spec = expected_memory_config.shard_spec
        actual_shard_spec = out_mc.shard_spec
        assert (
            actual_shard_spec.shape == expected_shard_spec.shape
        ), f"Shard shape mismatch: {actual_shard_spec.shape} != {expected_shard_spec.shape}"

    assert (
        actual_shard_spec.grid == expected_shard_spec.grid
    ), f"Shard grid mismatch: {actual_shard_spec.grid} != {expected_shard_spec.grid}"
    assert (
        actual_shard_spec.orientation == expected_shard_spec.orientation
    ), f"Shard orientation mismatch: {actual_shard_spec.orientation} != {expected_shard_spec.orientation}"


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single shard (1 core)
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards, 1×3 grid
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 6 shards, 1x3 grid
        ([6, 8, 64], [2, 4, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards, 4×1 grid
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor sharded across batch+channel → 6 shards, disjoint grid (2 + 4 cores)
        (
            [4, 3, 16, 32],
            [2, 1, 16, 32],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 1)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, grid with more cores than shards (6 cores)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 3-D tensor sharded across all dims → 8 shards, 1x3 grid
        ([6, 8, 64], [3, 4, 32], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across all dims → 24 shards, 1x3 grid, uneven sharded, unaligned shard width
        ([6, 8, 64], [4, 3, 20], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        (
            [1, 1, 1024, 1],
            [1, 1, 256, 1],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_memory_config_rm_interleaved_to_nd_sharded(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single DRAM bank
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards on 3 DRAM banks
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 6 shards on 6 DRAM banks
        ([6, 8, 64], [2, 4, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards on 4 DRAM banks
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor → 6 shards on disjoint DRAM banks (banks 0-1 and 4-7)
        (
            [4, 3, 16, 32],
            [2, 1, 16, 32],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, more DRAM banks than shards (8 banks)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_rm_interleaved_to_nd_sharded_dram(device, tensor_shape, shard_shape, grid, shard_orientation):
    torch.manual_seed(0)

    num_dram_banks = device.dram_grid_size().x
    num_cores_in_grid = grid.num_cores()
    if num_cores_in_grid > num_dram_banks:
        pytest.skip(f"Test requires {num_cores_in_grid} DRAM banks but this device only has {num_dram_banks}")

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid, " "output_nd_shard_shape, output_grid",
    [
        # HEIGHT_SHARDED even → ND sharded (reshard height into 3-D ND shard)
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT_SHARDED uneven input (100 rows / 32 → 4 shards, last has 4 rows) → ND sharded
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 50, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED even → ND sharded
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED uneven input (100 cols / 32 → 4 shards, last has 4 cols) → ND sharded
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED uneven input with unaligned shard width (100 cols / 21 → 5 shards, last has 5 cols) → ND sharded
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 21),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
            [1, 1, 64, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED even (2×2 grid) → ND sharded
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED uneven input (100×100 on 2×2, last row/col shards smaller) → ND sharded
        (
            [1, 1, 100, 100],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 50, 100],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_legacy_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_sharded_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_sharded_mem_config,
    )

    nd_shard_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_sharded_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_sharded_mem_config)

    check_mem_config(output_tensor, output_sharded_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)

    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard_shape, input_grid, output_nd_shard_shape, output_grid",
    [
        # ---- Even input & output, sharded across dim 0 (batch) ----
        # 3-D: [6,32,64] sharded [2,32,64]→3 shards, reshard to [3,32,64]→2 shards
        (
            [6, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Even input & output, sharded across dims 0 and 1 ----
        # 3-D: [6,8,64] sharded [2,4,64]→6 shards → reshard to [3,8,64]→2 shards
        (
            [6, 8, 64],
            [2, 4, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 8, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- 4-D sharded across batch+channel (dims 0,1), even ----
        # [4,6,16,32] sharded [1,2,16,32]→12 shards → reshard to [2,3,16,32]→4 shards
        (
            [4, 6, 16, 32],
            [1, 2, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            [2, 3, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # ---- Uneven input shards (dim 0 doesn't divide evenly) ----
        # [5,32,64] sharded [2,32,64]→3 shards (last shard has 1 along dim0), output even [5,32,64]→1 shard
        (
            [5, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [5, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # ---- Uneven output shards (dim 0 doesn't divide evenly) ----
        # [6,32,64] sharded [6,32,64]→1 shard, reshard to [4,32,64]→2 shards (last shard has 2 along dim0)
        (
            [6, 32, 64],
            [6, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Both input and output uneven ----
        # [7,32,64] sharded [3,32,64]→3 shards (last has 1), reshard to [4,32,64]→2 shards (last has 3)
        (
            [7, 32, 64],
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Unaligned shard width on input ----
        # [6,8,50] sharded [2,4,17]→6 shards (shard width 17, not 16/32 aligned), output aligned
        (
            [6, 8, 50],
            [2, 4, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 8, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Unaligned shard width on output ----
        # [6,8,64] sharded [2,4,64]→6 shards (aligned), output [3,4,21]→unaligned width 21
        (
            [6, 8, 64],
            [2, 4, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 4, 21],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # ---- Both input and output unaligned shard widths ----
        # [6,8,50] sharded [2,4,17]→18 shards, reshard to [3,4,13]→16 shards
        (
            [6, 8, 50],
            [2, 4, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 4, 13],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
        ),
        # ---- Uneven + unaligned combined ----
        # [5,7,50] sharded [2,3,17]→12 shards (uneven dim0, dim1; unaligned width), reshard to [3,4,25]→4 shards (uneven; width 25)
        (
            [5, 7, 50],
            [2, 3, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            [3, 4, 25],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_nd_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_nd_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(input_nd_shard_shape, input_grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_nd_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, output_nd_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Interleaved → legacy 2D sharded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # HEIGHT_SHARDED uneven (100 rows / 32 → last shard 4 rows)
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED even
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED uneven (100 cols / 32 → last shard 4 cols)
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # BLOCK_SHARDED even 2×2
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # BLOCK_SHARDED uneven (100×100 on 2×2)
        (
            [1, 1, 100, 100],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_rm_interleaved_to_legacy_2d_sharded(
    device, tensor_shape, output_shard_layout, output_shard_shape_2d, output_grid, shard_orientation
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Legacy 2D sharded → legacy 2D sharded (reshard)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, "
    "input_shard_layout, input_shard_shape_2d, input_grid, "
    "output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # HEIGHT → HEIGHT, different shard sizes
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT → HEIGHT, uneven input (100 rows / 32)
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (50, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH → WIDTH, different shard sizes
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH → WIDTH, uneven input (100 cols / 32)
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 50),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK → BLOCK, different grid
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # BLOCK → BLOCK, uneven (100×100 on 2×2 → 1×2)
        (
            [1, 1, 100, 100],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (100, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # HEIGHT → WIDTH cross-type reshard
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (128, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_legacy_2d_sharded_to_legacy_2d_sharded(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape_2d,
    input_grid,
    output_shard_layout,
    output_shard_shape_2d,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape_2d, shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# ND sharded → legacy 2D sharded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard_shape, input_grid, " "output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # ND sharded → HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded uneven → HEIGHT_SHARDED
        (
            [1, 1, 100, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (50, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded → WIDTH_SHARDED even
        (
            [1, 1, 64, 128],
            [1, 1, 64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded uneven → WIDTH_SHARDED
        (
            [1, 1, 64, 100],
            [1, 1, 64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 50),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded → BLOCK_SHARDED even
        (
            [1, 1, 128, 128],
            [1, 1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # ND sharded uneven → BLOCK_SHARDED
        (
            [1, 1, 100, 100],
            [1, 1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # 3-D ND sharded across batch → HEIGHT_SHARDED
        (
            [4, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 3-D ND sharded with unaligned width → WIDTH_SHARDED
        (
            [6, 8, 50],
            [2, 4, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (48, 25),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_nd_sharded_to_legacy_2d_sharded(
    device,
    tensor_shape,
    input_nd_shard_shape,
    input_grid,
    output_shard_layout,
    output_shard_shape_2d,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(input_nd_shard_shape, input_grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized interleaved → ND sharded (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single shard (1 core)
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards, 1×3 grid
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 4 shards, 1×4 grid
        ([4, 64, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards, 4×1 grid
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor sharded across batch+channel → 6 shards, disjoint grid (2 + 4 cores)
        (
            [4, 3, 32, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 1)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, grid with more cores than shards (6 cores)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 4-D tensor, larger tiles, sharded across batch → 2 shards
        (
            [2, 1, 64, 128],
            [1, 1, 64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # 3-D tensor sharded across all dims → 8 shards, 1×4 grid with uneven sharding on last 2 dims
        ([4, 96, 96], [2, 64, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_memory_config_tilized_interleaved_to_nd_sharded(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized interleaved → ND sharded with dtype conversion (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single shard (1 core)
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards, 1×3 grid
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards, 4×1 grid
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 3-D tensor with uneven shards → 3 shards, grid with more cores than shards (6 cores)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 3-D tensor sharded across all dims → 8 shards, 1×4 grid with uneven sharding on last 2 dims
        ([4, 96, 96], [3, 64, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})),
        # 3-D tensor sharded across all dims → 18 shards, 1×4 grid with uneven sharding on all dims disjoint grid
        (
            [3, 160, 160],
            [2, 64, 64],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 2)),
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype, pcc",
    [
        (ttnn.bfloat16, ttnn.float32, 0.9999),
        (ttnn.float32, ttnn.bfloat16, 0.9999),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.9999),
        (ttnn.bfloat8_b, ttnn.bfloat16, 0.9999),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_interleaved_to_nd_sharded_dtype_conversion(
    device, tensor_shape, shard_shape, grid, input_dtype, output_dtype, pcc, shard_orientation
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config, dtype=output_dtype)

    assert output_tensor.dtype == output_dtype, f"Expected dtype {output_dtype}, got {output_tensor.dtype}"
    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input, output_torch, pcc=pcc)


# ---------------------------------------------------------------------------
# Tilized interleaved → ND sharded DRAM (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single DRAM bank
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards on 3 DRAM banks
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 4 shards on 4 DRAM banks
        ([4, 64, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards on 4 DRAM banks
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor → 6 shards on disjoint DRAM banks (banks 0-1 and 4-7)
        (
            [4, 3, 32, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, more DRAM banks than shards (8 banks)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})),
        # 3-D tensor sharded across all dims → 8 shards on 8 DRAM banks
        ([4, 64, 64], [2, 32, 32], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_interleaved_to_nd_sharded_dram(
    device, tensor_shape, shard_shape, grid, shard_orientation
):
    torch.manual_seed(0)

    num_dram_banks = device.dram_grid_size().x
    num_cores_in_grid = grid.num_cores()
    if num_cores_in_grid > num_dram_banks:
        pytest.skip(f"Test requires {num_cores_in_grid} DRAM banks but this device only has {num_dram_banks}")

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized interleaved → ND sharded DRAM with dtype conversion (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single DRAM bank
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards on 3 DRAM banks
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards on 4 DRAM banks
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 3-D tensor with uneven shards → 3 shards, more DRAM banks than shards (8 banks)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})),
        # 3-D tensor sharded across all dims → disjoint DRAM banks
        (
            [3, 160, 160],
            [2, 64, 64],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0)),
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype, pcc",
    [
        (ttnn.bfloat16, ttnn.float32, 0.9999),
        (ttnn.float32, ttnn.bfloat16, 0.9999),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.9999),
        (ttnn.bfloat8_b, ttnn.bfloat16, 0.9999),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_interleaved_to_nd_sharded_dram_dtype_conversion(
    device, tensor_shape, shard_shape, grid, input_dtype, output_dtype, pcc, shard_orientation
):
    torch.manual_seed(0)

    num_dram_banks = device.dram_grid_size().x
    num_cores_in_grid = grid.num_cores()
    if num_cores_in_grid > num_dram_banks:
        pytest.skip(f"Test requires {num_cores_in_grid} DRAM banks but this device only has {num_dram_banks}")

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config, dtype=output_dtype)

    assert output_tensor.dtype == output_dtype, f"Expected dtype {output_dtype}, got {output_tensor.dtype}"
    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input, output_torch, pcc=pcc)


# ---------------------------------------------------------------------------
# Tilized legacy 2D sharded → ND sharded (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid, " "output_nd_shard_shape, output_grid",
    [
        # HEIGHT_SHARDED even → ND sharded
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT_SHARDED larger → ND sharded
        (
            [1, 1, 256, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 128, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED even → ND sharded
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED larger → ND sharded
        (
            [1, 1, 64, 256],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED even (2×2 grid) → ND sharded
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED larger (2×2 grid) → ND sharded
        (
            [1, 1, 256, 256],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 128, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT_SHARDED → ND sharded with uneven height in output (160/96 → 2 shards, last has 64)
        (
            [1, 1, 160, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [1, 1, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED (3×3 grid) → ND sharded with uneven last 2 dims (192/128, 192/128)
        (
            [1, 1, 192, 192],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
            [1, 1, 128, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # HEIGHT_SHARDED 4-D → ND sharded uneven in all dims (3/2, 5/3, 160/96, 96/64)
        (
            [3, 5, 160, 96],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            [2, 3, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_legacy_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_sharded_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_sharded_mem_config,
    )

    nd_shard_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_sharded_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_sharded_mem_config)

    check_mem_config(output_tensor, output_sharded_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized ND sharded → ND sharded (reshard, tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard_shape, input_grid, output_nd_shard_shape, output_grid",
    [
        # Even input & output, sharded across dim 0 (batch)
        # 3-D: [6,32,64] sharded [2,32,64]→3 shards, reshard to [3,32,64]→2 shards
        (
            [6, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # 4-D sharded across batch+channel (dims 0,1), even
        # [4,6,32,64] sharded [1,2,32,64]→12 shards → reshard to [2,3,32,64]→4 shards
        (
            [4, 6, 32, 64],
            [1, 2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            [2, 3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Uneven input shards (dim 0 doesn't divide evenly)
        # [5,32,64] sharded [2,32,64]→3 shards (last has 1), output even [5,32,64]→1 shard
        (
            [5, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [5, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # Uneven output shards (dim 0 doesn't divide evenly)
        # [6,32,64] sharded [6,32,64]→1 shard, reshard to [4,32,64]→2 shards (last has 2)
        (
            [6, 32, 64],
            [6, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Both input and output uneven
        # [7,32,64] sharded [3,32,64]→3 shards (last has 1), reshard to [4,32,64]→2 shards (last has 3)
        (
            [7, 32, 64],
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Sharded across last 2 dims
        # [4,64,128] sharded [2,32,64]→8 shards → reshard to [2,64,64]→4 shards
        (
            [4, 64, 128],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D sharded across all dims, even
        # [4,2,64,128] sharded [2,1,32,64]→16 shards → reshard to [4,2,32,64]→4 shards
        (
            [4, 2, 64, 128],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [4, 2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Uneven last 2 dims on both input and output
        # [4,96,96] sharded [2,64,64]→8 shards (96/64 uneven) → reshard to [3,64,64]→8 shards
        (
            [4, 96, 96],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [3, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        # Uneven last dim on input, even on output
        # [2,64,160] sharded [1,64,96]→4 shards (160/96 uneven) → reshard to [2,64,64]→5 shards
        (
            [2, 64, 160],
            [1, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_nd_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_nd_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(input_nd_shard_shape, input_grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_nd_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, output_nd_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized interleaved → legacy 2D sharded (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # HEIGHT_SHARDED larger
        (
            [1, 1, 256, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED even
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED larger
        (
            [1, 1, 64, 256],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # BLOCK_SHARDED even 2×2
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # BLOCK_SHARDED larger 2×2
        (
            [1, 1, 256, 256],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # HEIGHT_SHARDED uneven (160 rows / 64 → 3 shards, last has 32 rows)
        (
            [1, 1, 160, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # WIDTH_SHARDED uneven (160 cols / 64 → 3 shards, last has 32 cols)
        (
            [1, 1, 64, 160],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # BLOCK_SHARDED uneven (160×160 on 2×2, last row/col shards have 32)
        (
            [1, 1, 160, 160],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_interleaved_to_legacy_2d_sharded(
    device, tensor_shape, output_shard_layout, output_shard_shape_2d, output_grid, shard_orientation
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized legacy 2D sharded → legacy 2D sharded (reshard, tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, "
    "input_shard_layout, input_shard_shape_2d, input_grid, "
    "output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # HEIGHT → HEIGHT, different shard sizes
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT → HEIGHT, larger tensor
        (
            [1, 1, 256, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH → WIDTH, different shard sizes
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH → WIDTH, larger tensor
        (
            [1, 1, 64, 256],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK → BLOCK, different grid
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # HEIGHT → WIDTH cross-type reshard
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (128, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT → HEIGHT, uneven output (160 rows / 96 → 2 shards, last has 64 rows)
        (
            [1, 1, 160, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH → WIDTH, uneven output (160 cols / 96 → 2 shards, last has 64 cols)
        (
            [1, 1, 64, 160],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK → BLOCK, uneven on both dims (160×160 with shard 96×96 → last row/col has 64)
        (
            [1, 1, 160, 160],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_legacy_2d_sharded_to_legacy_2d_sharded(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape_2d,
    input_grid,
    output_shard_layout,
    output_shard_shape_2d,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape_2d, shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized ND sharded → legacy 2D sharded (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard_shape, input_grid, " "output_shard_layout, output_shard_shape_2d, output_grid",
    [
        # ND sharded → HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded → WIDTH_SHARDED even
        (
            [1, 1, 64, 128],
            [1, 1, 64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded → BLOCK_SHARDED even
        (
            [1, 1, 128, 128],
            [1, 1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # 3-D ND sharded across batch → HEIGHT_SHARDED
        (
            [4, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # ND sharded larger → HEIGHT_SHARDED
        (
            [1, 1, 256, 64],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded larger → WIDTH_SHARDED
        (
            [1, 1, 64, 256],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded with uneven last 2 dims → HEIGHT_SHARDED uneven (160/96 → last shard 64)
        (
            [1, 1, 160, 96],
            [1, 1, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ND sharded with uneven both dims → BLOCK_SHARDED uneven (160×160 / 96×96)
        (
            [1, 1, 160, 160],
            [1, 1, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # 4-D ND sharded uneven all dims (3/2, 5/3, 160/96, 96/64) → HEIGHT_SHARDED
        (
            [3, 5, 160, 96],
            [2, 3, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_nd_sharded_to_legacy_2d_sharded(
    device,
    tensor_shape,
    input_nd_shard_shape,
    input_grid,
    output_shard_layout,
    output_shard_shape_2d,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(input_nd_shard_shape, input_grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_shard_spec = ttnn.ShardSpec(output_grid, output_shard_shape_2d, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_shard_layout, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# ND sharded → interleaved (row-major layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D even sharding, single core
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D even sharding across first dim
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D uneven first dim (5 / 2 → last shard has 1)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D uneven last two dims (100 / 32 rows, 100 / 32 cols)
        (
            [2, 100, 100],
            [1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
        ),
        # 4-D uneven in all dims (3/2 batches, 5/3 channels, 100/32 rows, 100/32 cols)
        (
            [3, 5, 100, 100],
            [2, 3, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
        ),
        # 3-D shard width not aligned to 16 bytes (width=10, 10*2=20 bytes, not multiple of 16)
        ([4, 32, 10], [2, 32, 10], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})),
        # 3-D shard width=3 (3*2=6 bytes, not aligned to 16)
        ([2, 16, 3], [1, 16, 3], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})),
        # 3-D shard width=5 (5*2=10 bytes, not aligned to 16)
        ([6, 8, 5], [2, 8, 5], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D shard width=7 (7*2=14 bytes, not aligned to 16), uneven first dim
        ([5, 16, 7], [2, 16, 7], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 4-D uneven all dims + shard width not aligned (width=10)
        (
            [3, 5, 50, 55],
            [2, 3, 16, 10],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 1)),
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_to_memory_config_rm_nd_sharded_to_interleaved(
    device, tensor_shape, shard_shape, grid, shard_orientation, output_buffer_type
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Legacy 2D sharded → interleaved (row-major layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid",
    [
        # HEIGHT_SHARDED even (128 rows / 32 = 4 shards)
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # HEIGHT_SHARDED uneven (100 rows / 32 → last shard has 4 rows)
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED even (128 cols / 32 = 4 shards)
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED uneven (100 cols / 32 → last shard has 4 cols)
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED shard width not aligned to 16 bytes (21*2=42 bytes)
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 21),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        # BLOCK_SHARDED even (2×2 grid)
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # BLOCK_SHARDED uneven (100×100 on 2×2)
        (
            [1, 1, 100, 100],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_to_memory_config_rm_legacy_2d_sharded_to_interleaved(
    device, tensor_shape, input_shard_layout, input_shard_shape, input_grid, shard_orientation, output_buffer_type
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# Test for large row major input
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        ([160, 131072], [32, 131072], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        ([160, 131072], [32, 65536], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        ([160, 65536], [32, 131072], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        (
            [160, 5210112],
            [32, 131072],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        ),
        # Test for uneven sharding and unaligned shard width
        (
            [160, 5210112],
            [96, 1302529],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM])
def test_to_memory_config_rm_interleaved_to_nd_sharded_large_row(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    if os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("Skipping large row test on ttsim to avoid timeout")
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# Test for large row major input legacy 2D sharding
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        (
            [160, 5210112],
            [160, 434176],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM])
def test_to_memory_config_rm_interleaved_to_legacy_2D_sharded_large_row(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    if os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("Skipping large row test on ttsim to avoid timeout")
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")
    torch.manual_seed(0)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type, shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=False)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# Test for large row major ND sharded to interleaved
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        ([160, 131072], [32, 131072], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        ([160, 131072], [32, 65536], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        ([160, 65536], [32, 131072], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})),
        (
            [160, 5210112],
            [32, 131072],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        ),
        # Test for uneven sharding and unaligned shard width
        (
            [160, 5210112],
            [96, 1302529],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM])
def test_to_memory_config_rm_nd_sharded_to_interleaved_large_row(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    if os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("Skipping large row test on ttsim to avoid timeout")
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sharded_memory_config,
        device=device,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized ND sharded → interleaved (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D even sharding, single core
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D even sharding across first dim
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D uneven first dim (5 / 2 → last shard has 1)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D uneven last two dims (96 / 64 tiles height, 96 / 64 tiles width)
        (
            [4, 96, 96],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D uneven in all dims (3/2 batches, 5/3 channels, 160/96 rows, 96/64 cols)
        (
            [3, 5, 160, 96],
            [2, 3, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
        ),
        # 4-D even sharding across batch dim
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 3-D uneven all dims, disjoint grid
        (
            [3, 160, 160],
            [2, 64, 64],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 2)),
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_to_memory_config_tilized_nd_sharded_to_interleaved(
    device, tensor_shape, shard_shape, grid, shard_orientation, output_buffer_type
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized ND sharded → interleaved with dtype conversion (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D even sharding
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D uneven last two dims
        (
            [4, 96, 96],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D uneven in all dims
        (
            [3, 5, 160, 96],
            [2, 3, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype, pcc",
    [
        (ttnn.bfloat16, ttnn.float32, 0.9999),
        (ttnn.float32, ttnn.bfloat16, 0.9999),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.9999),
        (ttnn.bfloat8_b, ttnn.bfloat16, 0.9999),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_nd_sharded_to_interleaved_dtype_conversion(
    device, tensor_shape, shard_shape, grid, input_dtype, output_dtype, pcc, shard_orientation
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config, dtype=output_dtype)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    assert output_tensor.dtype == output_dtype, f"Expected dtype {output_dtype}, got {output_tensor.dtype}"
    output_torch = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input, output_torch, pcc=pcc)


# ---------------------------------------------------------------------------
# Tilized legacy 2D sharded → interleaved (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid",
    [
        # HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # HEIGHT_SHARDED uneven (160 rows / 64 → 3 shards, last has 32)
        (
            [1, 1, 160, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # WIDTH_SHARDED even
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # WIDTH_SHARDED uneven (160 cols / 64 → 3 shards, last has 32)
        (
            [1, 1, 64, 160],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # BLOCK_SHARDED even (2×2 grid)
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # BLOCK_SHARDED uneven (192×192 on 3×3, each shard 64×64)
        (
            [1, 1, 192, 192],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
        # HEIGHT_SHARDED 4-D (3×5×160×96)
        (
            [3, 5, 160, 96],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_to_memory_config_tilized_legacy_2d_sharded_to_interleaved(
    device, tensor_shape, input_shard_layout, input_shard_shape, input_grid, shard_orientation, output_buffer_type
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Tilized legacy 2D sharded → interleaved with dtype conversion (tile layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid",
    [
        # HEIGHT_SHARDED even
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # BLOCK_SHARDED uneven (192×192 on 3×3)
        (
            [1, 1, 192, 192],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
        # HEIGHT_SHARDED 4-D
        (
            [3, 5, 160, 96],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype, pcc",
    [
        (ttnn.bfloat16, ttnn.float32, 0.9999),
        (ttnn.float32, ttnn.bfloat16, 0.9999),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.9999),
        (ttnn.bfloat8_b, ttnn.bfloat16, 0.9999),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_memory_config_tilized_legacy_2d_sharded_to_interleaved_dtype_conversion(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape,
    input_grid,
    input_dtype,
    output_dtype,
    pcc,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config, dtype=output_dtype)

    assert not output_tensor.is_sharded(), "Output tensor should be interleaved, not sharded"
    assert output_tensor.dtype == output_dtype, f"Expected dtype {output_dtype}, got {output_tensor.dtype}"
    output_torch = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input, output_torch, pcc=pcc)


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # Uneven first dim: 5 / 2 → last shard width 1 along dim 0; grid has spare cores
        (
            [5, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
        ),
        # Uneven + unaligned shard width on last dim
        (
            [6, 8, 64],
            [4, 3, 20],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_memory_config_rm_preallocated_output(device, tensor_shape, shard_shape, grid, buffer_type):
    """Interleaved row-major → ND sharded (row-major orientation) with uneven shards; write into preallocated output."""
    torch.manual_seed(0)
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    # NOTE: This next step may result in an implicit conversion from an nd_shard_spec to an equivalent legacy 2D shard_spec, causing the actual memory_config of the preallocated tensor to be different from the one passed in to the copy call.
    preallocated = ttnn.allocate_tensor_on_device(
        ttnn.Shape(tensor_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        sharded_memory_config,
    )

    output_tensor = ttnn.to_memory_config(
        input_tensor,
        memory_config=preallocated.memory_config(),
        output_tensor=preallocated,
    )

    check_mem_config(output_tensor, preallocated.memory_config(), is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


# ---------------------------------------------------------------------------
# Program cache callback (override_runtime_arguments) tests
# ---------------------------------------------------------------------------


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        ([6, 8, 64], [4, 3, 20], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
    ],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_memory_config_rm_override_runtime_arguments(
    device, isolate_program_cache, tensor_shape, shard_shape, grid, buffer_type
):
    """Call the row-major program twice with different data to exercise override_runtime_arguments on the second call."""
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)

    outputs = []
    assert device.num_program_cache_entries() == 0, "Program cache should be empty before the test"
    for seed in (0, 42):
        torch.manual_seed(seed)
        torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_device(input_tensor, device)

        output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

        check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)
        output_torch = ttnn.to_torch(output_tensor)
        assert_equal(torch_input, output_torch)
        outputs.append(output_torch)

    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 program cache entry (cache hit on second call), got {device.num_program_cache_entries()}"
    assert not torch.equal(outputs[0], outputs[1]), "Outputs should differ to confirm different data was processed"


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        (
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        (
            [3, 5, 160, 96],
            [2, 3, 96, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_memory_config_tilized_override_runtime_arguments(
    device, isolate_program_cache, tensor_shape, shard_shape, grid, buffer_type
):
    """Call the tilized program twice with different data to exercise override_runtime_arguments on the second call."""
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)

    outputs = []
    assert device.num_program_cache_entries() == 0, "Program cache should be empty before the test"
    for seed in (0, 42):
        torch.manual_seed(seed)
        torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        output_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_memory_config)

        check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)
        output_torch = ttnn.to_torch(output_tensor)
        assert_equal(torch_input, output_torch)
        outputs.append(output_torch)

    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 program cache entry (cache hit on second call), got {device.num_program_cache_entries()}"
    assert not torch.equal(outputs[0], outputs[1]), "Outputs should differ to confirm different data was processed"


def test_to_memory_config_tile_interleaved_to_width_sharded_bf8(device):
    torch.manual_seed(0)
    shape = [1, 1, 8192, 2048]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    shard_shape = (8192, 256)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=output_mem_config)

    output_torch = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input, output_torch, 0.9999)


@pytest.mark.parametrize(
    "src_buffer, dst_buffer",
    [
        (ttnn.BufferType.L1, ttnn.BufferType.DRAM),
        (ttnn.BufferType.DRAM, ttnn.BufferType.L1),
    ],
)
def test_to_memory_config_rm_interleaved_l1_dram(device, src_buffer, dst_buffer):
    torch.manual_seed(0)
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    src_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, src_buffer)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=src_mem_config
    )

    dst_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, dst_buffer)
    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=dst_mem_config)

    assert output_tensor.memory_config().buffer_type == dst_buffer
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "src_buffer, dst_buffer",
    [
        (ttnn.BufferType.L1, ttnn.BufferType.DRAM),
        (ttnn.BufferType.DRAM, ttnn.BufferType.L1),
    ],
)
def test_to_memory_config_tile_interleaved_l1_dram(device, src_buffer, dst_buffer):
    torch.manual_seed(0)
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    src_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, src_buffer)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=src_mem_config
    )

    dst_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, dst_buffer)
    output_tensor = ttnn.to_memory_config(input_tensor, memory_config=dst_mem_config)

    assert output_tensor.memory_config().buffer_type == dst_buffer
    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)
