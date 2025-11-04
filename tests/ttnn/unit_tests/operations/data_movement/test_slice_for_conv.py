# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole
from ..test_utils import round_up


def num_to_core_range_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


@pytest.mark.parametrize(
    "dims, slice_size, cores",
    [
        [[2, 256, 300, 64], 128, 22],
        [[2, 256, 128, 32], 64, 8],
        [[2, 256, 256, 128], 64, 64],
        [[2, 256, 256, 9], 64, 64],
        [[2, 256, 256, 17], 64, 64],
        [[2, 1024, 1024, 3], 64, 64],
        [[2, 313, 71, 32], 32, 7],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_slice_write_height_sharded(device, dims, slice_dim, slice_size, cores, layout, orientation, dtype):
    core_grid = device.compute_with_storage_grid_size()

    if core_grid.x * core_grid.y < cores:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)

    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=dtype)
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)

    core_range = ttnn.num_cores_to_corerangeset(cores, core_grid, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range, shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, shard_orientation=orientation
    )
    num_slices = round_up(dims[slice_dim], slice_size) // slice_size
    padded_channels = round_up(dims[-1], 32)

    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))

    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], padded_channels]
        begins[slice_dim] = i * slice_size
        if i == num_slices - 1:
            ends[slice_dim] = dims[slice_dim]
        else:
            ends[slice_dim] = (i + 1) * slice_size
        this_torch_input = padded_torch_input[
            begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : ends[3]
        ]

        this_ttnn_input = ttnn.from_torch(
            this_torch_input,
            layout=layout,
            dtype=dtype,
        )
        this_ttnn_input = ttnn.to_device(
            this_ttnn_input,
            device=device,
        )
        this_ttnn_input = ttnn.reshape(this_ttnn_input, this_ttnn_input.padded_shape)
        this_ttnn_input = ttnn.reshape(this_ttnn_input, [1, 1, -1, this_ttnn_input.padded_shape[-1]])
        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ends[-1] = ttnn_output.shape[-1]
        ttnn.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


@pytest.mark.parametrize(
    "dims, slice_size, cores",
    [
        [[2, 64, 64, 2048], 32, 64],
        [[2, 48, 48, 2944], 32, 46],
        [[2, 48, 48, 2904], 32, 46],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_slice_write_width_sharded(device, dims, slice_dim, slice_size, cores, layout, orientation, dtype):
    core_grid = device.compute_with_storage_grid_size()

    if core_grid.x * core_grid.y < cores:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)
    torch_input = torch.tensor(range(dims[-1]), dtype=torch.int32).reshape(1, 1, 1, dims[-1]).broadcast_to(dims)

    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=dtype)
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)

    core_range = ttnn.num_cores_to_corerangeset(cores, core_grid, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range, shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED, shard_orientation=orientation
    )
    num_slices = round_up(dims[slice_dim], slice_size) // slice_size
    padded_channels = round_up(dims[-1], 32 * cores)

    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))

    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], padded_channels]
        begins[slice_dim] = i * slice_size
        if i == num_slices - 1:
            ends[slice_dim] = dims[slice_dim]
        else:
            ends[slice_dim] = (i + 1) * slice_size
        this_torch_input = padded_torch_input[
            begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : ends[3]
        ]

        this_ttnn_input = ttnn.from_torch(
            this_torch_input,
            layout=layout,
            dtype=dtype,
        )
        this_ttnn_input = ttnn.to_device(
            this_ttnn_input,
            device=device,
        )
        this_ttnn_input = ttnn.reshape(this_ttnn_input, this_ttnn_input.padded_shape)
        this_ttnn_input = ttnn.reshape(this_ttnn_input, [1, 1, -1, this_ttnn_input.padded_shape[-1]])
        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ends[-1] = ttnn_output.shape[-1]
        ttnn.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


@pytest.mark.parametrize(
    "dims, slice_size, core_x, core_y, layout",
    [
        [[2, 256, 256, 64], 128, 2, 8, ttnn.ROW_MAJOR_LAYOUT],
        [[2, 256, 128, 128], 16, 4, 4, ttnn.ROW_MAJOR_LAYOUT],
        [[2, 32, 32, 128], 32, 2, 2, ttnn.ROW_MAJOR_LAYOUT],
        [[2, 256, 256, 64], 64, 2, 4, ttnn.TILE_LAYOUT],
        [[2, 256, 128, 128], 32, 4, 4, ttnn.TILE_LAYOUT],
        [[2, 64, 64, 128], 32, 2, 2, ttnn.TILE_LAYOUT],
        [[2, 64, 64, 512], 32, 3, 5, ttnn.TILE_LAYOUT],
        [[2, 128, 128, 63], 32, 2, 8, ttnn.TILE_LAYOUT],
        [[2, 128, 128, 63], 32, 2, 8, ttnn.ROW_MAJOR_LAYOUT],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_slice_write_block_sharded(device, dims, slice_dim, slice_size, core_x, core_y, layout, orientation, dtype):
    core_grid = device.core_grid
    if core_grid.x < core_x or core_grid.y < core_y:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)
    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=dtype)
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
    num_slices = dims[slice_dim] // slice_size

    padded_channels = round_up(dims[-1], 32 * core_x)
    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (core_x - 1, core_y - 1))])
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_grid, shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED, shard_orientation=orientation
    )

    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], dims[3]]
        begins[slice_dim] = i * slice_size
        if i == num_slices - 1:
            ends[slice_dim] = dims[slice_dim]
        else:
            ends[slice_dim] = (i + 1) * slice_size
        this_ttnn_input = ttnn.from_torch(
            padded_torch_input[
                begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : padded_channels
            ],
            device=device,
            layout=layout,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        core_grid = ttnn.CoreGrid(x=core_x, y=core_y)

        this_ttnn_input = ttnn.reshape(this_ttnn_input, [1, 1, -1, this_ttnn_input.padded_shape[-1]])
        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ttnn.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


@pytest.mark.parametrize(
    "dims, slice_size, cores",
    [
        [[2, 100, 100, 32], 50, 64],
        [[2, 512, 256, 32], 128, 64],
        [[2, 256, 128, 64], 32, 8],
        [[2, 67, 35, 64], 14, 8],
        [[2, 256, 256, 37], 64, 64],
        [[2, 312, 489, 100], 53, 64],
        [[2, 255, 255, 63], 37, 64],
        [[2, 299, 299, 99], 99, 64],
        [[2, 8, 8, 32], 2, 4],
        [[2, 8, 16, 2], 2, 8],
        [[2, 981, 39, 63], 63, 41],
        [[1, 1024, 1024, 128], 37, 64],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("pad_value", [8, 16, 32])
def test_slice_height_sharded_for_conv2d(device, dims, slice_dim, slice_size, cores, layout, input_dtype, pad_value):
    if input_dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is not supported in row major layout")

    orientation = ttnn.ShardOrientation.ROW_MAJOR
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x * core_grid.y < cores:
        pytest.skip(
            "Skipping test_slice_height_sharded_for_conv2d as device does not have enough Tensix cores. Needs %d, but device has %d"
            % (cores, core_grid.x * core_grid.y)
        )

    strides = [1, 1, 1, 1]
    torch.manual_seed(2001)
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randint(-10, 10, dims).to(dtype=torch_dtype)

    core_range = ttnn.num_cores_to_corerangeset(cores, core_grid, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    num_slices = dims[slice_dim] // slice_size
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, layout=layout, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range, shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, shard_orientation=orientation
    )
    padded_channels = round_up(dims[-1], pad_value)
    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))
    torch.set_printoptions(sci_mode=False, precision=2)
    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], dims[3]]
        begins[slice_dim] = i * slice_size
        ends[slice_dim] = (i + 1) * slice_size
        this_torch_output = padded_torch_input[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2]]
        output_shape = this_torch_output.shape
        output_shape = [1, 1, output_shape[0] * output_shape[1] * output_shape[2], round_up(output_shape[3], pad_value)]

        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            output_shape, parallel_config, 1
        )
        this_ttnn_output = ttnn.padded_slice(ttnn_input, begins, ends, strides, memory_config=memory_config)
        output = ttnn.to_torch(this_ttnn_output)
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "dims, slice_size, core_y, core_x",
    [
        [[2, 64, 64, 256], 32, 4, 4],
        [[2, 64, 64, 512], 16, 4, 4],
        [[2, 16, 16, 1024], 4, 4, 4],
        [[2, 128, 128, 256], 32, 8, 4],
        [[2, 128, 128, 63], 32, 8, 2],
        [[2, 128, 128, 528], 96, 8, 6],
        [[2, 128, 128, 96], 96, 8, 3],
        [[2, 1024, 1024, 256], 33, 10, 11],
        [[1, 64, 128, 256], 65, 4, 5],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("pad_value", [8, 32])
def test_slice_block_sharded_for_conv2d(
    device, dims, slice_dim, slice_size, core_x, core_y, layout, input_dtype, pad_value
):
    if input_dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is not supported in row major layout")
    if round_up(dims[-1], pad_value) / pad_value < core_x:
        pytest.skip("Skipping test with dim %s where all cores %d are not used in block sharding" % (dims, core_x))

    orientation = ttnn.ShardOrientation.ROW_MAJOR
    core_grid = device.core_grid
    if core_grid.x < core_x or core_grid.y < core_y:
        pytest.skip(
            "Skipping test_slice_block_sharded_for_conv2d as device does not have enough Tensix cores. Needs %s, but device has %s"
            % ((core_x, core_y), (core_grid.x, core_grid.y))
        )

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randint(-10, 10, dims).to(dtype=torch_dtype)
    torch_input = torch.tensor(range(dims[-1]), dtype=torch_dtype).reshape(1, 1, 1, dims[-1]).broadcast_to(dims)
    num_slices = dims[slice_dim] // slice_size
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, layout=layout, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    padded_channels = round_up(dims[-1], core_x * pad_value)
    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))
    core_range_start = ttnn.CoreCoord(0, 0)
    core_range_end = ttnn.CoreCoord(core_x - 1, core_y - 1)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core_range_start, core_range_end)])
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range, shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED, shard_orientation=orientation
    )
    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], dims[3]]
        begins[slice_dim] = i * slice_size
        ends[slice_dim] = (i + 1) * slice_size
        this_torch_output = padded_torch_input[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2]]
        output_shape = this_torch_output.shape
        output_shape = [
            1,
            1,
            output_shape[0] * output_shape[1] * output_shape[2],
            round_up(output_shape[3], core_x * pad_value),
        ]
        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            output_shape, parallel_config, 1
        )
        this_ttnn_output = ttnn.padded_slice(ttnn_input, begins, ends, strides, memory_config=memory_config)
        output = this_ttnn_output.cpu().to_torch_with_padded_shape()
        this_torch_output = this_torch_output[:, :, :, : output.shape[-1]]
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "dims, slice_size, cores",
    [
        [[1, 32, 32, 1024], 16, 32],
        [[1, 29, 29, 999], 16, 32],
        [[1, 29, 29, 510], 16, 16],
        [[1, 6, 58, 2048], 3, 64],
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("pad_value", [8, 32])
def test_slice_width_sharded_for_conv2d(device, dims, slice_dim, slice_size, cores, layout, input_dtype, pad_value):
    if input_dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is not supported in row major layout")

    orientation = ttnn.ShardOrientation.ROW_MAJOR
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x * core_grid.y < cores:
        pytest.skip(
            "Skipping test_slice_height_sharded_for_conv2d as device does not have enough Tensix cores. Needs %d, but device has %d"
            % (cores, core_grid.x * core_grid.y)
        )

    strides = [1, 1, 1, 1]
    torch.manual_seed(2001)
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randint(-10, 10, dims).to(dtype=torch_dtype)

    core_range = ttnn.num_cores_to_corerangeset(cores, core_grid, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    num_slices = dims[slice_dim] // slice_size
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, layout=layout, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range, shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED, shard_orientation=orientation
    )
    padded_channels = round_up(dims[-1], pad_value * cores)
    padded_torch_input = torch.nn.functional.pad(torch_input, (0, padded_channels - dims[-1]))
    torch.set_printoptions(sci_mode=False, precision=2)
    for i in range(num_slices):
        begins = [0, 0, 0, 0]
        ends = [dims[0], dims[1], dims[2], dims[3]]
        begins[slice_dim] = i * slice_size
        ends[slice_dim] = (i + 1) * slice_size
        this_torch_output = padded_torch_input[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2]]
        output_shape = this_torch_output.shape
        output_shape = [
            1,
            1,
            output_shape[0] * output_shape[1] * output_shape[2],
            round_up(output_shape[3], pad_value * cores),
        ]

        memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            output_shape, parallel_config, 1
        )
        this_ttnn_output = ttnn.padded_slice(ttnn_input, begins, ends, strides, memory_config=memory_config)
        output = this_ttnn_output.cpu().to_torch_with_padded_shape()
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)
