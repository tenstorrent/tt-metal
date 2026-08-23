# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.test_utils import round_up


def create_sharded_memory_config_from_parallel_config(tensor_shape, parallel_config, tile_size):
    """
    Create a sharded memory config from a parallel config.
    tensor_shape is expected to be [N, H, W, C] where N=1 and H=1.
    """
    grid = parallel_config.grid
    shard_scheme = parallel_config.shard_scheme
    shard_orientation = parallel_config.shard_orientation

    grid_size = grid.bounding_box().grid_size()
    num_cores = grid.num_cores()

    # Calculate num_cores_nhw
    if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        num_cores_nhw = 1
    elif shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores_nhw = num_cores
    elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
        num_cores_nhw = grid_size.x
    else:  # ROW_MAJOR
        num_cores_nhw = grid_size.y

    # Calculate num_cores_channels
    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores_channels = 1
    elif shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        num_cores_channels = num_cores
    elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
        num_cores_channels = grid_size.y
    else:  # ROW_MAJOR
        num_cores_channels = grid_size.x

    channels = tensor_shape[3]
    nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]

    if shard_scheme != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        nhw_padded = round_up(nhw_shape, num_cores_nhw * tile_size)
    else:
        nhw_padded = nhw_shape

    nhw_shard = nhw_padded // num_cores_nhw
    channel_shard = channels // num_cores_channels

    shard_spec = ttnn.ShardSpec(grid, (nhw_shard, channel_shard), shard_orientation)
    return ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)


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


@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_padded_slice_rm_aligned_row_misaligned_begin(device, input_buffer_type):
    """An aligned 16-element output row still needs the unaligned reader when begin=1."""
    torch_input = torch.arange(1 * 1 * 8 * 17, dtype=torch.bfloat16).reshape(1, 1, 8, 17)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_buffer_type)
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
    )

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, (8, 16), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output = ttnn.experimental.padded_slice(
        input_tensor,
        [0, 0, 0, 1],
        [1, 1, 8, 17],
        [1, 1, 1, 1],
        memory_config=output_memory_config,
    )
    assert torch.equal(torch_input[..., 1:17], ttnn.to_torch(output))


def test_padded_slice_rm_misaligned_row_and_begin_dram(device):
    """Exercise the staged reader when both the DRAM row width and begin offset are misaligned."""
    torch_input = torch.arange(1 * 1 * 8 * 9, dtype=torch.bfloat16).reshape(1, 1, 8, 9)
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, (8, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output = ttnn.experimental.padded_slice(
        input_tensor,
        [0, 0, 0, 1],
        [1, 1, 8, 9],
        [1, 1, 1, 1],
        memory_config=output_memory_config,
    )
    assert torch.equal(torch_input[..., 1:9], ttnn.to_torch(output))


_HEIGHT_SHARDED_DIMS = [
    [[2, 256, 300, 64], 128, 22],
    [[2, 256, 128, 32], 64, 8],
    [[2, 256, 256, 128], 64, 64],
    [[2, 256, 256, 9], 64, 64],
    [[2, 256, 256, 17], 64, 64],
    [[2, 1024, 1024, 3], 64, 64],
    [[2, 313, 71, 32], 32, 7],
]


def _slice_write_height_sharded_body(device, dims, slice_dim, slice_size, cores, layout, orientation, dtype):
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
        memory_config = create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ends[-1] = ttnn_output.shape[-1]
        ttnn.experimental.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


@pytest.mark.parametrize("dims, slice_size, cores", _HEIGHT_SHARDED_DIMS)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_slice_write_height_sharded(device, dims, slice_dim, slice_size, cores, layout):
    _slice_write_height_sharded_body(
        device, dims, slice_dim, slice_size, cores, layout, ttnn.ShardOrientation.ROW_MAJOR, ttnn.bfloat16
    )


# orientation/dtype crossing is spot-checked here (instead of fully crossing them with
# dims x slice_dim x layout above) to avoid a 7*2*2*2*2=224-case product; this covers each
# remaining (orientation, dtype) pair at least once against a representative subset of dims.
@pytest.mark.parametrize(
    "orientation, dtype",
    [
        (ttnn.ShardOrientation.ROW_MAJOR, ttnn.float32),
        (ttnn.ShardOrientation.COL_MAJOR, ttnn.bfloat16),
        (ttnn.ShardOrientation.COL_MAJOR, ttnn.float32),
    ],
)
@pytest.mark.parametrize("dims, slice_size, cores", _HEIGHT_SHARDED_DIMS[::2])
def test_slice_write_height_sharded_orientation_dtype_coverage(device, orientation, dtype, dims, slice_size, cores):
    _slice_write_height_sharded_body(device, dims, 1, slice_size, cores, ttnn.TILE_LAYOUT, orientation, dtype)


_WIDTH_SHARDED_DIMS = [
    [[2, 64, 64, 2048], 32, 64],
    [[2, 48, 48, 2944], 32, 46],
    [[2, 48, 48, 2904], 32, 46],
]


def _slice_write_width_sharded_body(device, dims, slice_dim, slice_size, cores, layout, orientation, dtype):
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
        memory_config = create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ends[-1] = ttnn_output.shape[-1]
        ttnn.experimental.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


@pytest.mark.parametrize("dims, slice_size, cores", _WIDTH_SHARDED_DIMS)
@pytest.mark.parametrize("slice_dim", [1, 2])
def test_slice_write_width_sharded(device, dims, slice_dim, slice_size, cores):
    _slice_write_width_sharded_body(
        device, dims, slice_dim, slice_size, cores, ttnn.TILE_LAYOUT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.bfloat16
    )


# orientation/dtype crossing spot-checked separately to avoid a 3*2*1*2*2=24-case product.
@pytest.mark.parametrize(
    "orientation, dtype",
    [
        (ttnn.ShardOrientation.ROW_MAJOR, ttnn.float32),
        (ttnn.ShardOrientation.COL_MAJOR, ttnn.bfloat16),
        (ttnn.ShardOrientation.COL_MAJOR, ttnn.float32),
    ],
)
@pytest.mark.parametrize("dims, slice_size, cores", _WIDTH_SHARDED_DIMS)
def test_slice_write_width_sharded_orientation_dtype_coverage(device, orientation, dtype, dims, slice_size, cores):
    _slice_write_width_sharded_body(device, dims, 1, slice_size, cores, ttnn.TILE_LAYOUT, orientation, dtype)


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
        memory_config = create_sharded_memory_config_from_parallel_config(
            this_ttnn_input.shape,
            parallel_config,
            32 if layout == ttnn.TILE_LAYOUT else 1,
        )

        this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, memory_config)
        ttnn.experimental.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


_HEIGHT_CONV2D_DIMS = [
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
]


def _slice_height_sharded_for_conv2d_body(device, dims, slice_dim, slice_size, cores, layout, input_dtype, pad_value):
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

        memory_config = create_sharded_memory_config_from_parallel_config(output_shape, parallel_config, 1)
        this_ttnn_output = ttnn.experimental.padded_slice(
            ttnn_input, begins, ends, strides, memory_config=memory_config
        )
        output = ttnn.to_torch(this_ttnn_output)
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dims, slice_size, cores", _HEIGHT_CONV2D_DIMS)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_height_sharded_for_conv2d(device, dims, slice_dim, slice_size, cores, layout):
    _slice_height_sharded_for_conv2d_body(device, dims, slice_dim, slice_size, cores, layout, ttnn.bfloat16, 32)


# input_dtype/pad_value crossing is spot-checked here (instead of fully crossing them with
# dims x slice_dim x layout above) to avoid a 12*2*2*3*3=432-case product; this covers every
# remaining (input_dtype, pad_value) pair at least once against a representative subset of dims.
@pytest.mark.parametrize(
    "input_dtype, pad_value",
    [
        (ttnn.bfloat8_b, 8),
        (ttnn.bfloat8_b, 16),
        (ttnn.bfloat8_b, 32),
        (ttnn.bfloat16, 8),
        (ttnn.bfloat16, 16),
        (ttnn.float32, 8),
        (ttnn.float32, 16),
        (ttnn.float32, 32),
    ],
)
@pytest.mark.parametrize("dims, slice_size, cores", _HEIGHT_CONV2D_DIMS[::3])
def test_slice_height_sharded_for_conv2d_dtype_pad_coverage(device, input_dtype, pad_value, dims, slice_size, cores):
    _slice_height_sharded_for_conv2d_body(device, dims, 1, slice_size, cores, ttnn.TILE_LAYOUT, input_dtype, pad_value)


_BLOCK_CONV2D_DIMS = [
    [[2, 64, 64, 256], 32, 4, 4],
    [[2, 64, 64, 512], 16, 4, 4],
    [[2, 16, 16, 1024], 4, 4, 4],
    [[2, 128, 128, 256], 32, 8, 4],
    [[2, 128, 128, 63], 32, 8, 2],
    [[2, 128, 128, 528], 96, 8, 6],
    [[2, 128, 128, 96], 96, 8, 3],
    [[2, 1024, 1024, 256], 33, 10, 11],
    [[1, 64, 128, 256], 65, 4, 5],
]


def _slice_block_sharded_for_conv2d_body(
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
        memory_config = create_sharded_memory_config_from_parallel_config(output_shape, parallel_config, 1)
        this_ttnn_output = ttnn.experimental.padded_slice(
            ttnn_input, begins, ends, strides, memory_config=memory_config
        )
        output = this_ttnn_output.cpu().to_torch_with_padded_shape()
        this_torch_output = this_torch_output[:, :, :, : output.shape[-1]]
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dims, slice_size, core_y, core_x", _BLOCK_CONV2D_DIMS)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_block_sharded_for_conv2d(device, dims, slice_dim, slice_size, core_x, core_y, layout):
    _slice_block_sharded_for_conv2d_body(device, dims, slice_dim, slice_size, core_x, core_y, layout, ttnn.bfloat16, 32)


# input_dtype/pad_value crossing is spot-checked here (instead of fully crossing them with
# dims x slice_dim x layout above) to avoid a 9*2*2*3*2=216-case product; this covers every
# remaining (input_dtype, pad_value) pair at least once against a representative subset of dims.
@pytest.mark.parametrize(
    "input_dtype, pad_value",
    [
        (ttnn.bfloat8_b, 8),
        (ttnn.bfloat8_b, 32),
        (ttnn.bfloat16, 8),
        (ttnn.float32, 8),
        (ttnn.float32, 32),
    ],
)
@pytest.mark.parametrize("dims, slice_size, core_y, core_x", _BLOCK_CONV2D_DIMS[::3])
def test_slice_block_sharded_for_conv2d_dtype_pad_coverage(
    device, input_dtype, pad_value, dims, slice_size, core_x, core_y
):
    _slice_block_sharded_for_conv2d_body(
        device, dims, 1, slice_size, core_x, core_y, ttnn.TILE_LAYOUT, input_dtype, pad_value
    )


_WIDTH_CONV2D_DIMS = [
    [[1, 32, 32, 1024], 16, 32],
    [[1, 29, 29, 999], 16, 32],
    [[1, 29, 29, 510], 16, 16],
    [[1, 6, 58, 2048], 3, 64],
]


def _slice_width_sharded_for_conv2d_body(device, dims, slice_dim, slice_size, cores, layout, input_dtype, pad_value):
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

        memory_config = create_sharded_memory_config_from_parallel_config(output_shape, parallel_config, 1)
        this_ttnn_output = ttnn.experimental.padded_slice(
            ttnn_input, begins, ends, strides, memory_config=memory_config
        )
        output = this_ttnn_output.cpu().to_torch_with_padded_shape()
        output = torch.reshape(output, this_torch_output.shape)
        assert torch.allclose(this_torch_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dims, slice_size, cores", _WIDTH_CONV2D_DIMS)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_width_sharded_for_conv2d(device, dims, slice_dim, slice_size, cores, layout):
    _slice_width_sharded_for_conv2d_body(device, dims, slice_dim, slice_size, cores, layout, ttnn.bfloat16, 32)


# input_dtype/pad_value crossing is spot-checked here (instead of fully crossing them with
# dims x slice_dim x layout above) to avoid a 4*2*2*3*2=96-case product; this covers every
# remaining (input_dtype, pad_value) pair at least once against all (small) dims.
@pytest.mark.parametrize(
    "input_dtype, pad_value",
    [
        (ttnn.bfloat8_b, 8),
        (ttnn.bfloat8_b, 32),
        (ttnn.bfloat16, 8),
        (ttnn.float32, 8),
        (ttnn.float32, 32),
    ],
)
@pytest.mark.parametrize("dims, slice_size, cores", _WIDTH_CONV2D_DIMS)
def test_slice_width_sharded_for_conv2d_dtype_pad_coverage(device, input_dtype, pad_value, dims, slice_size, cores):
    _slice_width_sharded_for_conv2d_body(device, dims, 1, slice_size, cores, ttnn.TILE_LAYOUT, input_dtype, pad_value)
