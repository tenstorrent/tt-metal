# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.test_utils import round_up
import math
import random


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def _rand_shape(dim_size, choices=(8, 16, 32, 64, 128, 256)):
    # Choose a reasonable size per dim (keeps DRAM/L1 happy)
    ret = [random.choice(choices) for _ in range(dim_size)]
    if dim_size > 5:
        ret[: dim_size - 5] = [1 for _ in range(dim_size - 5)]  # rank 6+ tensors are too big
    return ret


def _rand_slice_params(shape):
    begins, ends, strides = [], [], []
    ndim = len(shape)
    for dim, size in enumerate(shape):
        s = 1  # stride always 1
        if dim == ndim - 1:
            # last dimension: take the whole thing
            b, e = 0, size
        else:
            # pick a random start < size
            b = random.randint(0, size - 1)
            # choose a random end > b, ≤ size
            e = random.randint(b + 1, size)
        begins.append(b)
        ends.append(e)
        strides.append(s)
    return begins, ends, strides


@pytest.mark.parametrize("rank", range(2, 9))  # 1D … 8D
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_slice(rank, layout, device):
    base_seed = 2005
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    shape = _rand_shape(rank)
    begins, ends, strides = _rand_slice_params(shape)
    print(f"Testing slice_write with shape {shape}, begins {begins}, ends {ends}, strides {strides}")
    # Build PyTorch reference slice
    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, strides))

    # Destination and source (match slice shape)
    torch_out_ref = torch.zeros(shape, dtype=torch.bfloat16)
    torch_src = torch.rand(torch_out_ref[slices].shape, dtype=torch.bfloat16)

    # PyTorch ground truth
    torch_out_ref[slices] = torch_src

    # TTNN copies
    tt_out = ttnn.from_torch(torch_out_ref * 0, device=device, layout=layout, dtype=ttnn.bfloat16)
    tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)

    tt_in = ttnn.from_torch(torch_src, device=device, layout=layout, dtype=ttnn.bfloat16)
    tt_in = ttnn.to_memory_config(tt_in, ttnn.L1_MEMORY_CONFIG)

    # Perform the slice write
    ttnn.slice_write(tt_in, tt_out, begins, ends, strides)

    # Compare full tensors and the written region explicitly
    out_host = ttnn.to_torch(tt_out)
    written_region = out_host[slices]

    assert_with_pcc(written_region, torch_src, 0.9999)
    assert_with_pcc(out_host, torch_out_ref, 0.9999)


@pytest.mark.parametrize(
    "dims, begins, ends",
    [
        [[16, 256, 256, 64], [0, 0, 0, 0], [1, 1, 256, 64]],
        [[1, 256, 128, 64], [0, 128, 0, 0], [1, 256, 128, 64]],
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_slice_write_four_dim(dims, begins, ends, layout, device):
    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_output = torch.zeros(dims)
    slices = []
    for i in range(len(dims)):
        slices.append(slice(begins[i], ends[i], strides[i]))

    torch_input = torch_output[slices[0], slices[1], slices[2], slices[3]]
    torch_input = torch.rand(torch_input.shape)

    ttnn_output = ttnn.from_torch(torch_output, device=device, layout=layout, dtype=ttnn.bfloat16)
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)
    ttnn_input = ttnn.to_memory_config(ttnn_input, ttnn.L1_MEMORY_CONFIG)
    ttnn.slice_write(ttnn_input, ttnn_output, begins, ends, strides)
    output = ttnn.to_torch(ttnn_output)
    torch_output[slices[0], slices[1], slices[2], slices[3]] = torch_input
    written_output = output[slices[0], slices[1], slices[2], slices[3]]
    # assert False
    assert_with_pcc(written_output, torch_input, 0.9999)
    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize(
    "dims, slice_dim, slice_size",
    [[[2, 256, 256, 64], 1, 128], [[2, 256, 128, 32], 2, 16], [[1, 46, 46, 2904], 2, 23]],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_slice_write_copy(device, dims, slice_dim, slice_size, layout):
    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randn(dims)
    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=ttnn.bfloat16)
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
    for b in range(dims[0]):
        for i in range(dims[slice_dim] // slice_size):
            begins = [b, 0, 0, 0]
            ends = [b + 1, dims[1], dims[2], dims[3]]
            begins[slice_dim] = i * slice_size
            ends[slice_dim] = (i + 1) * slice_size
            this_ttnn_input = ttnn.from_torch(
                torch_input[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : ends[3]],
                device=device,
                layout=layout,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            this_ttnn_input = ttnn.to_memory_config(this_ttnn_input, ttnn.L1_MEMORY_CONFIG)
            ttnn.slice_write(this_ttnn_input, ttnn_output, begins, ends, strides)

    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_input, output, 0.9999)


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
def test_slice_write_height_sharded(device, dims, slice_dim, slice_size, cores, layout, orientation):
    core_grid = device.compute_with_storage_grid_size()

    if core_grid.x * core_grid.y < cores:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)

    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=ttnn.bfloat16)
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
            dtype=ttnn.bfloat16,
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
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_slice_write_width_sharded(device, dims, slice_dim, slice_size, cores, layout, orientation):
    core_grid = device.compute_with_storage_grid_size()

    if core_grid.x * core_grid.y < cores:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)

    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=ttnn.bfloat16)
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
            dtype=ttnn.bfloat16,
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
    ],
)
@pytest.mark.parametrize("slice_dim", [1, 2])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_slice_write_block_sharded(device, dims, slice_dim, slice_size, core_x, core_y, layout, orientation):
    core_grid = device.core_grid
    if core_grid.x < core_x or core_grid.y < core_y:
        pytest.skip("Device does not have enough cores")

    strides = [1, 1, 1, 1]
    torch.manual_seed(2005)
    torch_input = torch.randint(-10, 10, dims)
    ttnn_output = ttnn.zeros(dims, device=device, layout=layout, dtype=ttnn.bfloat16)
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
            dtype=ttnn.bfloat16,
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
