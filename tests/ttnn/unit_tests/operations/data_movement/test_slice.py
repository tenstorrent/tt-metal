# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.common.utility_functions import is_grayskull, is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.test_utils import round_up
import math


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def run_slice_rm_sharded(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    n_unpadded = n
    c_unpadded = 115
    h_unpadded = 115
    torch_output_tensor = torch_input_tensor[:n_unpadded, :c_unpadded, :h_unpadded, :]
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # shard config
    num_cores_x = 8
    num_cores_y = 7
    shard_h = (n * c * h + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    # output shard config
    num_cores_x = 8
    num_cores_y = 7
    shard_h = (n_unpadded * c_unpadded * h_unpadded + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    tt_output_tensor = ttnn.slice(
        tt_input_tensor,
        (0, 0, 0, 0),
        (n_unpadded, c_unpadded, h_unpadded, w),
        memory_config=output_mem_config,
    )
    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize(
    "dims, begins, ends",
    [
        [[16, 256, 256, 64], [0, 0, 0, 0], [1, 1, 256, 64]],
        [[1, 256, 128, 64], [0, 128, 0, 0], [1, 256, 128, 64]],
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_slice_write_four_dim(dims, begins, ends, layout, dtype, device):
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


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_slice_rm_sharded_with_program_cache(device, n, c, h, w):
    for _ in range(2):
        run_slice_rm_sharded(device, n, c, h, w)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 3


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_slice_rm(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor[:, :115, 2:115, :]
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.slice(
        activation_pyt_padded,
        (0, 0, 2, 0),
        (n, 115, 115, w),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded_out = ttnn.to_memory_config(activation_pyt_padded, ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.from_device(activation_pyt_padded_out)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded_out)
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


def slice_test(
    input_layout,
    input_tensor_shape,
    output_tensor_start,
    output_tensor_end,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
    slice_step=(1, 1, 1, 1),
):
    if dtype == ttnn.float32:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.float)
    else:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_layout, device=device, memory_config=in_mem_config
    )

    tt_output_tensor = ttnn.slice(
        tt_input_tensor,
        slice_start=output_tensor_start,
        slice_end=output_tensor_end,
        slice_step=slice_step,
        memory_config=out_mem_config,
    )

    a_pt = ttnn.to_torch(tt_output_tensor)

    # Pytorch reference
    a_ref = torch_input_tensor[
        output_tensor_start[0] : output_tensor_end[0] : slice_step[0],
        output_tensor_start[1] : output_tensor_end[1] : slice_step[1],
        output_tensor_start[2] : output_tensor_end[2] : slice_step[2],
        output_tensor_start[3] : output_tensor_end[3] : slice_step[3],
    ]

    return a_pt, a_ref, device.num_program_cache_entries()


# from https://github.com/tenstorrent/tt-metal/issues/23237
def test_slice_rm_program_cache_collison(device):
    shape = (32, 64, 4096)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for i in range(shape[-1]):
        tt_out = tt_input[:, :, i : i + 1]
        torch_out = torch_input[:, :, i : i + 1]
        assert_with_pcc(torch_out, ttnn.to_torch(tt_out), 0.99)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "input_tensor_shape_0, output_tensor_start_0, output_tensor_end_0",
    (
        ((4, 3, 64, 64), (0, 0, 0, 0), (4, 3, 32, 32)),
        ((1, 1, 64, 64), (0, 0, 0, 0), (1, 1, 32, 64)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
        ((1, 3, 32, 32), (0, 1, 0, 0), (1, 2, 32, 32)),
        ((1, 6, 32, 32), (0, 2, 0, 0), (1, 4, 32, 32)),
        ((1, 6, 128, 64), (0, 2, 64, 32), (1, 4, 96, 64)),
        ((4, 6, 128, 64), (1, 2, 64, 32), (2, 4, 96, 64)),
    ),
)
@pytest.mark.parametrize(
    "input_tensor_shape_1, output_tensor_start_1, output_tensor_end_1",
    (((9, 8, 128, 128), (0, 0, 0, 0), (9, 8, 32, 32)),),
)
@pytest.mark.parametrize(
    "slice_step",
    ((1, 1, 1, 1),),
)
def test_run_slice_test(
    input_tensor_shape_0,
    output_tensor_start_0,
    output_tensor_end_0,
    input_tensor_shape_1,
    output_tensor_start_1,
    output_tensor_end_1,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
    slice_step,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor_shape_0,
        output_tensor_start_0,
        output_tensor_end_0,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
        slice_step,
    )
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
    assert num_cache_entries == 1

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor_shape_1,
        output_tensor_start_1,
        output_tensor_end_1,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
        slice_step,
    )
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
    # different width for row major
    assert num_cache_entries == 2

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.TILE_LAYOUT,
        input_tensor_shape_0,
        output_tensor_start_0,
        output_tensor_end_0,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
        slice_step,
    )
    # change from RM to TILE
    assert num_cache_entries == 3
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.TILE_LAYOUT,
        input_tensor_shape_1,
        output_tensor_start_1,
        output_tensor_end_1,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
        slice_step,
    )
    # CACHE HIT
    assert num_cache_entries == 4
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_start, output_tensor_end",
    (
        ((4, 3, 640, 640), (0, 0, 0, 0), (4, 3, 320, 320)),
        ((4, 3, 64, 64), (0, 0, 0, 0), (4, 3, 32, 32)),
        ((1, 1, 64, 64), (0, 0, 0, 0), (1, 1, 32, 64)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (1, 1, 96, 96)),
        ((1, 1, 128, 96), (0, 0, 64, 33), (1, 1, 96, 96)),
        ((1, 3, 32, 32), (0, 1, 0, 0), (1, 2, 32, 32)),
        ((1, 6, 32, 32), (0, 2, 0, 0), (1, 4, 32, 32)),
        ((1, 6, 128, 64), (0, 2, 64, 32), (1, 4, 96, 64)),
        ((4, 6, 128, 64), (1, 2, 64, 32), (2, 4, 96, 64)),
        ((4, 6, 128, 64), (1, 2, 64, 33), (2, 4, 96, 64)),
    ),
)
@pytest.mark.parametrize(
    "slice_step",
    (
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (1, 3, 2, 5),
    ),
)
def test_run_slice_rm_multi_core_test(
    input_tensor_shape,
    output_tensor_start,
    output_tensor_end,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
    slice_step,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor_shape,
        output_tensor_start,
        output_tensor_end,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
        slice_step,
    )
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
    assert num_cache_entries == 1


# slice alternate elements in a given tensor
@pytest.mark.parametrize("dim", [4, 12, 20, 68])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.uint8])
def test_stride_slice_single_dim_skip_2(dim, dtype, device):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, (dim,))
    torch_output = torch_input[::2]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=dtype)
    ttnn_output = ttnn_input[::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("h", [18, 34])
@pytest.mark.parametrize("w", [18, 34])
@pytest.mark.parametrize("begins_h", [2])
@pytest.mark.parametrize("begins_w", [2])
@pytest.mark.parametrize("stride_h", [2])
@pytest.mark.parametrize("stride_w", [2])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.uint8])
def test_stride_slice_two_dim(h, w, begins_h, begins_w, stride_h, stride_w, dtype, device):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, (h, w))
    torch_output = torch_input[begins_h:h:stride_h, begins_w:w:stride_w]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=dtype)
    ttnn_output = ttnn_input[begins_h::stride_h, begins_w::stride_w]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("c", [8])
@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [2])
@pytest.mark.parametrize("begins_c", [0])
@pytest.mark.parametrize("begins_h", [0])
@pytest.mark.parametrize("begins_w", [0])
@pytest.mark.parametrize("stride_c", [2])
@pytest.mark.parametrize("stride_h", [1])
@pytest.mark.parametrize("stride_w", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.uint8])
def test_stride_slice_three_dim(c, h, w, begins_c, begins_h, begins_w, stride_c, stride_h, stride_w, dtype, device):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, (c, h, w))
    torch_output = torch_input[begins_c:c:stride_c, begins_h:h:stride_h, begins_w:w:stride_w]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=dtype)
    ttnn_output = ttnn_input[begins_c:c:stride_c, begins_h:h:stride_h, begins_w:w:stride_w]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("dims", [[18, 18, 18, 18]])
@pytest.mark.parametrize("begins", [[2, 0, 0, 2]])
@pytest.mark.parametrize("ends", [[18, 16, 16, 18]])
@pytest.mark.parametrize("strides", [[2, 2, 2, 2]])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.uint8])
def test_stride_slice_four_dim(dims, begins, ends, strides, layout, dtype, device):
    # Skip if tiled and uint8/16
    if layout == ttnn.TILE_LAYOUT and (dtype == ttnn.uint16 or dtype == ttnn.uint8):
        pytest.skip("Skipping test for tiled layout with uint8/16 dtype")
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, dims)
    slices = []
    for i in range(len(dims)):
        slices.append(slice(begins[i], ends[i], strides[i]))

    torch_output = torch_input[slices[0], slices[1], slices[2], slices[3]]

    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=dtype)
    ttnn_output = ttnn_input[slices[0], slices[1], slices[2], slices[3]]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("dims", [[1, 56, 56, 96]])
@pytest.mark.parametrize("begins", [[0, 0, 0, 0], [0, 0, 0, 90]])
@pytest.mark.parametrize("ends", [[1, -1, 56, 96], [1, 56, 56, 95], [-1, 1, -1, -1]])
@pytest.mark.parametrize("strides", [[1, 2, 1, 1]])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.uint8])
def test_stride_slice_four_dim_tiled(dims, begins, ends, strides, layout, dtype, device):
    if layout == ttnn.TILE_LAYOUT and (dtype == ttnn.uint16 or dtype == ttnn.uint8):
        pytest.skip("Skipping test for tiled layout with uint8/16 dtype")
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, dims)
    slices = []
    for i in range(len(dims)):
        slices.append(slice(begins[i], ends[i], strides[i]))

    torch_output = torch_input[slices[0], slices[1], slices[2], slices[3]]

    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=dtype)
    ttnn_output = ttnn_input[slices[0], slices[1], slices[2], slices[3]]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


# these tests are copy and paste from the yolo customers #8920
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_usecase1(layout, device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, ::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_usecase2(layout, device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, 1::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, 1::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_usecase3(layout, device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., 1::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., 1::2, ::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_slice_usecase4(layout, device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., 1::2, 1::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., 1::2, 1::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_ellipses(device):
    torch_input = torch.randn(32, 32, 32, 32)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[...]
    ttnn_output = ttnn_input[...]
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("ends", [-2, -4, -6, -32])
def test_slice_negative_ends(layout, dim, ends, device):
    torch_input = torch.randn(32, 32, 32, 32)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    if dim == 3:
        if layout == ttnn.ROW_MAJOR_LAYOUT and ends == -32:
            pytest.skip("Page size will become 0 and we don't handle transforming pages to second last dimension")
        torch_output = torch_input[:, :, :, 0:ends]
        ttnn_output = ttnn_input[:, :, :, 0:ends]
    elif dim == 2:
        torch_output = torch_input[:, :, 0:ends, :]
        ttnn_output = ttnn_input[:, :, 0:ends, :]
    elif dim == 1:
        torch_output = torch_input[:, 0:ends, :, :]
        ttnn_output = ttnn_input[:, 0:ends, :, :]
    elif dim == 0:
        torch_output = torch_input[0:ends, :, :, :]
        ttnn_output = ttnn_input[0:ends, :, :, :]

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends",
    (
        ((1, 1, 1, 256), (0, 0, 0, 0), (1, 1, 1, -1)),
        ((1, 256), (0, 0), (-1, 256)),
        ((1, 512), (0, 0), (-1, 512)),
        ((1, 512), (0, 0), (1, 256)),
        ((1, 32, 32, 32), (0, 0, 0, 0), (1, 32, 32, 1)),
        ((1, 32, 32, 64), (0, 0, 0, 0), (1, 32, 32, 32)),
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
def test_slice_bert(input_shape, input_start, input_ends, layout, device):
    if layout == ttnn.TILE_LAYOUT:
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
    else:
        if (input_ends[-1] - input_start[-1]) % 2 != 0:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    if len(input_shape) == 4:
        torch_output = torch_input[
            input_start[0] : input_ends[0],
            input_start[1] : input_ends[1],
            input_start[2] : input_ends[2],
            input_start[3] : input_ends[3],
        ]
        ttnn_output = ttnn_input[
            input_start[0] : input_ends[0],
            input_start[1] : input_ends[1],
            input_start[2] : input_ends[2],
            input_start[3] : input_ends[3],
        ]
    elif len(input_shape) == 2:
        torch_output = torch_input[input_start[0] : input_ends[0], input_start[1] : input_ends[1]]
        ttnn_output = ttnn_input[input_start[0] : input_ends[0], input_start[1] : input_ends[1]]

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends",
    (
        ((1, 1, 1, 256), (0, 0, 0, 0), (1, 1, 1, -1)),
        ((1, 256), (0, 0), (-1, 256)),
        ((1, 512), (0, 0), (-1, 512)),
        ((1, 512), (0, 0), (1, 256)),
        ((1, 32, 32, 32), (0, 0, 0, 0), (1, 32, 32, 1)),
        ((1, 32, 32, 64), (0, 0, 0, 0), (1, 32, 32, 32)),
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
@pytest.mark.parametrize(
    "memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
)
def test_ttnn_slice_bert(input_shape, input_start, input_ends, layout, memory_config, device):
    if layout == ttnn.TILE_LAYOUT:
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
    else:
        if (input_ends[-1] - input_start[-1]) % 2 != 0:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    if len(input_shape) == 4:
        torch_output = torch_input[
            input_start[0] : input_ends[0],
            input_start[1] : input_ends[1],
            input_start[2] : input_ends[2],
            input_start[3] : input_ends[3],
        ]
    elif len(input_shape) == 2:
        torch_output = torch_input[input_start[0] : input_ends[0], input_start[1] : input_ends[1]]

    ttnn_output = ttnn.slice(ttnn_input, input_start, input_ends, memory_config=memory_config)

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_output_tensor_rm(device):
    torch_input = torch.ones(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
    torch_zeros = torch.zeros(1, 3, 320, 320)
    ttnn_output = ttnn.from_torch(torch_zeros, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_input[..., ::2, ::2]  # torch_output shape: [1, 3, 320, 320]

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1), output_tensor=ttnn_output)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_output_tensor_tile(device):
    torch_input = torch.ones(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    torch_zeros = torch.zeros(1, 3, 320, 320)
    ttnn_output = ttnn.from_torch(
        torch_zeros, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_output = torch_input[..., ::2, ::2]  # torch_output shape: [1, 3, 320, 320]

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1), output_tensor=ttnn_output)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends",
    (
        ((1, 1, 1, 256), (0, 0, 0, 0), (1, 1, 1, 255)),
        ((1, 32, 32, 32), (0, 0, 0, 0), (1, 32, 32, 1)),
        ((1, 32, 32, 64), (0, 0, 0, 0), (1, 32, 1, 32)),
        ((1, 1, 64, 64), (0, 0, 0, 0), (1, 1, 1, 1)),
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
@pytest.mark.parametrize(
    "memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
)
def test_ttnn_slice_optimized_shapes(input_shape, input_start, input_ends, layout, memory_config, device):
    if layout == ttnn.TILE_LAYOUT:
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
    else:
        if (input_ends[-1] - input_start[-1]) % 2:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    torch_output = torch_input[
        input_start[0] : input_ends[0],
        input_start[1] : input_ends[1],
        input_start[2] : input_ends[2],
        input_start[3] : input_ends[3],
    ]

    ttnn_output = ttnn.slice(
        ttnn_input, starts=input_start, ends=input_ends, steps=(1, 1, 1, 1), memory_config=memory_config
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends",
    (
        ((1, 1, 1, 1, 256), (0, 0, 0, 0, 0), (1, 1, 1, 1, 255)),
        ((1, 1, 32, 32, 32), (0, 0, 0, 0, 0), (1, 1, 32, 32, 1)),
        ((1, 1, 32, 32, 64), (0, 0, 0, 0, 0), (1, 1, 32, 1, 32)),
        ((1, 1, 1, 64, 64), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1)),
        ((4, 3, 2, 1, 4), (1, 1, 1, 0, 0), (1, 1, 2, 1, 4)),
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
@pytest.mark.parametrize(
    "memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
)
def test_ttnn_slice_5d(input_shape, input_start, input_ends, layout, memory_config, device):
    if layout == ttnn.TILE_LAYOUT:
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
    else:
        if (input_ends[-1] - input_start[-1]) % 2:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    torch_output = torch_input[
        input_start[0] : input_ends[0],
        input_start[1] : input_ends[1],
        input_start[2] : input_ends[2],
        input_start[3] : input_ends[3],
        input_start[4] : input_ends[4],
    ]

    ttnn_output = ttnn.slice(ttnn_input, input_start, input_ends, (1, 1, 1, 1, 1), memory_config=memory_config)

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends, input_stride",
    (
        ((1, 1, 5, 1, 256), (0, 0, 0, 0, 0), (1, 1, 1, 1, 234), (1, 1, 1, 1, 1)),
        ((1, 2, 32, 32, 32), (0, 0, 0, 0, 0), (1, 1, 32, 32, 1), (1, 1, 1, 1, 1)),
        ((1, 1, 32, 32, 64), (0, 0, 0, 0, 0), (1, 1, 32, 1, 32), (1, 1, 2, 1, 2)),
        ((2, 1, 1, 64, 64), (1, 0, 0, 0, 0), (2, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((4, 3, 2, 1, 18), (1, 1, 1, 0, 0), (1, 1, 2, 1, -2), (1, 1, 1, 1, 2)),
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
def test_slice_5d(input_shape, input_start, input_ends, input_stride, layout, device):
    if layout == ttnn.TILE_LAYOUT:
        if input_stride != (1, 1, 1, 1, 1):
            pytest.skip("Cannot untilize 5D tensor")
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
    else:
        if (input_ends[-1] - input_start[-1]) % 2:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)

    torch_output = torch_input[
        input_start[0] : input_ends[0] : input_stride[0],
        input_start[1] : input_ends[1] : input_stride[1],
        input_start[2] : input_ends[2] : input_stride[2],
        input_start[3] : input_ends[3] : input_stride[3],
        input_start[4] : input_ends[4] : input_stride[4],
    ]
    ttnn_output = ttnn_input[
        input_start[0] : input_ends[0] : input_stride[0],
        input_start[1] : input_ends[1] : input_stride[1],
        input_start[2] : input_ends[2] : input_stride[2],
        input_start[3] : input_ends[3] : input_stride[3],
        input_start[4] : input_ends[4] : input_stride[4],
    ]

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_7d_strided(device):
    torch_input = torch.randn(1, 1, 1, 1, 1, 1, 256)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_output = torch_input[..., 0:1, 0:1, 0:1, 0:1, 0:1, 0:256:2]
    ttnn_output = ttnn_input[..., 0:1, 0:1, 0:1, 0:1, 0:1, 0:256:2]

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_7d(device):
    torch_input = torch.randn(1, 1, 1, 1, 1, 1, 256)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    torch_output = torch_input[..., 0:1, 0:1, 0:1, 0:1, 0:1, 0:200]
    ttnn_output = ttnn_input[..., 0:1, 0:1, 0:1, 0:1, 0:1, 0:200]

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, dim, start, end, step, layout",
    (
        ([1, 28, 56, 96], 2, 0, -1, 2, ttnn.TILE_LAYOUT),  # Formerly bad pcc
        ([1, 56, 56, 96], 1, 0, -1, 2, ttnn.TILE_LAYOUT),  # Formerly bad pcc
        ([8732, 4], 1, 0, 2, 1, ttnn.ROW_MAJOR_LAYOUT),  # Formerly bad pcc
        ([1, 14, 28, 192], 2, 1, -1, 2, ttnn.TILE_LAYOUT),  # Bad pcc on sweeps but not on unit test (low priority)
        ([1, 23, 40, 128], 3, 0, -1, 2, ttnn.TILE_LAYOUT),  # Bad pcc on sweeps but not on unit test
        ([1, 28, 28, 256], 1, 1, -1, 2, ttnn.TILE_LAYOUT),  # Bad pcc on sweeps but not on unit test
        (
            [1, 3],
            1,
            0,
            -1,
            1,
            ttnn.TILE_LAYOUT,
        ),  # works when you turn it into a 2D tensor (compared to [3] example in the next test)
    ),
)
def test_slice_adversarial_fixed(input_shape, dim, start, end, step, layout, device):
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    slice_obj = slice(start, end, step)

    # Prepare indices for slicing in the specified dimension
    indices = [slice(None)] * len(input_shape)  # By default, select all elements along every dimension
    indices[dim] = slice_obj  # Apply slicing to the target dimension
    indices = tuple(indices)

    # Apply slicing to the input_tensor
    torch_output_tensor = torch_input[indices]

    ttnn_tensor = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)
    ttnn_output = ttnn_tensor[indices]

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)


# Op parameters from pytorch2 sweep tests that failed prior to 2025-03
@pytest.mark.parametrize(
    "input_shape, dim, start, end, step, layout",
    (
        ((1, 145, 768), 1, 1, -1, 1, ttnn.TILE_LAYOUT),  # tile-unaligned slice start were previously not supported
        ((1, 1445, 192), 1, -100, -1, 1, ttnn.TILE_LAYOUT),  # tile-unaligned slice start were previously not supported
        ([1, 7], 0, 0, -1, 1, ttnn.ROW_MAJOR_LAYOUT),  # page size must equal buffer size
        ([1, 8, 2, 2], 2, -1, -1, 1, ttnn.TILE_LAYOUT),  # Buffer size and page size should be larger than 0 bytes
        ([8732, 4], 1, 0, -1, 4, ttnn.TILE_LAYOUT),  # Need tensor for this or a padding aware tiled kernel
        ([3], 0, 0, -1, 1, ttnn.TILE_LAYOUT),  # unaligned 1D
        (
            [1, 7, 71, 64],
            3,
            0,
            -1,
            1,
            ttnn.ROW_MAJOR_LAYOUT,
        ),  # An unpadding slice operations for a RowMajor layout on the output tensor requires the last dimension to be on a 32 bit boundary
    ),
)
def test_slice_former_pytorch2_failures(input_shape, dim, start, end, step, layout, device):
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    slice_obj = slice(start, end, step)

    # Prepare indices for slicing in the specified dimension
    indices = [slice(None)] * len(input_shape)  # By default, select all elements along every dimension
    indices[dim] = slice_obj  # Apply slicing to the target dimension
    indices = tuple(indices)

    # Apply slicing to the input_tensor
    torch_output_tensor = torch_input[indices]

    ttnn_tensor = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=ttnn.bfloat16)
    ttnn_output = ttnn_tensor[indices]

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shape",
    ([8, 8, 8, 33, 33],),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
)
@pytest.mark.parametrize(
    "input_memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
)
@pytest.mark.parametrize(
    "indices",
    (
        [0, 0, 0, slice(0, 33, 1), slice(0, 33, 1)],
        [1, -1, 2, slice(0, 33, 1), slice(0, 33, 1)],
    ),
)
def test_slice_index(device, input_shape, layout, input_memory_config, indices):
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=input_memory_config, layout=layout
    )

    torch_output = torch_input[
        indices[0],
        indices[1],
        indices[2],
        indices[3],
        indices[4],
    ]

    ttnn_output = ttnn_input[
        indices[0],
        indices[1],
        indices[2],
        indices[3],
        indices[4],
    ]

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "input_shape, input_start, input_ends, input_steps",
    (
        ((1, 1504, 1280), (0, 0, 0), (1, 1500, 1280), (1, 1, 1)),  # fill pad case
        ((448, 1280), (0, 0), (1, 1280), (1, 1)),  # fill pad case
    ),
)
@pytest.mark.parametrize(
    "layout",
    (ttnn.TILE_LAYOUT,),
)
@pytest.mark.parametrize(
    "memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
)
def test_ttnn_slice_whisper(input_shape, input_start, input_ends, input_steps, memory_config, layout, device):
    # A couple of slices in whisper that only work for "logical" slicing.

    for _ in range(3):
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=layout)
        if len(input_shape) == 4:
            torch_output = torch_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
                input_start[2] : input_ends[2] : input_steps[2],
                input_start[3] : input_ends[3] : input_steps[3],
            ]
            ttnn_output = ttnn_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
                input_start[2] : input_ends[2] : input_steps[2],
                input_start[3] : input_ends[3] : input_steps[3],
            ]

        if len(input_shape) == 3:
            torch_output = torch_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
                input_start[2] : input_ends[2] : input_steps[2],
            ]
            ttnn_output = ttnn_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
                input_start[2] : input_ends[2] : input_steps[2],
            ]

        if len(input_shape) == 2:
            torch_output = torch_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
            ]
            ttnn_output = ttnn_input[
                input_start[0] : input_ends[0] : input_steps[0],
                input_start[1] : input_ends[1] : input_steps[1],
            ]

        ttnn_output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, ttnn_output, 0.999)


@pytest.mark.parametrize(
    "input_shape, dim, start, end, step, layout",
    (
        # ([4, 4], 1, [0, 1], [1, 2], 1, ttnn.ROW_MAJOR_LAYOUT),
        # ([1, 28, 56, 96], 2, [0, 0, 0, 0], [1, 16, 32, 32], 2, ttnn.ROW_MAJOR_LAYOUT),
        # ([10], 1, [2], [7], 1, ttnn.ROW_MAJOR_LAYOUT),
        ([3, 16, 128, 2880], 2, [0, 0, 32, 0], [3, 16, 64, 2880], 1, ttnn.TILE_LAYOUT),
    ),
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 32), id="1x32_grid")], indirect=True)
def test_slice_tensor_args(mesh_device, input_shape, dim, start, end, step, layout):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    torch_start_tensor = torch.tensor(start)
    torch_end_tensor = torch.tensor(end)

    slices = tuple(slice(start[i], end[i]) for i in range(len(start)))

    # Slice the tensor using the slices for each dimension
    torch_output_tensor = torch_input[slices]

    ttnn_start_tensor = ttnn.from_torch(
        torch_start_tensor, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ttnn_end_tensor = ttnn.from_torch(
        torch_end_tensor, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    ttnn_tensor = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    print("before running slice")
    # Calculate num_devices from the slice pattern (input_shape[dim] / output_shape[dim])
    num_devices_calc = input_shape[dim] // (end[dim] - start[dim])
    ttnn_output = ttnn.slice(
        ttnn_tensor, ttnn_start_tensor, ttnn_end_tensor, slice_dim=dim, num_devices=num_devices_calc
    )
    print("after running slice")

    ttnn_output_tensor = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    print("shape of ttnn output:", ttnn_output_tensor.shape)
    print("shape of torch output:", torch_output_tensor.shape)

    num_devices = 32
    batch_size = input_shape[0]  # First dimension of input

    ttnn_reshaped = ttnn_output_tensor.view(num_devices, batch_size, *ttnn_output_tensor.shape[1:])

    ttnn_device_output = ttnn_reshaped[0]  # Shape: [3, 16, 32, 2880]

    print("shape of ttnn device output:", ttnn_device_output.shape)
    assert_with_pcc(torch_output_tensor, ttnn_device_output, 0.999)


@pytest.mark.parametrize(
    "input_shape, dim, start, end, step, layout",
    (
        ([1, 1, 128, 2880], 2, [0, 0, 0, 0], [1, 1, 32, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 2880], 2, [0, 0, 32, 0], [1, 1, 64, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 2880], 2, [0, 0, 64, 0], [1, 1, 96, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 2880], 2, [0, 0, 96, 0], [1, 1, 128, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 2880], 2, [0, 0, 0, 0], [1, 1, 64, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 2880], 2, [0, 0, 64, 0], [1, 1, 128, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 2880], 2, [0, 0, 128, 0], [1, 1, 192, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 2880], 2, [0, 0, 192, 0], [1, 1, 256, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 512, 2880], 2, [0, 0, 0, 0], [1, 1, 128, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 512, 2880], 2, [0, 0, 128, 0], [1, 1, 256, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 512, 2880], 2, [0, 0, 256, 0], [1, 1, 384, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 512, 2880], 2, [0, 0, 384, 0], [1, 1, 512, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 1024, 2880], 2, [0, 0, 0, 0], [1, 1, 256, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 1024, 2880], 2, [0, 0, 256, 0], [1, 1, 512, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 1024, 2880], 2, [0, 0, 512, 0], [1, 1, 768, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 1024, 2880], 2, [0, 0, 768, 0], [1, 1, 1024, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 2880], 2, [0, 0, 0, 0], [1, 1, 512, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 2880], 2, [0, 0, 512, 0], [1, 1, 1024, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 2880], 2, [0, 0, 1024, 0], [1, 1, 1536, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 2880], 2, [0, 0, 1536, 0], [1, 1, 2048, 2880], 1, ttnn.TILE_LAYOUT),
        # 4096
        ([1, 1, 4096, 2880], 2, [0, 0, 0, 0], [1, 1, 1024, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 4096, 2880], 2, [0, 0, 1024, 0], [1, 1, 2048, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 4096, 2880], 2, [0, 0, 2048, 0], [1, 1, 3072, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 4096, 2880], 2, [0, 0, 3072, 0], [1, 1, 4096, 2880], 1, ttnn.TILE_LAYOUT),
        # 8192
        ([1, 1, 8192, 2880], 2, [0, 0, 0, 0], [1, 1, 2048, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 8192, 2880], 2, [0, 0, 2048, 0], [1, 1, 4096, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 8192, 2880], 2, [0, 0, 4096, 0], [1, 1, 6144, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 8192, 2880], 2, [0, 0, 6144, 0], [1, 1, 8192, 2880], 1, ttnn.TILE_LAYOUT),
        # 16384
        ([1, 1, 16384, 2880], 2, [0, 0, 0, 0], [1, 1, 4096, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 16384, 2880], 2, [0, 0, 4096, 0], [1, 1, 8192, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 16384, 2880], 2, [0, 0, 8192, 0], [1, 1, 12288, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 16384, 2880], 2, [0, 0, 12288, 0], [1, 1, 16384, 2880], 1, ttnn.TILE_LAYOUT),
        # 32768
        ([1, 1, 32768, 2880], 2, [0, 0, 0, 0], [1, 1, 8192, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 32768, 2880], 2, [0, 0, 8192, 0], [1, 1, 16384, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 32768, 2880], 2, [0, 0, 16384, 0], [1, 1, 24576, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 32768, 2880], 2, [0, 0, 24576, 0], [1, 1, 32768, 2880], 1, ttnn.TILE_LAYOUT),
        # 65536
        ([1, 1, 65536, 2880], 2, [0, 0, 0, 0], [1, 1, 16384, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 65536, 2880], 2, [0, 0, 16384, 0], [1, 1, 32768, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 65536, 2880], 2, [0, 0, 32768, 0], [1, 1, 49152, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 65536, 2880], 2, [0, 0, 49152, 0], [1, 1, 65536, 2880], 1, ttnn.TILE_LAYOUT),
        # 131072 (128*1024)
        ([1, 1, 131072, 2880], 2, [0, 0, 0, 0], [1, 1, 32768, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 131072, 2880], 2, [0, 0, 32768, 0], [1, 1, 65536, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 131072, 2880], 2, [0, 0, 65536, 0], [1, 1, 98304, 2880], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 131072, 2880], 2, [0, 0, 98304, 0], [1, 1, 131072, 2880], 1, ttnn.TILE_LAYOUT),
    ),
)
def test_slice_tensor_args_before(input_shape, dim, start, end, step, layout, device):
    # Create tensor where each tile has a unique value for easy tile debugging
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    """
    # Calculate tile dimensions (32x32 tiles)
    tile_height, tile_width = 32, 32
    height_tiles = input_shape[2] // tile_height  # Number of tile rows
    width_tiles = input_shape[3] // tile_width    # Number of tile columns

    print(f"DEBUG: Input shape {input_shape} = {height_tiles}x{width_tiles} tiles")

    # Fill each tile with its unique tile ID
    tile_id = 0
    for tile_row in range(height_tiles):
        for tile_col in range(width_tiles):
            # Calculate pixel coordinates for this tile
            start_h = tile_row * tile_height
            end_h = start_h + tile_height
            start_w = tile_col * tile_width
            end_w = start_w + tile_width

            # Fill entire tile with the tile_id value
            torch_input[0, 0, start_h:end_h, start_w:end_w] = float(tile_id)
            tile_id += 1

    print(f"DEBUG: Created {tile_id} tiles total")
    print(f"DEBUG: Tile 0 (top-left): {torch_input[0, 0, 0, 0]} at [0,0]")
    print(f"DEBUG: Tile 1 (top, second): {torch_input[0, 0, 0, 32]} at [0,32]")
    print(f"DEBUG: Tile {width_tiles} (second row, first): {torch_input[0, 0, 32, 0]} at [32,0]")

    # For the slice [0,0,0,0] to [1,1,32,2880], we should get:
    # - First row of tiles (tiles 0 to width_tiles-1)
    print(f"DEBUG: Expected output should contain tiles 0 to {width_tiles-1}")
    """
    torch_start_tensor = torch.tensor(start)
    torch_end_tensor = torch.tensor(end)

    slices = tuple(slice(start[i], end[i]) for i in range(len(start)))

    # Slice the tensor using the slices for each dimension
    torch_output_tensor = torch_input[slices]

    ttnn_start_tensor = ttnn.from_torch(torch_start_tensor, device=device)
    ttnn_end_tensor = ttnn.from_torch(torch_end_tensor, device=device)

    ttnn_tensor = ttnn.from_torch(torch_input, layout=layout, dtype=ttnn.bfloat16, device=device)

    # Calculate num_devices from the slice pattern (input_shape[dim] / output_shape[dim])
    num_devices_calc = input_shape[dim] // (end[dim] - start[dim])
    ttnn_output = ttnn.slice(
        ttnn_tensor, ttnn_start_tensor, ttnn_end_tensor, slice_dim=dim, num_devices=num_devices_calc
    )

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    print("ttnn_output_tensor shape:", ttnn_output_tensor.shape)
    print("output tensor:", ttnn_output_tensor)
    print("expected tensor: ", torch_output_tensor)

    assert_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)
