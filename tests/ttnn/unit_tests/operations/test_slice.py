# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR, False)
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
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR, False)
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


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_slice_rm_sharded_with_program_cache(device, n, c, h, w, use_program_cache):
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
):
    if dtype == ttnn.float32:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.float)
    else:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_layout, device=device, memory_config=in_mem_config
    )

    tt_output_tensor = ttnn.slice(tt_input_tensor, output_tensor_start, output_tensor_end, memory_config=out_mem_config)

    a_pt = ttnn.to_torch(tt_output_tensor)

    # Pytorch reference
    a_ref = torch_input_tensor[
        output_tensor_start[0] : output_tensor_end[0],
        output_tensor_start[1] : output_tensor_end[1],
        output_tensor_start[2] : output_tensor_end[2],
        output_tensor_start[3] : output_tensor_end[3],
    ]

    return a_pt, a_ref, device.num_program_cache_entries()


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
    use_program_cache,
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
    )
    # CACHE HIT
    assert num_cache_entries == 4
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


# slice alternate elements in a given tensor
@pytest.mark.parametrize("dim", [4, 12, 20, 68])
def test_stride_slice_single_dim_skip_2(dim, device):
    torch.manual_seed(2005)
    torch_input = torch.rand(dim)
    torch_output = torch_input[::2]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
    ttnn_output = ttnn_input[::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("h", [18, 34])
@pytest.mark.parametrize("w", [18, 34])
@pytest.mark.parametrize("begins_h", [2])
@pytest.mark.parametrize("begins_w", [2])
@pytest.mark.parametrize("stride_h", [2])
@pytest.mark.parametrize("stride_w", [2])
def test_stride_slice_two_dim(h, w, begins_h, begins_w, stride_h, stride_w, device):
    torch.manual_seed(2005)
    torch_input = torch.rand(h, w)
    torch_output = torch_input[begins_h:h:stride_h, begins_w:w:stride_w]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
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
def test_stride_slice_three_dim(c, h, w, begins_c, begins_h, begins_w, stride_c, stride_h, stride_w, device):
    torch.manual_seed(2005)
    torch_input = torch.rand(c, h, w)
    torch_output = torch_input[begins_c:c:stride_c, begins_h:h:stride_h, begins_w:w:stride_w]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
    ttnn_output = ttnn_input[begins_c:c:stride_c, begins_h:h:stride_h, begins_w:w:stride_w]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize("dims", [[18, 18, 18, 18]])
@pytest.mark.parametrize("begins", [[2, 0, 0, 2]])
@pytest.mark.parametrize("ends", [[18, 16, 16, 18]])
@pytest.mark.parametrize("strides", [[2, 2, 2, 2]])
def test_stride_slice_four_dim(dims, begins, ends, strides, device):
    torch.manual_seed(2005)
    torch_input = torch.rand(dims)
    slices = []
    for i in range(len(dims)):
        slices.append(slice(begins[i], ends[i], strides[i]))

    torch_output = torch_input[slices[0], slices[1], slices[2], slices[3]]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
    ttnn_output = ttnn_input[slices[0], slices[1], slices[2], slices[3]]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


# these tests are copy and paste from the yolo customers #8920
def test_slice_usecase1(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, ::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase2(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, 1::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, 1::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase3(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., 1::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., 1::2, ::2]
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase4(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

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
        if input_ends[-1] - input_start[-1] == 1:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.float32, layout=layout)

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
        if input_ends[-1] - input_start[-1] == 1:
            pytest.skip("Cannot slice the last dimension to 1 in row major layout")
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.float32, layout=layout)

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

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1), output_tensor=ttnn_output)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

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

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1), output_tensor=ttnn_output)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

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
        ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.float32, layout=layout)

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
