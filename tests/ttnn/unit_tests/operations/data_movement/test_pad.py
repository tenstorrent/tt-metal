# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch

from models.common.utility_functions import skip_for_wormhole_b0, skip_for_blackhole
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn

torch.manual_seed(0)


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 2**15, shape, dtype=torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [230])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize(
    "padding,torch_padding",
    [
        (((0, 1), (3, 25), (32, 32)), (32, 32, 3, 25, 0, 1)),
        (((0, 1), (3, 25), (4, 6)), (4, 6, 3, 25, 0, 1)),
        (((0, 1), (3, 25), (4, 7)), (4, 7, 3, 25, 0, 1)),  # Odd padding widths (5 and 7)
    ],
)
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16])
def test_pad_rm(device, n, c, h, w, padding, torch_padding, value, dtype):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


def run_pad_with_program_cache(device, n, c, h, w, padding, torch_padding, value, dtype, layout):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 1))])
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_with_program_cache(device, n, c, h, w, padding, torch_padding, value, dtype, layout):
    if layout == ttnn.TILE_LAYOUT and dtype != ttnn.bfloat16:
        pytest.skip("tiled multicore pad only supported for bf16")
    for _ in range(2):
        run_pad_with_program_cache(device, n, c, h, w, padding, torch_padding, value, dtype, layout)
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
    assert device.num_program_cache_entries() == 1


def run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    n_unpadded = n
    c_unpadded = c + padding[0][1] + padding[0][0]
    h_unpadded = h + padding[1][1] + padding[1][0]

    # shard config
    num_cores_x = 8
    num_cores_y = 8
    if num_cores_y > device.core_grid.y:
        num_cores_y = device.core_grid.y
    shard_h = (n * c * h + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    # output shard config
    num_cores_x = 8
    num_cores_y = 8
    if num_cores_y > device.core_grid.y:
        num_cores_y = device.core_grid.y
    shard_h = (n_unpadded * c_unpadded * h_unpadded + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=output_mem_config)

    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert tt_output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, tt_output_tensor)


def to_torch_padding(padspec):
    def flatten_to_tuple(padding):
        return tuple(sum(padding, ()))

    def ttnn_pad_spec_to_padding(padspec):
        input_tensor_start = padspec["input_tensor_start"]
        pad_to_shape = padspec["pad_to_shape"]
        input_shape = padspec["input_shape"]

        padding = []
        for i in range(len(pad_to_shape)):
            this_dim_padding = (input_tensor_start[i], pad_to_shape[i] - input_shape[i] - input_tensor_start[i])
            padding.append(this_dim_padding)
        return padding

    torch_padding = flatten_to_tuple(reversed(ttnn_pad_spec_to_padding(padspec)))
    return torch_padding


@pytest.mark.parametrize(
    "input_shape, pad_to_shape, input_tensor_start, pad_value, input_sharded_memory_config_args",
    [
        [
            (1, 1, 1, 4),
            (1, 1, 1, 16),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=1), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # a reduced version of esmal's test case for UNet
            (1, 1, 4, 4),
            (1, 1, 4, 16),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=1), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding across large core grid, 3 sticks per core
            (1, 1, 3 * 64, 4),
            (1, 1, 3 * 64, 16),
            (0, 0, 0, 0),
            0.0,
            {"core_grid": ttnn.CoreGrid(x=8, y=8), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding across large core grid, 3 sticks per core, n300 version
            (1, 1, 3 * 8 * 7, 4),
            (1, 1, 3 * 8 * 7, 16),
            (0, 0, 0, 0),
            0.0,
            {"core_grid": ttnn.CoreGrid(x=8, y=7), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding only, reduced core grid
            (1, 1, 12, 8),
            (1, 1, 12, 64),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=2, y=6), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # height and width padding, small core grid
            (1, 1, 2, 4),
            (1, 1, 4, 8),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=2), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # borys's second test case
            (1, 2, 3, 4),
            (1, 2, 32, 32),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=6), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.float32])
def test_pad_rm_sharded_stickwise(
    device, input_shape, pad_to_shape, input_tensor_start, pad_value, input_sharded_memory_config_args, dtype
):
    core_grid_x_ok = device.core_grid.x >= input_sharded_memory_config_args["core_grid"].x
    core_grid_y_ok = device.core_grid.y >= input_sharded_memory_config_args["core_grid"].y
    device_core_grid_ok = core_grid_x_ok and core_grid_y_ok
    if not device_core_grid_ok:
        pytest.skip("core grid for this test is not compatible with the device")

    input_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)

    torch_input_tensor = torch.ones(input_shape, dtype=torch.float32)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Still relay on keep_l1_aligned = True to make it work with the current implementation
    ttnn_sharded_input_tensor = ttnn.interleaved_to_sharded(
        ttnn_input_tensor, input_shard_memory_config, keep_l1_aligned=True
    )
    padded_tensor = ttnn.pad(ttnn_sharded_input_tensor, pad_to_shape, input_tensor_start, pad_value)

    tt_output_tensor = ttnn.to_memory_config(padded_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    padspec = {
        "input_shape": input_shape,
        "pad_to_shape": pad_to_shape,
        "input_tensor_start": input_tensor_start,
    }
    torch_padded_tensor = torch.nn.functional.pad(
        torch_input_tensor, to_torch_padding(padspec), mode="constant", value=pad_value
    )

    assert torch_output_tensor.shape == torch_padded_tensor.shape
    assert_with_pcc(torch_padded_tensor, torch_output_tensor, 0.99)


@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("padding,torch_padding", [(((1, 1), (2, 32), (0, 0)), (0, 0, 2, 32, 1, 1))])
@pytest.mark.parametrize("value", [8])
@pytest.mark.parametrize("shard_orient", [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.bfloat16, ttnn.uint16])
def test_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype):
    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8x8 grid")
    for _ in range(2):
        run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        device.set_program_cache_misses_allowed(False)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 64),), (0, 64)), (((16, 16), (0, 32)), (0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    assert output_tensor.shape == torch_output_tensor.shape

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((32, 32),), (32, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_front_pad_not_supported(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: on device tile padding does not support front padding" in str(e.value)
    return


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 32), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_length(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: padding len can't be larger than input tensor rank" in str(e.value)
    return


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_back_to_back(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.pad(output_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape(
        (h + (padding[0][0] + padding[0][1]) * 2, w + (padding[1][0] + padding[1][1]) * 2)
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.skip(reason="ttnn.pad requires pad to start at 0")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding", [((0, 32), (0, 32)), ((1, 64), (0, 96)), ((0, 64), (0, 43)), ((32, 64), (64, 96))])
@pytest.mark.parametrize("value", [0])
def test_pad_for_tensor_in_tile_layout(device, h, w, padding, value):
    torch.manual_seed(0)
    torch_padding = (padding[1][0], padding[1][1], padding[0][0], padding[0][1])

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    if (
        padding[0][0] % ttnn.TILE_SIZE != 0
        or padding[0][1] % ttnn.TILE_SIZE != 0
        or padding[1][0] % ttnn.TILE_SIZE != 0
        or padding[1][1] % ttnn.TILE_SIZE != 0
    ):
        with pytest.raises(RuntimeError) as e:
            output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
        assert "must be a multiple of the tile size on height and width" in str(e.value)
        return
    else:
        output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


@skip_for_blackhole("Fails on Blackhole. Issue #20698")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
@pytest.mark.parametrize("use_multicore", [True, False], ids=["multicore", "singlecore"])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "shape, padded_shape",
    [
        [[1392, 1392, 3, 3], [1408, 1408, 3, 3]],
        [[32, 32, 3, 3], [64, 64, 3, 3]],
        [[3, 3, 1392, 1392], [3, 3, 1408, 1408]],
    ],
)
def test_pad_conv2d_sweep(device, dtype, use_multicore, shape, padded_shape, mem_config):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    in_torch = torch.randint(-5, 5, shape, dtype=torch_dtype).float()
    in_ttnn = ttnn.from_torch(in_torch, memory_config=mem_config, device=device, dtype=dtype)

    out_ttnn = ttnn.pad(in_ttnn, padded_shape, [0, 0, 0, 0], 0, use_multicore=use_multicore)
    out_torch = out_ttnn.cpu().to_torch().float()

    out_torch = out_torch[: shape[0], : shape[1], : shape[2], : shape[3]]
    assert torch.equal(in_torch, out_torch)


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16])
@pytest.mark.parametrize("shape", [[1, 1, 18, 13]])
@pytest.mark.parametrize("padshape", [[1, 1, TILE_HEIGHT, TILE_WIDTH]])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_pad_op(device, in_dtype, shape, padshape, use_multicore, layout, mem_config):
    torch_input = random_torch_tensor(in_dtype, shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=mem_config, dtype=in_dtype, layout=layout)
    output_tt = ttnn.pad(ttnn_input, padshape, [0, 0, 0, 0], value=0, use_multicore=use_multicore)
    output_tt = ttnn.to_torch(output_tt)
    assert output_tt.shape == torch.Size(padshape)

    shape_diff = list(map(lambda x, y: x - y, padshape, shape))
    output_torch = torch.nn.functional.pad(torch_input, [0, shape_diff[-1], 0, shape_diff[-2]], value=0)
    assert torch.equal(output_tt, output_torch)


def _unsqueeze(smaller, larger, fill):
    diff = len(larger) - len(smaller)
    return [fill] * diff + smaller


@pytest.mark.parametrize(
    "shape",
    [[2, 8], [1, 2, 3, 4], [5, 4, 3, 2, 1], [2, 128], [2, 60], [30, 128], [30, 60], [320, 320], [1, 1, 320, 320]],
)
@pytest.mark.parametrize(
    "padding", [[25, 1], [5, 4], [64], [32, 32], [1, 0, 0, 0], [1, 0, 0], [32, 32, 32, 64], [0, 64], [0, 0, 0, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16])
def test_pad_tile(shape, padding, dtype, device):
    if (shape, padding) in [([5, 4, 3, 2, 1], [1, 0, 0, 0]), ([5, 4, 3, 2, 1], [32, 32, 32, 64])]:
        pytest.xfail("Can't pad upper dims with rank>4")

    if len(shape) < len(padding):
        shape = _unsqueeze(shape, padding, 1)
    elif len(padding) < len(shape):
        padding = _unsqueeze(padding, shape, 0)

    input = torch.ones(prod(shape), dtype=torch.bfloat16).reshape(shape)
    input_tensor = ttnn.from_torch(input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    torch_padding = sum([[0, p] for p in reversed(padding)], [])
    torch_output = torch.nn.functional.pad(input, torch_padding, value=5)

    output = ttnn.pad(input_tensor, [(0, p) for p in padding], value=5)

    out_tt = ttnn.to_torch(output)

    assert torch.equal(out_tt, torch_output)
