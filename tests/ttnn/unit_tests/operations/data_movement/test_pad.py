# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch
import math

from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0, skip_for_blackhole
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose
from tests.ttnn.utils_for_testing import assert_equal as assert_tensors_equal
import ttnn

torch.manual_seed(0)


def dm_codegen_supported_on_arch(device):
    # Mirrors codegen_supported_on_arch in data_movement/common: on other arches the ops route to native and the
    # forced codegen entries raise.
    return ttnn.device.is_wormhole_b0(device) or ttnn.device.is_blackhole(device)


def skip_unless_dm_codegen_arch(device):
    if not dm_codegen_supported_on_arch(device):
        pytest.skip(f"data-movement codegen is not supported on {device.arch()}")


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 2**16, shape, dtype=torch.uint16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**32, shape, dtype=torch.uint32)
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


@pytest.mark.parametrize(
    "shape,padding,torch_padding,use_multicore",
    [
        ((8, 1, 1, 1), ((0, 0), (0, 0), (0, 0), (0, 191)), (0, 191, 0, 0, 0, 0, 0, 0), True),
        ((1, 1, 1, 2), ((0, 0), (0, 0), (0, 0), (0, 254)), (0, 254, 0, 0, 0, 0, 0, 0), True),
        ((4, 1, 1, 4), ((0, 0), (0, 0), (0, 0), (0, 60)), (0, 60, 0, 0, 0, 0, 0, 0), True),
        ((1, 1, 1, 1985), ((0, 0), (0, 0), (0, 0), (49290, 0)), (49290, 0, 0, 0, 0, 0, 0, 0), True),
        pytest.param(
            (1, 1, 1, 1985),
            ((0, 0), (0, 0), (0, 0), (49290, 0)),
            (49290, 0, 0, 0, 0, 0, 0, 0),
            False,
            marks=pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="single-core RM pad writes the padding after the data instead of before it: "
                "https://github.com/tenstorrent/tt-metal/issues/56323",
            ),
        ),
    ],
)
@pytest.mark.parametrize("value", [0])
def test_pad_rm_small_to_large_width(device, shape, padding, torch_padding, use_multicore, value):
    """Regression test for issue #39875: padding from very small width to large width
    caused CB allocation to exceed L1 size due to using input width for stick batching,
    and for wide padded rows whose fixed 16-row CB depth exceeded L1 on both the
    multi-core and single-core row-major factories."""
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape).bfloat16().float()
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


def run_pad_rm_small_to_large_width_with_program_cache(device, shape, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape).bfloat16().float()
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    with device.cache_entries_counter.measure():
        output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape,padding,torch_padding",
    [
        ((8, 1, 1, 1), ((0, 0), (0, 0), (0, 0), (0, 191)), (0, 191, 0, 0, 0, 0, 0, 0)),
    ],
)
@pytest.mark.parametrize("value", [0])
def test_pad_rm_small_to_large_width_with_program_cache(device, shape, padding, torch_padding, value):
    """Regression test for issue #39875 with program cache: ensure override_runtime_arguments
    also uses output stick size for CB batching."""
    for _ in range(2):
        run_pad_rm_small_to_large_width_with_program_cache(device, shape, padding, torch_padding, value)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.cache_entries_counter.total == 1


def run_pad_with_program_cache(device, n, c, h, w, padding, torch_padding, value, dtype, layout):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    with device.cache_entries_counter.measure():
        output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    if dtype == ttnn.bfloat8_b:
        assert_allclose(torch_output_tensor, output_tensor, rtol=0.05, atol=0.025)
    else:
        assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 1))])
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_with_program_cache(device, n, c, h, w, padding, torch_padding, value, dtype, layout):
    if layout == ttnn.TILE_LAYOUT and dtype != ttnn.bfloat16 and dtype != ttnn.bfloat8_b:
        pytest.skip("tiled multicore pad only supported for bf16 and bfp8")
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat8_b:
        pytest.skip("row major bfloat8 is not supported")
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
    # This bfloat8_b tile pad routes to codegen, which goes through bfloat16 (typecast, pad, typecast) only for a
    # nonzero value; a zero value pads the block-float tiles directly. Native always takes the bfloat16 round trip.
    bf16_round_trip = value != 0 or not dm_codegen_supported_on_arch(device)
    expected_cache_entries = 3 if dtype == ttnn.bfloat8_b and bf16_round_trip else 1
    assert device.cache_entries_counter.total == expected_cache_entries


def test_pad_program_cache_hit_updates_pad_value_buffer(device):
    """Regression for #44565: pad must update its pad-value buffer on cache hits.

    A minimal pad-only reproducer did not trigger the stale pad-value buffer bug; it
    only reproduced through prepare_conv3d_weights, which exercises pad after
    from_torch, to_device, and permute.
    """
    torch.manual_seed(42)
    weights = torch.randn(32, 12, 3, 3, 3, dtype=torch.float32)

    outputs = []
    for _ in range(5):
        tt_weight = ttnn.from_torch(weights, dtype=ttnn.DataType.BFLOAT16, pad_value=0)
        prepared_weight = ttnn.experimental.prepare_conv3d_weights(
            weight_tensor=tt_weight, groups=1, C_in_block=32, alignment=32, device=device
        )
        outputs.append(ttnn.to_torch(prepared_weight).to(torch.float32))

    max_diff = max((output - outputs[0]).abs().max().item() for output in outputs[1:])
    assert max_diff < 1e-3, f"pad output is non-deterministic across cache hits: max diff = {max_diff}"


def run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype, buffer_type):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    n_unpadded = n
    c_unpadded = c + padding[0][1] + padding[0][0]
    h_unpadded = h + padding[1][1] + padding[1][0]

    # core grid: compute cores for L1, DRAM banks for DRAM
    if buffer_type == ttnn.types.BufferType.DRAM:
        dram_grid = device.dram_grid_size()
        num_cores_x = dram_grid.x
        num_cores_y = dram_grid.y
    else:
        compute_grid = device.compute_with_storage_grid_size()
        num_cores_x = min(8, compute_grid.x)
        num_cores_y = min(8, compute_grid.y)

    # shard config
    num_cores = num_cores_x * num_cores_y
    shard_h = ttnn.core.divup(n * c * h, num_cores)
    grid_coord = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, shard_spec)

    # output shard config
    shard_h = ttnn.core.divup(n_unpadded * c_unpadded * h_unpadded, num_cores)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient)
    output_mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, shard_spec)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=output_mem_config)
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
@pytest.mark.parametrize("buffer_type", [ttnn.types.BufferType.L1, ttnn.types.BufferType.DRAM])
def test_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype, buffer_type):
    for _ in range(2):
        run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, dtype, buffer_type)
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


@pytest.mark.parametrize("shard_orient", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_pad_rm_sharded_height_only_non_contiguous_grid(device, shard_orient, dtype):
    """Height-only RM pad on a shard grid with a hole in it.

    Regression test for the Qwen3-32B Blackhole Galaxy decode path, where the RoPE cos/sin slices are
    height-sharded on ``{[1-0 - 3-7], [5-0 - 6-7]}`` (columns 0 and 4 excluded) and then padded to a
    tile-aligned height. Deriving the per-core runtime args from the grid's bounding box emitted args
    for gap core (4, 0), where no kernel runs, and mapped source shards to the wrong cores.
    """
    torch.manual_seed(0)
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < 7 or compute_grid.y < 8:
        pytest.skip(f"needs a 7x8 compute grid, device has {compute_grid.x}x{compute_grid.y}")

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7)),
        }
    )
    num_cores = shard_grid.num_cores()
    assert num_cores == 40

    # The decode-path tensor: [1, 8, 8, 128] padded to [1, 8, 32, 128], shard [32, 128] on all 40 cores,
    # so only the first 2 (input) / 8 (output) cores in grid order hold data. Core index 3 in grid order
    # is (5, 0); a bounding-box walk puts it at (4, 0), the hole.
    n, c, h, w = 1, 8, 8, 128
    shard_h = 32
    padding = ((0, 0), (0, shard_h - h), (0, 0))
    torch_padding = (0, 0, 0, shard_h - h, 0, 0)
    value = 0

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=sharded_mem_config)
    tt_output_tensor = ttnn.to_torch(ttnn.from_device(tt_output_tensor))

    assert tt_output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, tt_output_tensor)


@pytest.mark.parametrize("shard_orient", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_pad_rm_sharded_height_only_non_contiguous_grid_straddles_range_boundary(device, shard_orient, dtype):
    """Height-only RM pad where one output shard gathers rows from input shards on both sides of the hole.

    Same 40-core grid as the test above, but the input fills all 40 shards ([1, 1, 1280, 128], shard
    [32, 128]) and the output shard is taller ([40, 128], H padded to 1600). In grid order the input
    shards on either side of the hole are shard 23 on core (3, 7) and shard 24 on core (5, 0); output
    shard 19 (rows 760..799) needs rows 24..31 of shard 23 followed by rows 0..31 of shard 24, so the
    result is only correct if the gather plan emits those source cores in grid order rather than in
    sorted physical-coordinate order.
    """
    torch.manual_seed(0)
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < 7 or compute_grid.y < 8:
        pytest.skip(f"needs a 7x8 compute grid, device has {compute_grid.x}x{compute_grid.y}")

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7)),
        }
    )
    assert shard_grid.num_cores() == 40

    n, c, h, w = 1, 1, 1280, 128
    in_shard_h, out_shard_h = 32, 40
    h_padded = out_shard_h * 40
    assert h == in_shard_h * 40
    padding = ((0, 0), (0, h_padded - h), (0, 0))
    torch_padding = (0, 0, 0, h_padded - h, 0, 0)
    value = 0

    torch_input_tensor = random_torch_tensor(dtype, (n, c, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    in_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.types.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (in_shard_h, w), shard_orient),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.types.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (out_shard_h, w), shard_orient),
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mem_config,
    )
    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=out_mem_config)
    tt_output_tensor = ttnn.to_torch(ttnn.from_device(tt_output_tensor))

    assert tt_output_tensor.shape == torch_output_tensor.shape
    # Report the first mismatching output row so a wrong gather order is visible in the CI log.
    mismatched_rows = torch.nonzero(~torch.all(torch_output_tensor[0, 0] == tt_output_tensor[0, 0], dim=-1)).flatten()
    assert (
        mismatched_rows.numel() == 0
    ), f"rows differ: {mismatched_rows[:16].tolist()} (first of {mismatched_rows.numel()})"


def test_pad_rm_sharded_height_only_override_addr_change(device):
    """Cache-hit hook (override_runtime_arguments) for the CB-bound height-sharded RM factory: buffer
    addresses are excluded from the pad hash, so a second identical pad at DIFFERENT input AND output
    addresses must hit the cache and still re-point both sharded CB bases instead of reading/writing
    the first dispatch's L1 shards."""
    torch.manual_seed(48928)

    n, c, h, w = 1, 3, 224, 256
    padded_shape = [n, c, 256, w]
    input_tensor_start = [0, 0, 0, 0]
    pad_value = 8.0

    num_cores_x, num_cores_y = 8, 7
    ncores = num_cores_x * num_cores_y
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    in_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (ttnn.core.divup(n * c * h, ncores), w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_grid,
            (ttnn.core.divup(n * c * padded_shape[2], ncores), w),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def _make():
        t = torch.rand((n, c, h, w), dtype=torch.bfloat16)
        return t, ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
        )

    def _run(tt_in):
        return ttnn.pad(tt_in, padded_shape, input_tensor_start, pad_value, memory_config=out_cfg)

    torch_a, tt_a = _make()
    tt_out_a = _run(tt_a)
    assert_equal(pad_reference(torch_a, padded_shape, input_tensor_start, pad_value), ttnn.to_torch(tt_out_a))
    entries_after_first = device.num_program_cache_entries()

    # Keep A alive so B's input and output both land at different L1 addresses.
    torch_b, tt_b = _make()
    assert tt_b.buffer_address() != tt_a.buffer_address(), "second input must land at a different address"
    tt_out_b = _run(tt_b)
    assert tt_out_b.buffer_address() != tt_out_a.buffer_address(), "second output must land at a different address"
    assert device.num_program_cache_entries() == entries_after_first, "identical pad must be a cache hit"
    assert_equal(pad_reference(torch_b, padded_shape, input_tensor_start, pad_value), ttnn.to_torch(tt_out_b))


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
def test_pad_padding_validation_front_pad_not_supported(device, h, w, padding, torch_padding, value, expect_error):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "ttnn.pad: on device tile padding does not support front padding"):
        ttnn.pad(input_tensor, padding=padding, value=value)
    return


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 32), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_length(device, h, w, padding, torch_padding, value, expect_error):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "ttnn.pad: padding len can't be larger than input tensor rank"):
        ttnn.pad(input_tensor, padding=padding, value=value)
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
def test_pad_for_tensor_in_tile_layout(device, h, w, padding, value, expect_error):
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
        with expect_error(RuntimeError, "must be a multiple of the tile size on height and width"):
            ttnn.pad(input_tensor, padding=padding, value=value)
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


@pytest.mark.parametrize(
    "in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.bfloat8_b]
)
@pytest.mark.parametrize("shape", [[1, 1, 18, 13]])
@pytest.mark.parametrize("padshape", [[1, 1, TILE_HEIGHT, TILE_WIDTH]])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_pad_op(device, in_dtype, shape, padshape, use_multicore, layout, mem_config):
    if layout == ttnn.ROW_MAJOR_LAYOUT and in_dtype == ttnn.bfloat8_b:
        pytest.skip("row major bfloat8 is not supported")
    torch_input = random_torch_tensor(in_dtype, shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=mem_config, dtype=in_dtype, layout=layout)
    output_tt = ttnn.pad(ttnn_input, padshape, [0, 0, 0, 0], value=0, use_multicore=use_multicore)
    output_tt = ttnn.to_torch(output_tt)
    assert output_tt.shape == torch.Size(padshape)

    shape_diff = list(map(lambda x, y: x - y, padshape, shape))
    output_torch = torch.nn.functional.pad(torch_input, [0, shape_diff[-1], 0, shape_diff[-2]], value=0)
    if in_dtype == ttnn.bfloat8_b:
        assert_allclose(output_torch, output_tt, rtol=0.05, atol=0.025)
    else:
        assert torch.equal(output_tt, output_torch)


# test_pad_op above parametrizes TILE x use_multicore=False, but its shapes are already tile-aligned,
# so invoke_tile() returns through the view() fast path and never reaches prim::pad.  Crossing a tile
# boundary is what actually selects PadTileCoreProgramFactory, the single-core tiled factory.
@pytest.mark.parametrize(
    "in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.bfloat8_b]
)
@pytest.mark.parametrize(
    "shape, padshape",
    [
        ([1, 1, 32, 32], [1, 1, 64, 64]),  # grow both tile dims
        ([1, 1, 32, 64], [2, 3, 64, 96]),  # grow the batch/channel dims too
        ([2, 1, 64, 32], [2, 2, 96, 64]),
    ],
)
@pytest.mark.parametrize("pad_value", [0.0, 7.0])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_pad_tile_single_core(device, in_dtype, shape, padshape, pad_value, mem_config):
    if in_dtype == ttnn.bfloat8_b and pad_value != 0.0:
        pytest.skip("bfloat8_b compares approximately; the exact-fill check below does not apply")
    if in_dtype == ttnn.float32 and pad_value != 0.0:
        pytest.xfail("#54223: FLOAT32 pad value is packed as two bfloat16s in the single-core TILE factory")

    torch_input = random_torch_tensor(in_dtype, shape)
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, memory_config=mem_config, dtype=in_dtype, layout=ttnn.TILE_LAYOUT
    )

    # Repeat so the second and third dispatches land on a program-cache hit, which is where a
    # stale tensor binding would show up.
    for _ in range(3):
        output_tt = ttnn.to_torch(ttnn.pad(ttnn_input, padshape, [0, 0, 0, 0], value=pad_value, use_multicore=False))
        assert output_tt.shape == torch.Size(padshape)

        output_torch = torch.full(padshape, pad_value).to(output_tt.dtype)
        output_torch[: shape[0], : shape[1], : shape[2], : shape[3]] = torch_input.to(output_tt.dtype)
        if in_dtype == ttnn.bfloat8_b:
            assert_allclose(output_torch, output_tt, rtol=0.05, atol=0.025)
        else:
            assert torch.equal(output_tt, output_torch)


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_multicore", [True])
def test_pad_op_row_major_unevernly_sharded(device, in_dtype, use_multicore):
    shape = [1, 1, 18, 50]
    padshape = [1, 1, TILE_HEIGHT, 2 * TILE_WIDTH]
    shard_shape = [1, 1, 18, 40]

    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = random_torch_tensor(in_dtype, shape)
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, memory_config=input_mem_config, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_tt = ttnn.pad(
        ttnn_input, padshape, [0, 0, 0, 0], value=0, use_multicore=use_multicore, memory_config=output_mem_config
    )
    output_tt = ttnn.to_torch(output_tt)
    assert output_tt.shape == torch.Size(padshape), f"Shape mismatch: expected {padshape}, got {list(output_tt.shape)}"

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
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint16, ttnn.bfloat8_b])
def test_pad_tile(shape, padding, dtype, device):
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


def pad_reference(input_torch, padded_shape, input_tensor_start, pad_value):
    """PyTorch reference for ttnn.pad using shape-based API."""
    rank = len(padded_shape)
    torch_padding = []
    for i in reversed(range(rank)):
        front = input_tensor_start[i]
        end = padded_shape[i] - input_torch.shape[i] - input_tensor_start[i]
        torch_padding.extend([front, end])
    return torch.nn.functional.pad(input_torch, torch_padding, mode="constant", value=pad_value)


def skip_if_tile_layout_incompatible(layout, input_shard_shape=None, output_shard_shape=None):
    """Skip test if tile layout is used but shapes aren't tile-aligned."""
    TILE_SIZE = 32
    if layout != ttnn.TILE_LAYOUT:
        return
    for shard_shape in (input_shard_shape, output_shard_shape):
        if shard_shape is None:
            continue
        if shard_shape[-1] % TILE_SIZE != 0 or shard_shape[-2] % TILE_SIZE != 0:
            pytest.skip(f"TILE_LAYOUT requires shard shape last 2 dims to be multiples of TILE_SIZE, got {shard_shape}")


def assert_equal(expected, actual):
    assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
    assert torch.equal(expected, actual), "Data mismatch"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape, output_shard_shape,shard_core_grid",
    [
        # Height-only end padding, same shard shape
        (
            [1, 1, 16, 64],
            [1, 1, 32, 64],
            [1, 1, 8, 64],
            None,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Width-only end padding, same shard shape
        (
            [1, 1, 16, 32],
            [1, 1, 16, 64],
            [1, 1, 8, 32],
            [1, 1, 8, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # N-dim padding
        (
            [2, 1, 16, 64],
            [4, 1, 16, 64],
            [1, 1, 16, 64],
            None,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Uneven input sharding (dim 0: 3 % 2 != 0)
        (
            [3, 1, 16, 64],
            [3, 1, 32, 64],
            [2, 1, 8, 64],
            [2, 1, 16, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Different input/output shard specs
        (
            [1, 1, 16, 64],
            [1, 1, 32, 64],
            [1, 1, 8, 64],
            [1, 1, 16, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Combined H and W padding, different output shard
        (
            [1, 1, 16, 32],
            [1, 1, 32, 64],
            [1, 1, 8, 32],
            [1, 1, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Uneven ND sharding, odd padding
        (
            [5, 4, 160, 160],
            [6, 7, 191, 187],
            [2, 3, 64, 96],
            [2, 3, 60, 40],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))}),
        ),
        # Uneven ND sharding, tile divisible
        (
            [5, 4, 160, 160],
            [6, 7, 192, 192],
            [2, 3, 64, 96],
            [2, 3, 128, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))}),
        ),
    ],
)
@pytest.mark.parametrize("input_is_nd_sharded", [True, False])
@pytest.mark.parametrize("output_is_nd_sharded", [True, False])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded(
    device,
    dtype,
    tensor_shape,
    padded_shape,
    input_shard_shape,
    output_shard_shape,
    shard_core_grid,
    input_is_nd_sharded,
    output_is_nd_sharded,
    layout,
):
    effective_output_shard_shape = output_shard_shape if output_shard_shape is not None else input_shard_shape
    skip_if_tile_layout_incompatible(layout, input_shard_shape, effective_output_shard_shape)
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=effective_output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    if input_is_nd_sharded:
        input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    else:
        input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if output_is_nd_sharded:
        output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 10, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 10)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape",
    [
        # ND-sharded input, height-only padding, interleaved output
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 8, 64]),
        # ND-sharded input, combined padding, interleaved output
        ([1, 1, 16, 32], [1, 1, 32, 64], [1, 1, 8, 32]),
        # Uneven input sharding
        ([3, 1, 16, 64], [3, 1, 32, 64], [2, 1, 8, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded_to_interleaved(
    device, dtype, tensor_shape, padded_shape, input_shard_shape, shard_core_grid, layout
):
    skip_if_tile_layout_incompatible(layout, input_shard_shape=input_shard_shape)
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, output_shard_shape",
    [
        # Interleaved input, height-only padding, ND-sharded output
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 16, 64]),
        # Interleaved input, combined padding, ND-sharded output
        ([1, 1, 16, 32], [1, 1, 32, 64], [1, 1, 16, 32]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_interleaved_to_nd_sharded(
    device, dtype, tensor_shape, padded_shape, output_shard_shape, shard_core_grid, layout
):
    skip_if_tile_layout_incompatible(layout, output_shard_shape=output_shard_shape)
    torch.manual_seed(42)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 8, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 16, 64]),
        # Uneven ND sharding: dim 0 has 3 % 2 != 0
        ([3, 1, 16, 64], [3, 1, 32, 64], [2, 1, 8, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
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
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded_to_legacy_sharded(
    device, dtype, tensor_shape, padded_shape, input_shard_shape, shard_core_grid, output_memory_layout, layout
):
    """pad: ND-sharded input -> legacy-sharded output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(padded_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= padded_shape[i]
    tensor_width = padded_shape[-1]

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

    if any(d == 0 for d in layout_info["shard_shape"]):
        pytest.skip("Shard shape has a zero dimension")
    if layout_info["shard_shape"][0] * layout_info["shard_shape"][1] == 0:
        pytest.skip("Invalid shard shape")
    skip_if_tile_layout_incompatible(
        layout,
        input_shard_shape=input_shard_shape,
        output_shard_shape=layout_info["shard_shape"],
    )

    output_shard_spec = ttnn.ShardSpec(
        layout_info["shard_grid"], layout_info["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, output_shard_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 16, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 16, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_legacy_sharded_to_nd_sharded(
    device, dtype, tensor_shape, padded_shape, output_shard_shape, shard_core_grid, input_memory_layout, layout
):
    """pad: legacy-sharded input -> ND-sharded output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    input_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(layout, input_shard_shape=input_shard_shape, output_shard_shape=output_shard_shape)
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64]),
        ([3, 1, 128, 128], [3, 1, 150, 150]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_legacy_sharded_to_interleaved(device, dtype, tensor_shape, padded_shape, shard_core_grid, layout):
    """pad: legacy HEIGHT_SHARDED input -> interleaved output via default factory."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    input_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(layout, input_shard_shape=input_shard_shape)
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape",
    [
        ([3, 1, 128, 128], [3, 1, 192, 192]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_interleaved_to_legacy_sharded(device, dtype, tensor_shape, padded_shape, shard_core_grid, layout):
    """pad: interleaved input -> legacy HEIGHT_SHARDED output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(padded_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= padded_shape[i]
    tensor_width = padded_shape[-1]

    output_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(layout, output_shard_shape=output_shard_shape)
    output_shard_spec = ttnn.ShardSpec(shard_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_tensor_start, input_shard_shape, shard_core_grid",
    [
        (
            [1, 1, 16, 32],
            [1, 1, 32, 64],
            [0, 0, 8, 16],
            [1, 1, 8, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [5, 3, 17, 24],
            [6, 5, 32, 64],
            [1, 2, 7, 20],
            [3, 2, 9, 16],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
    ids=["even_sharding", "uneven_sharding"],
)
def test_pad_nd_sharded_front_padding_row_major(
    device, dtype, tensor_shape, padded_shape, input_tensor_start, input_shard_shape, shard_core_grid
):
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 7, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 7)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_tensor_start, input_shard_shape, output_shard_shape, shard_core_grid",
    [
        (
            [1, 1, 16, 32],
            [1, 1, 32, 64],
            [0, 0, 8, 16],
            [1, 1, 8, 32],
            [1, 1, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [5, 3, 17, 24],
            [6, 5, 32, 64],
            [1, 2, 7, 20],
            [3, 2, 9, 16],
            [4, 4, 17, 24],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
    ids=["different_shards_even", "different_shards_uneven"],
)
def test_pad_nd_sharded_to_nd_sharded_front_padding_row_major(
    device,
    dtype,
    tensor_shape,
    padded_shape,
    input_tensor_start,
    input_shard_shape,
    output_shard_shape,
    shard_core_grid,
):
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 7, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 7)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_pad_nd_sharded_front_padding_tile_layout_not_supported(device, dtype, expect_error):
    torch.manual_seed(42)

    tensor_shape = [1, 1, 32, 32]
    padded_shape = [1, 1, 64, 64]
    input_tensor_start = [0, 0, 32, 32]
    input_shard_shape = [1, 1, 32, 32]
    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    with expect_error(RuntimeError, "ttnn.pad: on device tile padding does not support front padding"):
        ttnn.pad(
            input_ttnn_tensor,
            padded_shape,
            input_tensor_start,
            0,
            use_multicore=True,
            memory_config=output_memory_config,
        )


def _ttnn_padding_to_torch(padding):
    """Convert ttnn per-dim (before, after) padding to torch.nn.functional.pad format."""
    # torch.nn.functional.pad expects (last_dim_before, last_dim_after, ..., first_dim_before, first_dim_after).
    return tuple(v for i in reversed(range(len(padding))) for v in padding[i])


def _assert_pad_fill_regions(output_tensor, input_shape, padding, value):
    """Exact check that every padded slice equals the fill value."""
    for dim, (before, after) in enumerate(padding):
        if before:
            slices = [slice(None)] * len(input_shape)
            slices[dim] = slice(0, before)
            assert torch.all(output_tensor[tuple(slices)] == value)
        if after:
            slices = [slice(None)] * len(input_shape)
            slices[dim] = slice(before + input_shape[dim], before + input_shape[dim] + after)
            assert torch.all(output_tensor[tuple(slices)] == value)


def _assert_pad_body_matches(output_tensor, input_tensor, padding, *, atol=0.0):
    """Check the unpadded region matches the input."""
    slices = []
    for dim, (before, _after) in enumerate(padding):
        slices.append(slice(before, before + input_tensor.shape[dim]))
    body_out = output_tensor[tuple(slices)]
    if atol == 0.0:
        assert torch.equal(body_out, input_tensor)
    else:
        assert_allclose(body_out.float(), input_tensor.float(), rtol=0, atol=atol)


@pytest.mark.parametrize(
    "shape,padding",
    [
        ((1, 4, 4, 4, 128), ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0))),
        ((2, 4, 4, 4, 128), ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((1, 4, 4, 4, 128), ((0, 0), (0, 2), (0, 0), (0, 0), (0, 0))),
        ((1, 4, 4, 4, 128), ((0, 0), (0, 0), (1, 2), (0, 0), (0, 0))),
        ((1, 4, 4, 4, 128), ((0, 0), (0, 0), (0, 0), (0, 2), (0, 0))),
        ((1, 4, 4, 4, 128), ((0, 0), (0, 0), (0, 0), (0, 0), (0, 3))),
        ((1, 2, 3, 4, 4, 128), ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((1, 2, 3, 4, 4, 128), ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0), (0, 0))),
        ((1, 2, 3, 4, 4, 128), ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0), (0, 0))),
        ((1, 2, 3, 4, 4, 128), ((0, 0), (0, 0), (0, 0), (0, 0), (0, 2), (0, 0))),
        ((1, 2, 3, 4, 4, 128), ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1))),
        ((2, 4, 4, 4, 128), ((1, 1), (0, 2), (0, 0), (0, 0), (0, 0))),
        ((2, 3, 4, 5, 8), ((1, 1), (0, 1), (2, 0), (0, 2), (1, 1))),
        ((2, 2, 3, 4, 4, 16), ((0, 1), (0, 0), (1, 0), (0, 0), (0, 0), (0, 2))),
        ((2, 2, 2, 2, 2, 4, 16), ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((2, 2, 2, 2, 2, 2, 2, 8), ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((2, 2, 2, 2, 2, 2, 2, 8), ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0), (0, 0))),
        ((3, 3, 5, 7, 13), ((0, 0), (1, 2), (0, 0), (0, 0), (0, 0))),
    ],
    ids=[
        "rank5_dim1",
        "rank5_dim0",
        "rank5_dim1_after_only",
        "rank5_dim2",
        "rank5_dim3",
        "rank5_dim4",
        "rank6_dim1",
        "rank6_dim2",
        "rank6_dim3",
        "rank6_dim4",
        "rank6_dim5",
        "rank5_dim0_and_dim1",
        "rank5_all_dims",
        "rank6_dim0_dim2_dim5",
        "rank7_dim2",
        "rank8_dim1",
        "rank8_dim5",
        "rank5_unaligned",
    ],
)
@pytest.mark.parametrize("value", [0.0, 3.5])
def test_pad_rank_gt4_all_dimensions(device, shape, padding, value):
    torch.manual_seed(0)

    torch_padding = _ttnn_padding_to_torch(padding)
    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    output_tensor = ttnn.pad(input_tensor, padding, value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape,padding",
    [
        # Leading dims (reshape path). Tile layout has no front-padding support on device,
        # so these only exercise back padding.
        ((1, 4, 4, 32, 128), ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))),
        ((2, 4, 4, 32, 128), ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((1, 2, 3, 4, 32, 128), ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0), (0, 0))),
        # dim rank-3 (squeeze-to-4D path) and a combination across both paths.
        ((1, 4, 4, 32, 128), ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0))),
        ((2, 4, 4, 32, 128), ((0, 1), (0, 2), (0, 1), (0, 0), (0, 0))),
        # Height/width padding must stay tile aligned.
        ((1, 4, 4, 32, 128), ((0, 0), (0, 1), (0, 0), (0, 32), (0, 64))),
    ],
    ids=[
        "rank5_dim1",
        "rank5_dim0",
        "rank6_dim2",
        "rank5_dim2",
        "rank5_dim0_dim1_dim2",
        "rank5_dim1_with_hw",
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_pad_rank_gt4_tile_layout(device, shape, padding, dtype):
    torch.manual_seed(0)
    value = 0.0

    torch_padding = _ttnn_padding_to_torch(padding)
    if dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-100, 100, shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding, value))

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.int32, ttnn.uint32])
def test_pad_rank_gt4_row_major_integer_dtypes(device, dtype):
    """Leading-dim padding of a rank > 4 tensor must not depend on the element type."""
    torch.manual_seed(0)
    shape = (2, 4, 4, 4, 128)
    padding = ((1, 0), (1, 1), (0, 0), (0, 0), (0, 2))

    torch_input_tensor = torch.randint(0, 100, shape, dtype=torch.int32)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, _ttnn_padding_to_torch(padding), value=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding, 0.0)).to(torch.int32)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "padding", [((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0))]
)
def test_pad_rank_gt4_tile_bfloat8_b(device, padding):
    """bfloat8_b is padded via a bfloat16 round trip; make sure the rank > 4 path preserves that."""
    torch.manual_seed(0)
    shape = (1, 4, 4, 32, 128)
    value = 7.0

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, _ttnn_padding_to_torch(padding), value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding, value))

    assert output_tensor.shape == torch_output_tensor.shape
    _assert_pad_fill_regions(output_tensor, shape, padding, value)
    _assert_pad_body_matches(output_tensor, torch_input_tensor, padding, atol=0.05)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "shape,padding",
    [
        # Single leading dim: trailing squeeze path still applies memory_config.
        ((1, 4, 4, 32, 128), ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))),
        # Multiple leading dims only: memory_config must be applied after the reshape passes.
        ((2, 4, 4, 4, 128), ((0, 1), (0, 2), (0, 0), (0, 0), (0, 0))),
    ],
    ids=["single_leading_dim", "all_leading_dims"],
)
def test_pad_rank_gt4_memory_config(device, layout, memory_config, shape, padding):
    """The requested output memory config must survive the reshape/pad/reshape round trip."""
    torch.manual_seed(0)
    value = 0.0

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, _ttnn_padding_to_torch(padding), value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=layout, device=device)
    output = ttnn.pad(input_tensor, padding, value, memory_config=memory_config)

    assert output.memory_config().buffer_type == memory_config.buffer_type
    output_tensor = ttnn.to_torch(output)
    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


def test_pad_rank_gt4_padding_shorter_than_rank(device):
    """A padding spec shorter than the rank is left-filled with zeros, as for rank <= 4."""
    torch.manual_seed(0)
    shape = (1, 4, 4, 4, 128)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    full_padding = ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, _ttnn_padding_to_torch(full_padding), value=0.0)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, [(0, 2), (0, 0), (0, 0)], 0.0))

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "padding",
    [
        ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)),
        ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)),
    ],
    ids=["dim0", "dim1", "dim2"],
)
def test_pad_rank_gt4_tile_front_padding_not_supported(device, padding, expect_error):
    """Front padding stays unsupported in tile layout, on upper dims of rank > 4 as well."""
    torch.manual_seed(0)
    torch_input_tensor = torch.randn((2, 4, 4, 32, 128), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "ttnn.pad: on device tile padding does not support front padding"):
        ttnn.pad(input_tensor, padding, 0.0)


@pytest.mark.parametrize("use_multicore", [True, False], ids=["multicore", "singlecore"])
@pytest.mark.parametrize(
    "shape,padding",
    [
        # Leading dim (reshape path) and dim rank-3 (squeeze-to-4D path).
        ((2, 4, 4, 4, 128), ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))),
        ((2, 4, 4, 4, 128), ((0, 0), (0, 1), (0, 2), (0, 0), (0, 0))),
    ],
    ids=["dim0", "dim1_and_dim2"],
)
def test_pad_rank_gt4_use_multicore(device, shape, padding, use_multicore):
    """The rank > 4 path must agree with torch for both the multicore and single-core kernels."""
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(
        torch_input_tensor, _ttnn_padding_to_torch(padding), mode="constant", value=0.0
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding, 0.0, use_multicore=use_multicore))

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


# --- Codegen-path coverage ---
#
# ttnn.pad routes gate-supported TILE cases to codegen and offers no way to ask for a path, so the
# verification-only entries below (see pad_force.hpp) pin one. Goldens come from torch: native is
# the other implementation under test, not a reference.
_pad_force_native = ttnn._ttnn.operations.data_movement.pad_force_native
_pad_force_codegen = ttnn._ttnn.operations.data_movement.pad_force_codegen


def _codegen_input(shape, dtype):
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _torch_pad(x, padding, value):
    if x.dtype == torch.int32:
        value = int(value)
    return torch.nn.functional.pad(x, _ttnn_padding_to_torch(padding), mode="constant", value=value)


# A nonzero value is what exercises the partial-tile fill: from_torch already zeroes the unused lanes.
_pad_codegen_cases = [
    pytest.param(
        [1, 71, 32, 64],
        [[0, 0], [0, 25], [0, 0], [0, 0]],
        0.0,
        ttnn.bfloat16,
        ttnn.L1_MEMORY_CONFIG,
        id="falcon7b_decode_c",
    ),
    pytest.param(
        [1, 2, 32, 63], [[0, 0], [0, 0], [0, 0], [0, 33]], 1.5, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, id="w_mod_31"
    ),
    pytest.param(
        [1, 3, 33, 64],
        [[0, 0], [0, 0], [0, 37], [0, 0]],
        1.5,
        ttnn.bfloat16,
        ttnn.L1_MEMORY_CONFIG,
        id="h_partial_bf16",
    ),
    pytest.param(
        [1, 3, 33, 64],
        [[0, 0], [0, 0], [0, 37], [0, 0]],
        0.0,
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        id="h_partial_bf16_zero",
    ),
    pytest.param(
        [1, 1, 45, 70],
        [[0, 0], [0, 0], [0, 30], [0, 30]],
        1.5,
        ttnn.float32,
        ttnn.DRAM_MEMORY_CONFIG,
        id="hw_partial_fp32",
    ),
    pytest.param(
        [1, 1, 45, 70],
        [[0, 0], [0, 0], [0, 30], [0, 30]],
        -7.0,
        ttnn.int32,
        ttnn.DRAM_MEMORY_CONFIG,
        id="hw_partial_int32",
    ),
    pytest.param(
        [1, 1, 45, 70],
        [[0, 0], [0, 0], [0, 30], [0, 30]],
        3.0,
        ttnn.uint32,
        ttnn.L1_MEMORY_CONFIG,
        id="hw_partial_uint32",
    ),
    # 128 planes x 3 border tiles spreads the fill over many cores.
    pytest.param(
        [8, 16, 40, 40],
        [[0, 1], [0, 0], [0, 0], [0, 0]],
        1.5,
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        id="fill_multicore",
    ),
    # N > 1 with padding on inner dims, so every batch's planes land at a shifted output offset.
    pytest.param(
        [2, 3, 33, 64],
        [[0, 0], [0, 1], [0, 31], [0, 0]],
        1.5,
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        id="n2_inner_pad",
    ),
    pytest.param([45, 70], [[0, 19], [0, 26]], 1.5, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, id="rank2"),
    # Rank 5 pads its leading dim before the 4D codegen call, so both stages run.
    pytest.param(
        [2, 3, 2, 33, 64],
        [[0, 1], [0, 0], [0, 1], [0, 31], [0, 0]],
        1.5,
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        id="rank5",
    ),
]


@pytest.mark.parametrize("shape, padding, value, dtype, memory_config", _pad_codegen_cases)
def test_pad_codegen(device, shape, padding, value, dtype, memory_config):
    skip_unless_dm_codegen_arch(device)
    x = _codegen_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    output = ttnn.to_torch(_pad_force_codegen(xt, padding, value))
    assert_tensors_equal(_torch_pad(x, padding, value), output)


@pytest.mark.parametrize(
    "shape, padding, value, dtype, memory_config",
    [
        pytest.param(
            [1, 71, 32, 64],
            [[0, 0], [0, 25], [0, 0], [0, 0]],
            0.0,
            ttnn.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
            id="falcon7b_decode_c",
        ),
        pytest.param(
            [1, 1, 45, 70],
            [[0, 0], [0, 0], [0, 30], [0, 30]],
            1.5,
            ttnn.float32,
            ttnn.DRAM_MEMORY_CONFIG,
            id="hw_partial_fp32",
        ),
    ],
)
def test_pad_auto_routes_to_codegen(device, shape, padding, value, dtype, memory_config):
    skip_unless_dm_codegen_arch(device)
    x = _codegen_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    device.clear_program_cache()
    # Warm native first so that any program the auto call adds can only be a codegen one.
    _pad_force_native(xt, padding, value)
    entries_before = device.num_program_cache_entries()
    output = ttnn.to_torch(ttnn.pad(xt, padding, value))
    assert_tensors_equal(_torch_pad(x, padding, value), output)
    entries_after_auto = device.num_program_cache_entries()
    assert entries_after_auto > entries_before, "auto added no program over native's; expected the codegen path"
    _pad_force_codegen(xt, padding, value)
    msg = "forced codegen compiled new programs, so auto did not run the codegen ones"
    assert device.num_program_cache_entries() == entries_after_auto, msg


def test_pad_codegen_program_cache(device):
    skip_unless_dm_codegen_arch(device)
    shape = [1, 2, 45, 70]
    padding = [[0, 0], [0, 1], [0, 19], [0, 26]]
    value = 1.5
    entries = None
    resident = []
    outputs = []
    for _ in range(2):
        x = _codegen_input(shape, ttnn.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        resident.append(xt)
        output = _pad_force_codegen(xt, padding, value)
        outputs.append((output, _torch_pad(x, padding, value)))
        assert_tensors_equal(outputs[-1][1], ttnn.to_torch(output))
        if entries is None:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries, "second codegen dispatch missed the program cache"
        # Keep both dispatches' tensors alive and shift the allocator, so the cached programs must rebind buffers.
        resident.append(
            ttnn.from_torch(
                torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
    # The first output is still alive, so the cache hit can only be correct if it rebound the output buffer.
    (first, first_expected), (second, _) = outputs
    assert first.buffer_address() != second.buffer_address(), "second output reused the first output's buffer"
    assert_tensors_equal(first_expected, ttnn.to_torch(first))


def test_pad_codegen_program_rekeys_when_l1_frontier_drops(device):
    skip_unless_dm_codegen_arch(device)
    # The CB batching is planned against the L1 free at dispatch and is part of the program key, so a tensor landing
    # inside the full plan's CB region must build a shallower program rather than replay one that overlaps it. The
    # cost is that a plan cached at warm-up is not the one a trace capture under less L1 needs.
    shape = [1, 1, 64, 64]
    padding = [[0, 0], [0, 0], [0, 32], [0, 32]]
    value = 2.0
    tile_bytes = 32 * 32 * 2  # one bfloat16 tile
    # A plan holds a double-buffered output CB of 2 x batch tiles plus one pad tile. The batch starts at 8 and halves
    # down to 1 as L1 shrinks.
    full_batch, min_batch = 8, 1
    full_plan_bytes = (2 * full_batch + 1) * tile_bytes
    min_plan_bytes = (2 * min_batch + 1) * tile_bytes
    x = _codegen_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expected = _torch_pad(x, padding, value)
    assert_tensors_equal(expected, ttnn.to_torch(_pad_force_codegen(xt, padding, value)))
    entries = device.num_program_cache_entries()

    info = ttnn._ttnn.reports.get_device_info(device)
    tiles_per_bank = (info.cb_limit - full_plan_bytes // 2) // tile_bytes
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    try:
        headroom = resident.buffer_address() - info.address_at_first_l1_cb_buffer
        assert min_plan_bytes <= headroom < full_plan_bytes, f"pinned tensor left {headroom} B of CB headroom"
        assert_tensors_equal(expected, ttnn.to_torch(_pad_force_codegen(xt, padding, value)))
        msg = "the dispatch under less L1 hit the program planned for a clear L1"
        assert device.num_program_cache_entries() == entries + 1, msg
    finally:
        ttnn.deallocate(resident)


@pytest.mark.parametrize(
    "dtype, value",
    [
        pytest.param(ttnn.int32, 3.0e9, id="int32_above_range"),
        pytest.param(ttnn.uint32, -1.0, id="uint32_negative"),
        pytest.param(ttnn.int32, float("nan"), id="int32_nan"),
    ],
)
def test_pad_codegen_out_of_range_integer_value(device, dtype, value):
    skip_unless_dm_codegen_arch(device)
    # The partial-tile lanes and the new tiles must hold one word: the value truncated and wrapped to 32 bits, and
    # zero for NaN.
    shape = [1, 1, 45, 70]
    padding = [[0, 0], [0, 0], [0, 30], [0, 30]]
    x = torch.randint(0, 100, shape, dtype=torch.int32)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(_pad_force_codegen(xt, padding, value)).view(torch.int32)
    word = 0 if math.isnan(value) else (int(value) & 0xFFFFFFFF) - (1 << 32 if int(value) & 0x80000000 else 0)
    expected = torch.full(list(output.shape), word, dtype=torch.int32)
    expected[..., : shape[-2], : shape[-1]] = x
    assert_tensors_equal(expected, output)


def test_pad_codegen_matches_native_bitwise_for_unrepresentable_value(device):
    skip_unless_dm_codegen_arch(device)
    # 0.1 has no exact bfloat16 form, so both routes must round it the same way in the input's partial-tile lanes
    # and in the new tiles. Each route fills its input's partial tiles in place, hence one input per route.
    shape = [1, 2, 45, 70]
    padding = [[0, 0], [0, 0], [0, 19], [0, 26]]
    value = 0.1
    x = torch.rand(shape, dtype=torch.bfloat16)
    native_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    codegen_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    native = ttnn.to_torch(_pad_force_native(native_input, padding, value))
    codegen = ttnn.to_torch(_pad_force_codegen(codegen_input, padding, value))
    assert_tensors_equal(native.view(torch.int16), codegen.view(torch.int16))


def test_pad_codegen_declines_input_padded_past_tile_grid(device, expect_error):
    # The codegen kernels index tiles from the logical shape, so an input whose padded shape holds extra
    # tiles must stay on native. Native is the reference here because the contract is the routing.
    x = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    xt = ttnn.tilize_with_val_padding(rm, [1, 1, 64, 64], 7.0)
    assert list(xt.padded_shape) == [1, 1, 64, 64], f"padded shape {list(xt.padded_shape)}, expected [1, 1, 64, 64]"
    padding = [[0, 0], [0, 0], [0, 0], [0, 32]]
    device.clear_program_cache()
    golden = ttnn.to_torch(_pad_force_native(xt, padding, 0.0))
    entries_before = device.num_program_cache_entries()
    output = ttnn.to_torch(ttnn.pad(xt, padding, 0.0))
    assert_tensors_equal(golden, output)
    assert device.num_program_cache_entries() == entries_before, "auto routed the over-padded input to codegen"
    with expect_error(RuntimeError, "does not support"):
        _pad_force_codegen(xt, padding, 0.0)


def test_pad_codegen_declines_zero_volume_input(device, expect_error):
    # An empty input gives the codegen program no tiles to split over cores, so the gate must refuse it.
    x = torch.rand([0, 1, 32, 32], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    padding = [[0, 0], [0, 0], [0, 0], [0, 32]]
    auto = ttnn.pad(xt, padding, 0.0)
    assert list(auto.shape) == list(_torch_pad(x, padding, 0.0).shape)
    with expect_error(RuntimeError, "does not support"):
        _pad_force_codegen(xt, padding, 0.0)


def test_pad_codegen_partial_tile_input_with_thousands_of_border_tiles_per_core(device):
    """Every plane has a partial border tile, at least 1365 per core: all must be filled before the pad copies them."""
    skip_unless_dm_codegen_arch(device)
    grid = device.compute_with_storage_grid_size()
    c = 1024
    n = -(-1365 * grid.x * grid.y // c)
    shape = [n, c, 30, 30]  # one partial tile per plane
    padding = [[0, 0], [0, 0], [0, 2], [0, 0]]
    value = 1.5
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(_pad_force_codegen(xt, padding, value))
    assert_tensors_equal(_torch_pad(x, padding, value), output)


def test_pad_codegen_uneven_core_split(device):
    skip_unless_dm_codegen_arch(device)
    # One output tile more than twice the core count leaves the work split with a second core group that gets one
    # tile fewer, so that group's runtime args are exercised.
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    shape = [1, 2 * num_cores, 32, 32]
    padding = [[0, 0], [0, 1], [0, 0], [0, 0]]
    value = 1.5
    assert (shape[1] + 1) % num_cores != 0
    x = _codegen_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(_pad_force_codegen(xt, padding, value))
    assert_tensors_equal(_torch_pad(x, padding, value), output)


@pytest.mark.parametrize("value", [0.0, 2.0], ids=["zero", "nonzero"])
def test_pad_codegen_bfloat8_b(device, value):
    skip_unless_dm_codegen_arch(device)
    # Zero pads the block-float tiles directly; a nonzero value pads a bfloat16 copy and re-encodes it.
    shape = [1, 3, 64, 96]
    padding = [[0, 0], [0, 1], [0, 32], [0, 64]]
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(_pad_force_codegen(xt, padding, value))
    assert_allclose(_torch_pad(x, padding, value).float(), output.float(), rtol=0.05, atol=0.025)
    # The padding is whole tiles, so every shared-exponent block in it holds only the value and encodes it exactly.
    _assert_pad_fill_regions(output, shape, padding, value)


@pytest.mark.parametrize(
    "input_placement, output_placement",
    [
        pytest.param(ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, id="l1_to_dram"),
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="dram_to_l1"),
    ],
)
def test_pad_codegen_mixed_placement(device, input_placement, output_placement):
    skip_unless_dm_codegen_arch(device)
    shape = [1, 2, 45, 70]
    padding = [[0, 0], [0, 1], [0, 19], [0, 26]]
    value = 1.5
    x = _codegen_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_placement)
    output = _pad_force_codegen(xt, padding, value, memory_config=output_placement)
    assert output.memory_config().buffer_type == output_placement.buffer_type
    assert_tensors_equal(_torch_pad(x, padding, value), ttnn.to_torch(output))


# The codegen program spreads work over the whole compute grid, so it cannot honour either control.
_SUB_CORE_GRIDS = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
_codegen_execution_controls = [
    pytest.param({"use_multicore": False}, id="single_core"),
    pytest.param({"sub_core_grids": _SUB_CORE_GRIDS}, id="sub_core_grids"),
]
_CODEGEN_REFUSAL = "codegen path does not support"


@pytest.mark.parametrize(
    "case, controls, message",
    [
        pytest.param("row_major", {}, _CODEGEN_REFUSAL, id="row_major"),
        pytest.param("height_sharded", {}, _CODEGEN_REFUSAL, id="height_sharded"),
        pytest.param("uint16", {}, _CODEGEN_REFUSAL, id="uint16"),
        pytest.param("tile_16x32", {}, _CODEGEN_REFUSAL, id="tile_16x32"),
        pytest.param("bfloat8_b_partial_tile", {}, _CODEGEN_REFUSAL, id="bfloat8_b_partial_tile"),
        pytest.param("host", {}, "codegen path needs a device tensor", id="host_tensor"),
        *[pytest.param("tile", p.values[0], _CODEGEN_REFUSAL, id=p.id) for p in _codegen_execution_controls],
    ],
)
def test_pad_codegen_forced_refuses_out_of_scope(device, expect_error, case, controls, message):
    shape = [1, 1, 64, 64]
    padding = [[0, 0], [0, 0], [0, 32], [0, 32]]
    x = torch.rand(shape, dtype=torch.bfloat16)
    tile_kwargs = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "device": device}
    if case == "row_major":
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    elif case == "height_sharded":
        sharded = ttnn.create_sharded_memory_config(
            shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
        )
        xt = ttnn.from_torch(x, memory_config=sharded, **tile_kwargs)
    elif case == "uint16":
        xt = ttnn.from_torch(torch.ones(shape, dtype=torch.bfloat16), **{**tile_kwargs, "dtype": ttnn.uint16})
    elif case == "tile_16x32":
        xt = ttnn.from_torch(x, tile=ttnn.Tile((16, 32)), **tile_kwargs)
    elif case == "bfloat8_b_partial_tile":
        xt = ttnn.from_torch(
            torch.rand([1, 1, 45, 70], dtype=torch.bfloat16), **{**tile_kwargs, "dtype": ttnn.bfloat8_b}
        )
    elif case == "host":
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        xt = ttnn.from_torch(x, **tile_kwargs)
    with expect_error(RuntimeError, message):
        _pad_force_codegen(xt, padding, 1.5, **controls)


@pytest.mark.parametrize("controls", _codegen_execution_controls)
def test_pad_codegen_auto_keeps_execution_controls_on_native(device, controls):
    # Without the controls this input routes to codegen (see test_pad_auto_routes_to_codegen).
    shape = [1, 1, 45, 70]
    padding = [[0, 0], [0, 0], [0, 30], [0, 30]]
    value = 1.5
    x = _codegen_input(shape, ttnn.float32)
    xt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    device.clear_program_cache()
    # Warm native first so that any program the auto call adds can only be a codegen one. Compared against native
    # rather than torch: native's single-core tile pad does not keep float32 exact.
    expected = ttnn.to_torch(_pad_force_native(xt, padding, value, **controls))
    entries_before = device.num_program_cache_entries()
    output = ttnn.to_torch(ttnn.pad(xt, padding, value, **controls))
    assert torch.equal(output, expected)
    msg = "auto routed a pad with an execution control to codegen"
    assert device.num_program_cache_entries() == entries_before, msg
