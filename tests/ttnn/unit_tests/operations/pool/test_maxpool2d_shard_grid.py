# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shard grids that are not origin-anchored and row-filled must fail loudly.

Halo and pool address shard i as (i % row_width, i / row_width) from core (0, 0).
On any other grid, max_pool2d/avg_pool2d hang and conv2d with override_sharding_config
returns silently wrong data (#60513).
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})

IN_N, IN_C, IN_H, IN_W = 1, 32, 8, 8
POOL_KERNEL = (2, 2)
POOL_STRIDE = (2, 2)
POOL_PAD = (0, 0)
POOL_DILATION = (1, 1)

# (x0, y0, x1, y1) ranges. Supported grids are filled row by row from (0, 0).
SUPPORTED_GRIDS = [
    pytest.param([(0, 0, 3, 0)], id="origin-row"),
    pytest.param([(0, 0, 1, 1)], id="origin-rect"),
    pytest.param([(0, 0, 3, 0), (0, 1, 2, 1)], id="origin-short-last-row"),
]
REJECTED_GRIDS = [
    pytest.param([(0, 2, 3, 2)], id="offset-row"),
    pytest.param([(2, 2, 3, 3)], id="offset-rect"),
    pytest.param([(0, 0, 1, 3), (2, 0, 2, 2)], id="column-filled"),
    pytest.param([(0, 0, 1, 1), (4, 4, 5, 5)], id="disjoint"),
]


@pytest.fixture(scope="module")
def tensor_map():
    return {}


def _num_cores(ranges):
    return sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in ranges)


def _fits(device, ranges):
    grid = device.compute_with_storage_grid_size()
    return all(x1 < grid.x and y1 < grid.y for x0, y0, x1, y1 in ranges)


def _core_range_set(ranges):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges}
    )


def _height_sharded_config(ranges):
    num_cores = _num_cores(ranges)
    in_nhw = IN_N * IN_H * IN_W
    shard_height = math.ceil(math.ceil(in_nhw / num_cores) / 32) * 32
    shard_width = math.ceil(IN_C / 32) * 32
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            _core_range_set(ranges),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _sharded_input(device, ranges):
    torch.manual_seed(0)
    torch_input = torch.arange(IN_N * IN_H * IN_W * IN_C, dtype=torch.float32).reshape(1, 1, IN_N * IN_H * IN_W, IN_C)
    tt_input = ttnn.from_torch(
        torch_input,
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_height_sharded_config(ranges),
    )
    return torch_input, tt_input


@pytest.mark.parametrize("ranges", SUPPORTED_GRIDS)
def test_max_pool2d_supported_shard_grid(device, ranges, tensor_map):
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    run_max_pool2d(
        [IN_N, IN_C, IN_H, IN_W],
        POOL_KERNEL,
        POOL_PAD,
        POOL_STRIDE,
        POOL_DILATION,
        device,
        tensor_map,
        ttnn.bfloat16,
        in_memory_config=_height_sharded_config(ranges),
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("ranges", REJECTED_GRIDS)
def test_max_pool2d_rejected_shard_grid(device, ranges, expect_error):
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    _, tt_input = _sharded_input(device, ranges)
    with expect_error(RuntimeError, "origin-anchored"):
        ttnn.max_pool2d(
            input_tensor=tt_input,
            batch_size=IN_N,
            input_h=IN_H,
            input_w=IN_W,
            channels=IN_C,
            kernel_size=list(POOL_KERNEL),
            stride=list(POOL_STRIDE),
            padding=list(POOL_PAD),
            dilation=list(POOL_DILATION),
            config_tensor_in_dram=True,
        )


def test_avg_pool2d_rejected_shard_grid(device, expect_error):
    ranges = [(0, 2, 3, 2)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    _, tt_input = _sharded_input(device, ranges)
    with expect_error(RuntimeError, "origin-anchored"):
        ttnn.avg_pool2d(
            input_tensor=tt_input,
            batch_size=IN_N,
            input_h=IN_H,
            input_w=IN_W,
            channels=IN_C,
            kernel_size=list(POOL_KERNEL),
            stride=list(POOL_STRIDE),
            padding=list(POOL_PAD),
            config_tensor_in_dram=True,
        )


def _run_conv2d(
    device, ranges, *, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, channels=IN_C, transpose_shards=False
):
    torch.manual_seed(0)
    torch_input = torch.randn(IN_N, channels, IN_H, IN_W)
    torch_weight = torch.randn(channels, channels, 3, 3)
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    tt_input = ttnn.from_torch(torch_input_nhwc, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, ttnn.bfloat16)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        core_grid=_core_range_set(ranges),
        override_sharding_config=True,
        transpose_shards=transpose_shards,
    )
    tt_output, [out_h, out_w] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=channels,
        out_channels=channels,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        batch_size=IN_N,
        input_height=IN_H,
        input_width=IN_W,
        conv_config=conv_config,
        return_output_dim=True,
        dtype=ttnn.bfloat16,
    )
    return torch_input, torch_weight, tt_output, out_h, out_w


def test_conv2d_supported_shard_grid(device):
    ranges = [(0, 0, 1, 1)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    torch_input, torch_weight, tt_output, out_h, out_w = _run_conv2d(device, ranges)
    ref = torch.nn.functional.conv2d(torch_input, torch_weight, padding=1)
    out = ttnn.to_torch(tt_output).reshape(IN_N, out_h, out_w, -1)[:, :, :, :IN_C]
    out = out.permute(0, 3, 1, 2)
    assert_with_pcc(ref, out, pcc=0.99)


def test_conv2d_rejected_shard_grid(device, expect_error):
    # Column-filled grid: shard i is not at (i % width, i / width).
    ranges = [(0, 0, 1, 3), (2, 0, 2, 2)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    with expect_error(RuntimeError, "origin-anchored"):
        _run_conv2d(device, ranges)


# 64 channels on a 2-wide grid so each block-sharded core owns a full tile of channels.
# Block sharding takes its row width from ranges()[0].end.x + 1, not the bounding box.
BLOCK_CHANNELS = 64


def test_conv2d_block_sharded_supported_shard_grid(device):
    ranges = [(0, 0, 1, 1)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    torch_input, torch_weight, tt_output, out_h, out_w = _run_conv2d(
        device,
        ranges,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        channels=BLOCK_CHANNELS,
    )
    ref = torch.nn.functional.conv2d(torch_input, torch_weight, padding=1)
    out = ttnn.to_torch(tt_output).reshape(IN_N, out_h, out_w, -1)[:, :, :, :BLOCK_CHANNELS]
    out = out.permute(0, 3, 1, 2)
    assert_with_pcc(ref, out, pcc=0.99)


def test_conv2d_block_sharded_rejected_shard_grid(device, expect_error):
    ranges = [(2, 2, 3, 3)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    with expect_error(RuntimeError, "origin-anchored"):
        _run_conv2d(
            device,
            ranges,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            channels=BLOCK_CHANNELS,
        )


def test_conv2d_width_sharded_supported_shard_grid(device):
    ranges = [(0, 0, 1, 0)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    torch_input, torch_weight, tt_output, out_h, out_w = _run_conv2d(
        device,
        ranges,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        channels=BLOCK_CHANNELS,
    )
    ref = torch.nn.functional.conv2d(torch_input, torch_weight, padding=1)
    out = ttnn.to_torch(tt_output).reshape(IN_N, out_h, out_w, -1)[:, :, :, :BLOCK_CHANNELS]
    out = out.permute(0, 3, 1, 2)
    assert_with_pcc(ref, out, pcc=0.99)


def test_conv2d_width_sharded_rejected_shard_grid(device, expect_error):
    ranges = [(0, 2, 1, 2)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    with expect_error(RuntimeError, "origin-anchored"):
        _run_conv2d(
            device,
            ranges,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            channels=BLOCK_CHANNELS,
        )


def test_conv2d_block_sharded_col_major_origin_grid(device):
    # Column-major block sharding is transpose_mcast on an origin rectangle. The row walk must accept it.
    ranges = [(0, 0, 1, 1)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    torch_input, torch_weight, tt_output, out_h, out_w = _run_conv2d(
        device,
        ranges,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        channels=BLOCK_CHANNELS,
        transpose_shards=True,
    )
    ref = torch.nn.functional.conv2d(torch_input, torch_weight, padding=1)
    out = ttnn.to_torch(tt_output).reshape(IN_N, out_h, out_w, -1)[:, :, :, :BLOCK_CHANNELS]
    out = out.permute(0, 3, 1, 2)
    assert_with_pcc(ref, out, pcc=0.99)


def test_height_sharded_col_major_grid_rejected(device, expect_error):
    # Shard i is stored column-wise, while halo still addresses it with the row walk.
    ranges = [(0, 0, 1, 1)]
    if not _fits(device, ranges):
        pytest.skip("grid does not fit on this device")
    torch_input = torch.zeros((1, 4, 4, 8), dtype=torch.bfloat16)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core_range_set(ranges), [4, 8], ttnn.ShardOrientation.COL_MAJOR),
    )
    tt_input = ttnn.from_torch(
        torch_input, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    with expect_error(RuntimeError, "origin-anchored"):
        ttnn.fold(tt_input, stride_h=1, stride_w=1, padding=[1, 0])
