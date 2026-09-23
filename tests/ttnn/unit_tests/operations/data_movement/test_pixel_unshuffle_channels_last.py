# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.pixel_unshuffle(channels_last=True): NHWC, channel-padded, height-sharded L1 output,
checked bit-exact against torch for both channel orders, 2- and 4-byte elements, r=2 and r=4,
and shards that do not start on an image-row boundary."""

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn


def golden_nhwc(x, r, channel_order, padded_channels):
    if channel_order == ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR:
        s = F.pixel_unshuffle(x, r)  # c_out = c*r*r + dy*r + dx
    else:
        N, C, H, W = x.shape  # c_out = dy*(r*C) + dx*C + c  (ONNX SpaceToDepth)
        s = x.reshape(N, C, H // r, r, W // r, r).permute(0, 3, 5, 1, 2, 4).reshape(N, C * r * r, H // r, W // r)
    s = s.permute(0, 2, 3, 1)  # NHWC
    return F.pad(s, (0, padded_channels - s.shape[-1]))


def hs_config(grid_xy, shard_h, shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_xy[0] - 1, grid_xy[1] - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )


CM = ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR
SM = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR


@pytest.mark.parametrize(
    "shape, r, padded_channels, grid, channel_order, dtype",
    [
        # EVO50 stem: 3 ch -> 12, padded to 16, 64 cores x 2496 pixels (3 image rows each)
        ((1, 3, 384, 1664), 2, 16, (8, 8), CM, ttnn.bfloat16),
        # default padding (None -> round up to L1 alignment = 16 for 12 ch)
        ((1, 3, 384, 1664), 2, None, (8, 8), CM, ttnn.bfloat16),
        # same geometry, ONNX channel order (element path)
        ((1, 3, 64, 128), 2, 16, (4, 4), SM, ttnn.bfloat16),
        # r = 4, C = 1 -> 16 channels, no padding
        ((1, 1, 64, 256), 4, 16, (4, 4), CM, ttnn.bfloat16),
        # r = 2, C = 2 -> 8 channels, no padding, 8-element sticks (16 B)
        ((1, 2, 32, 64), 2, 8, (2, 2), CM, ttnn.bfloat16),
        # shards that cut image rows: 15 x 20 = 300 pixels over 7 cores -> 43 per core
        ((1, 3, 30, 40), 2, 16, (7, 1), CM, ttnn.bfloat16),
        ((1, 3, 30, 40), 2, 16, (7, 1), SM, ttnn.bfloat16),
        # batch > 1
        ((2, 3, 32, 64), 2, 16, (4, 2), CM, ttnn.bfloat16),
        # odd r (element path), 32 B-aligned split constraint on r=3
        ((1, 2, 30, 48), 3, 24, (5, 1), CM, ttnn.bfloat16),
        # 4-byte elements: 8-channel sticks = 32 B
        ((1, 2, 32, 64), 2, 8, (2, 2), CM, ttnn.float32),
        ((1, 2, 32, 64), 2, 8, (2, 2), SM, ttnn.float32),
    ],
)
def test_pixel_unshuffle_channels_last(device, shape, r, padded_channels, grid, channel_order, dtype):
    torch.manual_seed(0)
    N, C, H, W = shape
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.randn(shape, dtype=torch_dtype)

    total = N * (H // r) * (W // r)
    ncores = grid[0] * grid[1]
    shard_h = math.ceil(total / ncores)
    cp = padded_channels if padded_channels is not None else 16
    mem = hs_config(grid, shard_h, cp)

    tt_x = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = dict(memory_config=mem, channel_order=channel_order, channels_last=True)
    if padded_channels is not None:
        kwargs["padded_channels"] = padded_channels
    y = ttnn.pixel_unshuffle(tt_x, r, **kwargs)

    assert list(y.shape) == [N, H // r, W // r, cp], y.shape
    assert y.layout == ttnn.ROW_MAJOR_LAYOUT
    assert y.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    got = ttnn.to_torch(y)
    ref = golden_nhwc(x, r, channel_order, cp)
    assert got.shape == ref.shape
    mismatch = (got != ref).sum().item()
    assert (
        mismatch == 0
    ), f"{mismatch} of {ref.numel()} elements differ (max |d| {(got.float() - ref.float()).abs().max().item()})"
