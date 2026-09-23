# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Nearest upsample of ROW_MAJOR block-sharded inputs, including COL_MAJOR shard orientation and ragged last shards
(the shard grid is not a divisor of the rows): the sharded factory must map the config tensor and the stick bytes
with the shard orientation. Nearest-neighbour is exact, so the result must equal torch."""

import pytest
import torch

import ttnn

# (H = W, C, grid (x, y), orientation, shard rows)
CASES = [
    (64, 640, (8, 8), ttnn.ShardOrientation.ROW_MAJOR, 512),
    (64, 640, (8, 10), ttnn.ShardOrientation.COL_MAJOR, 512),
    (64, 640, (11, 10), ttnn.ShardOrientation.COL_MAJOR, 384),  # ragged: 11 x 384 >= 4096
    (32, 1280, (11, 10), ttnn.ShardOrientation.COL_MAJOR, 96),  # ragged: 11 x 96 >= 1024
    (64, 640, (10, 10), ttnn.ShardOrientation.ROW_MAJOR, 416),  # ragged: 10 x 416 >= 4096
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("hw, C, grid_xy, orientation, shard_rows", CASES)
def test_upsample_block_sharded(device, hw, C, grid_xy, orientation, shard_rows):
    gx, gy = grid_xy
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < gx or grid_size.y < gy:
        pytest.skip(f"needs a {gx}x{gy} worker grid, device has {grid_size.x}x{grid_size.y}")
    torch.manual_seed(0)
    x = torch.randn(1, hw, hw, C).bfloat16()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    shard_w = C // (gx if orientation == ttnn.ShardOrientation.ROW_MAJOR else gy)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows, shard_w], orientation),
    )
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    y = ttnn.upsample(t, (2, 2))
    ref = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2).float(), scale_factor=2, mode="nearest")
    assert torch.equal(ttnn.to_torch(y).float(), ref.permute(0, 2, 3, 1))
