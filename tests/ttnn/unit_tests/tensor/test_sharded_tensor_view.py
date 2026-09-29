# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

NUM_SHARDS = 2
OWNER_SHARD_HEIGHT = 64
VIEW_SHARD_HEIGHT = 32
SHARD_WIDTH = 64


def _height_sharded_l1_spec(dtype, layout, shard_height):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_SHARDS - 1, 0))})
    return ttnn.TensorSpec(
        shape=[NUM_SHARDS * shard_height, SHARD_WIDTH],
        dtype=dtype,
        layout=layout,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_spec=ttnn.ShardSpec(shard_grid, [shard_height, SHARD_WIDTH], ttnn.ShardOrientation.ROW_MAJOR),
        buffer_type=ttnn.BufferType.L1,
    )


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize(
    "dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)], ids=["bf16", "fp32"]
)
def test_sharded_tensor_view_reads_lower_rows_of_each_shard(device, layout, dtype, torch_dtype):
    rows = torch.arange(NUM_SHARDS * OWNER_SHARD_HEIGHT, dtype=torch.float32).unsqueeze(1)
    # Exact in bfloat16: magnitudes stay below 256 with at most one fractional bit.
    owner_values = (rows + 0.5 * (torch.arange(SHARD_WIDTH) % 2)).to(torch_dtype)
    owner = ttnn.from_torch(
        owner_values, spec=_height_sharded_l1_spec(dtype, layout, OWNER_SHARD_HEIGHT), device=device
    )
    upper_rows = OWNER_SHARD_HEIGHT - VIEW_SHARD_HEIGHT
    view = ttnn.experimental.create_sharded_tensor_view(
        owner,
        _height_sharded_l1_spec(dtype, layout, VIEW_SHARD_HEIGHT),
        upper_rows * SHARD_WIDTH * owner.element_size(),
    )
    expected = owner_values.reshape(NUM_SHARDS, OWNER_SHARD_HEIGHT, SHARD_WIDTH)[:, upper_rows:, :].reshape(
        -1, SHARD_WIDTH
    )
    assert torch.equal(ttnn.to_torch(view), expected), "view does not read the lower rows of each owner shard"

    # The view keeps the owner's allocation alive after the last Python reference to the owner is dropped.
    del owner
    assert torch.equal(ttnn.to_torch(view), expected), "view changed after the owner tensor was released"
    ttnn.deallocate(view)
