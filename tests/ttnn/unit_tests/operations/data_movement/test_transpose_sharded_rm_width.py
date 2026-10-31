# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "shape",
    [
        [2, 64, 16],
        [2, 256, 16],
        [4, 64, 8],
        [2, 64, 48],
        [2, 320, 16],
        [2, 64, 32],
        # Heights that are not a multiple of 32. The batch counts make each core's shard a multiple of 1024
        # elements, which a row-major shard needs to reach the sharded factory at all.
        [8, 48, 16],
        [16, 40, 48],
        [8, 16, 16],
    ],
    ids=["64x16", "256x16", "64x8", "64x48", "320x16-control", "64x32-control", "48x16", "40x48", "16x16"],
)
def test_transpose_height_sharded_row_major_width_not_multiple_of_32(device, shape):
    """Row-major height-sharded WH transpose must be exact and stay inside its output shard for any width."""
    torch.manual_seed(0)
    mem = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=2, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    x, v = torch.randn(shape), torch.randn(shape)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    # An unrelated tensor with the same sharding: a write past the output shard lands here or on the input.
    other = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = ttnn.transpose(t, 1, 2)
    assert torch.equal(ttnn.to_torch(out), x.transpose(1, 2))
    assert torch.equal(ttnn.to_torch(other), v)
    assert torch.equal(ttnn.to_torch(t), x)
