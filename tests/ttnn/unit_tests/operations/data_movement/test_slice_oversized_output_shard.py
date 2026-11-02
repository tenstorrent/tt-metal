# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("explicit_memory", [False, True])
@pytest.mark.parametrize(
    "shape,grid_shape,shard,start,end,step",
    [
        ((32, 63), (1, 1), (32, 63), (25, 36), (26, 43), (1, 1)),
        ((64, 40), (4, 2), (32, 10), (59, 24), (61, 29), (1, 1)),
        ((64, 40), (4, 2), (32, 10), (59, 24), (61, 29), (1, 3)),
        ((32, 64), (1, 1), (32, 64), (25, 36), (26, 43), (1, 1)),
    ],
)
def test_slice_preserves_values_with_oversized_output_shard(
    device, explicit_memory, shape, grid_shape, shard, start, end, step
):
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_shape[0] - 1, grid_shape[1] - 1))}
    )
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    host = (torch.arange(shape[0] * shape[1]).reshape(shape) % 127).to(torch.bfloat16)
    source = ttnn.from_torch(host, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory)
    kwargs = {"memory_config": memory} if explicit_memory else {}
    result = ttnn.slice(source, start, end, step, **kwargs)
    expected = host[start[0] : end[0] : step[0], start[1] : end[1] : step[1]]
    assert torch.equal(ttnn.to_torch(result), expected)
    assert torch.equal(ttnn.to_torch(source), host)
