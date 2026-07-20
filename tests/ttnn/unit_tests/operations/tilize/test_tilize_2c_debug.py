# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2c debug tests. DO NOT DELETE.

Isolates the 2c sub-levers with the exact golden shapes so each can be
debugged independently of the full golden matrix.
"""

import pytest
import torch
import ttnn


# --- Lever 2: cliff/padded same-spec nd (input spec == output spec) ---
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([4, 128, 128], [2, 64, 64]),  # even, no pad (2b baseline)
        ([3, 160, 160], [2, 64, 64]),  # cliff: 18 shards / 4 cores + padding
        ([5, 4, 160, 160], [2, 3, 64, 96]),  # 36 / 4 = 9, padded dims
        ([23, 96, 160], [4, 64, 96]),  # 24 / 4 = 6, cliff on dim0/dim2
    ],
)
def test_2c_samespec_cliff_nd(device, tensor_shape, shard_shape):
    torch.manual_seed(42)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    nd = ttnn.NdShardSpec(shard_shape=shard_shape, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd)

    x = torch.rand(tensor_shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = ttnn.tilize(t, memory_config=mc, use_multicore=True)
    res = ttnn.to_torch(out)
    assert torch.equal(x, res), f"mismatch shape={tensor_shape} max_diff={(x.float()-res.float()).abs().max()}"
