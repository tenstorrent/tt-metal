# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf lamp L1 (issue lanes per core) on both production cases, bit-exact checked.

Dispatches the op's own program descriptor with an explicit num_lanes, bypassing validate() so the
p1 production case (outside Phase 0 SUPPORTED) can be measured on the same program. Run under
`scripts/run_safe_pytest.sh --profile --no-py-zones` to read DEVICE KERNEL DURATION per case.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import (
    get_program_descriptor,
)

TILE = 32

CASES = {
    "p2": (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        (((0, 0), (7, 3)),),
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    "p1": (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        (((0, 0), (10, 8)), ((0, 9), (3, 9))),
        ttnn.L1_MEMORY_CONFIG,
    ),
}


def _make(shape):
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols)
    t = (torch.arange(rows) // TILE)[:, None] * (-(-cols // TILE)) + (torch.arange(cols) // TILE)[None, :]
    return (x * torch.pow(2.0, (t % 16 - 8).float())).reshape(shape).to(torch.bfloat16)


@pytest.mark.parametrize("num_lanes", [1, 2])
@pytest.mark.parametrize("case", list(CASES))
def test_lane_lamp(device, case, num_lanes):
    torch.manual_seed(0)
    shape, dtype, layout, shard, grid, out_mc = CASES[case]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
    in_mc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR))
    src = _make(shape)
    x = ttnn.from_torch(src, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mc)
    expected = src if dtype == ttnn.bfloat16 else ttnn.to_torch(x)
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, ttnn.TILE_LAYOUT, device, out_mc)
    out = ttnn.generic_op([x, out], get_program_descriptor(x, out, num_lanes))
    got = ttnn.to_torch(out)
    differ = int((got.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).sum())
    ttnn.deallocate(out)
    ttnn.deallocate(x)
    assert differ == 0, f"{differ} elements differ"
