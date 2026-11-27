# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace replay (BB-16): a replayed call writes the same bytes as the eager call, from the cached program."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved


def _p2_input(device):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(3)
    torch_input = torch.randn(1, 5120).to(torch.bfloat16)
    return torch_input, ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )


@pytest.mark.use_module_device({"trace_region_size": 64 * 1024})
def test_trace_replay_matches_eager(device):
    torch_input, tt_input = _p2_input(device)
    eager = g6_sharded_to_interleaved(tt_input, ttnn.DRAM_MEMORY_CONFIG)
    eager_bits = ttnn.to_torch(eager).view(torch.int16)
    assert torch.equal(eager_bits, torch_input.view(torch.int16))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = g6_sharded_to_interleaved(tt_input, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(traced).view(torch.int16), eager_bits)
    ttnn.release_trace(device, trace_id)

    for t in (traced, eager, tt_input):
        ttnn.deallocate(t)
