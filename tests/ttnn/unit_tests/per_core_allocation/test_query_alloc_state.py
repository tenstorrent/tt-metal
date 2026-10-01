# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental_get_l1_occupied_ranges under the HYBRID allocator."""

import torch
import ttnn
from conftest import requires_hybrid_allocator


def _single_core_tensor(device, core, shard_bytes, per_core):
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, shard_bytes], ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    if per_core:
        mem_config.experimental_set_per_core_allocation(True)
    data = torch.zeros(1, shard_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )


def _covers(ranges, begin, end):
    return any(start <= begin and end <= stop for start, stop in ranges)


def _overlaps(ranges, begin, end):
    return any(start < end and begin < stop for start, stop in ranges)


@requires_hybrid_allocator
def test_occupied_ranges_track_per_core_tensor(device):
    core = ttnn.CoreCoord(0, 0)
    other = ttnn.CoreCoord(1, 0)
    shard_bytes = 1024
    tensor = _single_core_tensor(device, core, shard_bytes, per_core=True)
    (coord,) = tensor.device_coords()
    addr = tensor.experimental_per_core_buffer_address(coord, core)

    ranges = ttnn.experimental_get_l1_occupied_ranges(device, coord, core)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)
    assert _covers(ranges, addr, addr + shard_bytes)
    assert not _overlaps(ttnn.experimental_get_l1_occupied_ranges(device, coord, other), addr, addr + shard_bytes)

    ttnn.deallocate(tensor)
    assert not _overlaps(ttnn.experimental_get_l1_occupied_ranges(device, coord, core), addr, addr + shard_bytes)


@requires_hybrid_allocator
def test_occupied_ranges_track_lockstep_tensor_on_every_core(device):
    core = ttnn.CoreCoord(0, 0)
    shard_bytes = 1024
    tensor = _single_core_tensor(device, core, shard_bytes, per_core=False)
    (coord,) = tensor.device_coords()
    addr = tensor.buffer_address()

    by_core = ttnn.experimental_get_l1_occupied_ranges(device, coord)
    assert by_core
    for ranges in by_core.values():
        assert _covers(ranges, addr, addr + shard_bytes)

    ttnn.deallocate(tensor)


@requires_hybrid_allocator
def test_whole_device_matches_per_core_query(device):
    core = ttnn.CoreCoord(0, 0)
    tensor = _single_core_tensor(device, core, 1024, per_core=True)
    (coord,) = tensor.device_coords()

    by_core = ttnn.experimental_get_l1_occupied_ranges(device, coord)
    assert core in by_core
    for c, ranges in by_core.items():
        assert ranges == ttnn.experimental_get_l1_occupied_ranges(device, coord, c)

    ttnn.deallocate(tensor)
