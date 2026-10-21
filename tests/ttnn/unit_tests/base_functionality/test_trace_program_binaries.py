# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _buffers(device):
    data = torch.linspace(-1, 1, 1024, dtype=torch.bfloat16).reshape(1, 1, 32, 32)
    source = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.empty_like(source)
    ttnn.neg(source, output_tensor=output)
    return data, source, output


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}, {"trace_region_size": 0}], indirect=True)
def test_late_program_binary_above_trace_footprint(device):
    data, source, output = _buffers(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    temporary = ttnn.neg(source)
    ttnn.neg(temporary, output_tensor=output)
    temporary.deallocate(True)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        programs = device.num_program_cache_entries()
        ttnn.relu(source, output_tensor=output)
        assert device.num_program_cache_entries() > programs
        assert not ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace)
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(output), data)
            ttnn.relu(source, output_tensor=output)
            assert torch.equal(ttnn.to_torch(output), data.relu())
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize("allocate_during_capture", [False, True])
@pytest.mark.parametrize("overlapping_captures", [False, True])
def test_late_program_binary_rejects_trace_footprint(
    device, expect_error, allocate_during_capture, overlapping_captures
):
    data, source, output = _buffers(device)
    memory = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    tiles_per_bank = memory.largest_contiguous_bytes_free_per_bank // 2048

    def reserve():
        # Reserve addresses without reading or writing the large backing allocation.
        return ttnn.empty(
            (1, memory.num_banks, tiles_per_bank * 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if not allocate_during_capture:
        scratch = reserve()
    first = ttnn.begin_trace_capture(device, cq_id=0)
    if allocate_during_capture:
        scratch = reserve()
    scratch.deallocate(True)
    ttnn.neg(source, output_tensor=output)
    if overlapping_captures:
        second = ttnn.begin_trace_capture(device, cq_id=1)
    ttnn.end_trace_capture(device, first, cq_id=0)
    if not overlapping_captures:
        second = ttnn.begin_trace_capture(device, cq_id=1)
    with ttnn.command_queue(1):
        ttnn.neg(source, output_tensor=output)
    ttnn.end_trace_capture(device, second, cq_id=1)
    try:
        with expect_error(RuntimeError, "Kernel binary buffer.*overlaps a live trace"):
            ttnn.relu(source, output_tensor=output)
        ttnn.execute_trace(device, first, cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(output), -data)
    finally:
        ttnn.release_trace(device, first)
        ttnn.release_trace(device, second)
    ttnn.relu(source, output_tensor=output)
    assert torch.equal(ttnn.to_torch(output), data.relu())
