# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.tools import trace_allocation_tracker


def run_global_circular_buffer(device):
    sender_cores = [ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2)]
    receiver_cores = [
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        ),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(2, 3),
                    ttnn.CoreCoord(2, 4),
                ),
            }
        ),
    ]
    sender_receiver_mapping = list(zip(sender_cores, receiver_cores))

    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 3200)


def test_global_circular_buffer(device):
    run_global_circular_buffer(device)


def test_global_circular_buffer_mesh(mesh_device):
    run_global_circular_buffer(mesh_device)


@pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 at startup"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}, {"trace_region_size": 0}], indirect=True)
def test_global_cb_acknowledgement_is_per_trace(device, expect_error):
    """Both backing buffers remain unsafe for every trace except the named one."""
    shape = (1, 1, 32, 32)
    trace_input = ttnn.from_torch(torch.ones(shape), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    output = ttnn.neg(trace_input)
    trace_ids = []
    try:
        for _ in range(2):
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            ttnn.neg(trace_input, output_tensor=output)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            trace_ids.append(trace_id)

        mapping = [
            (ttnn.CoreCoord(0, 0), ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}))
        ]
        global_cb = ttnn.create_global_circular_buffer(device, mapping, 3200)
        get_unsafe = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids
        before = get_unsafe(device, trace_ids[1])
        assert len(before) == 2  # GCB data and configuration allocations.
        assert get_unsafe(device, trace_ids[0]) == before

        # These tiny neg traces do not use the GCB's region. Only acknowledge one
        # so the other exercises rejection without dispatching unsafe work.
        global_cb.acknowledge_restored_trace(trace_ids[0])
        assert not get_unsafe(device, trace_ids[0])
        assert get_unsafe(device, trace_ids[1]) == before
        assert set(before) <= set(ttnn._ttnn.operations.trace.get_all_unsafe_tracked_ids())
        ttnn.execute_trace(device, trace_ids[0], cq_id=0, blocking=True)
        with expect_error(RuntimeError, "still alive before trace replay"):
            ttnn.execute_trace(device, trace_ids[1], cq_id=0, blocking=True)

        # Acknowledging the GCB must not hide an unrelated allocation.
        unrelated = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        global_cb.acknowledge_restored_trace(trace_ids[0])
        assert unrelated.buffer_unique_id() in get_unsafe(device, trace_ids[0])
        ttnn.deallocate(unrelated)
        ttnn.synchronize_device(device)
        del global_cb
        assert not get_unsafe(device, trace_ids[1])
        ttnn.execute_trace(device, trace_ids[1], cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(output), -torch.ones(shape))
    finally:
        for trace_id in trace_ids:
            ttnn.release_trace(device, trace_id)
