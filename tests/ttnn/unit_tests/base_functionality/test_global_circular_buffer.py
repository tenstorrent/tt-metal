# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import skip_for_slow_dispatch
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


def _mapping():
    return [(ttnn.CoreCoord(0, 0), ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}))]


def test_global_circular_buffer_fixed_addresses(device, expect_error):
    original = ttnn.create_global_circular_buffer(device, _mapping(), 3200)
    addresses = dict(buffer_address=original.buffer_address(), config_address=original.config_address())
    ttnn.synchronize_device(device)
    with expect_error(RuntimeError, "Requested buffer address"):
        ttnn.create_global_circular_buffer(device, _mapping(), 3200, **addresses)
    original.deallocate()
    for _ in range(3):
        restored = ttnn.create_global_circular_buffer(device, _mapping(), 3200, **addresses)
        assert restored.buffer_address() == addresses["buffer_address"]
        assert restored.config_address() == addresses["config_address"]
        ttnn.synchronize_device(device)
        restored.deallocate()
    for name, address in addresses.items():
        with expect_error(RuntimeError, "addresses must be supplied together"):
            ttnn.create_global_circular_buffer(device, _mapping(), 3200, **{name: address})


@skip_for_slow_dispatch()
@pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 at startup"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_global_circular_buffer_acknowledges_only_its_allocations(device, expect_error):
    trace_input = ttnn.from_torch(
        torch.ones((1, 1, 32, 32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    warmup = ttnn.neg(trace_input)
    warmup.deallocate()
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = ttnn.neg(trace_input)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        unrelated = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        before = set(ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id))
        assert unrelated.buffer_unique_id() in before
        global_cb = ttnn.create_global_circular_buffer(device, _mapping(), 3200)
        after = set(ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id))
        assert len(after - before) == 2
        global_cb.acknowledge_corruptible()
        assert set(ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)) == before
        with expect_error(RuntimeError, "still alive before trace replay"):
            trace_allocation_tracker.TraceAllocationTracker.verify_before_replay(device, trace_id)
        unrelated.deallocate()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        torch.testing.assert_close(ttnn.to_torch(output), -torch.ones((1, 1, 32, 32)), check_dtype=False)
        global_cb.deallocate()
    finally:
        ttnn.release_trace(device, trace_id)
