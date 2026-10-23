# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import tempfile
import threading
import typing
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import ttnn
from ttnn.tools import trace_allocation_tracker
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import skip_for_slow_dispatch


@skip_for_slow_dispatch()
@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_single_device_single_trace(device, shape, blocking):
    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chain to be traced
    def run_op_chain(input_0, input_1):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev)

    # Capture Trace
    logger.info("Capture Trace")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensor_1 = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        )

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(torch_input_tensor_0, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor_1 = ttnn.from_torch(torch_input_tensor_1, layout=ttnn.TILE_LAYOUT)

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)

        if blocking:
            ttnn.synchronize_device(device)
        logger.info("Execute Trace")
        # Execute trace
        ttnn.execute_trace(device, tid, cq_id=0, blocking=blocking)
        # Readback data
        logger.info("Read Back Trace Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)

    ttnn.release_trace(device, tid)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 32, 32)])
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_single_device_multi_trace(device, shape, blocking):
    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    weight_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chains to be traced
    def run_op_chain(input_0, input_1, weight):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1))) @ ttnn.silu(
            weight
        )

    def run_op_chain_1(input_0, input_1, weight):
        return ttnn.gelu(ttnn.add(ttnn.tanh(ttnn.mul(ttnn.sub(input_0, input_1), weight)), input_1))

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev, weight_dev)
    run_op_chain_1(input_0_dev, input_1_dev, weight_dev)

    # Capture Trace 0
    logger.info("Capture Trace 0")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Capture Trace 1
    logger.info("Capture Trace 1")
    tid_1 = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor_1 = run_op_chain_1(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(device, tid_1, cq_id=0)

    # Execute and verify trace against pytorch
    torch_silu = torch.nn.SiLU()
    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensor_1 = torch.rand(shape, dtype=torch.bfloat16)
        torch_weight = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        ) @ torch_silu(torch_weight)

        torch_output_golden_1 = torch.nn.functional.gelu(
            torch.add(
                torch.tanh(torch.mul(torch.sub(torch_input_tensor_0, torch_input_tensor_1), torch_weight)),
                torch_input_tensor_1,
            )
        )

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(torch_input_tensor_0, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor_1 = ttnn.from_torch(torch_input_tensor_1, layout=ttnn.TILE_LAYOUT)
        ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT)

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)
        ttnn.copy_host_to_device_tensor(ttnn_weight, weight_dev)

        if blocking:
            ttnn.synchronize_device(device)
        logger.info("Execute Trace 0")
        # Execute trace
        ttnn.execute_trace(device, tid, cq_id=0, blocking=blocking)
        logger.info("Execute Trace 1")
        ttnn.execute_trace(device, tid_1, cq_id=0, blocking=blocking)

        logger.info("Read Back Trace 0 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)
        logger.info("Read Back Trace 1 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor_1)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden_1, pcc=0.99)

    # Release trace buffer once workload is complete
    ttnn.release_trace(device, tid)
    ttnn.release_trace(device, tid_1)


@skip_for_slow_dispatch()
@pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 at startup"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}, {"trace_region_size": 0}], indirect=True)
def test_trace_allocation_tracking_is_per_trace(device, expect_error):
    shape = (1, 1, 32, 32)
    trace_input = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Compile before either capture so this test isolates ordinary buffer accounting.
    warmup_output = ttnn.neg(trace_input)
    del warmup_output

    trace_a = ttnn.begin_trace_capture(device, cq_id=0)
    trace_a_output = ttnn.neg(trace_input)
    ttnn.end_trace_capture(device, trace_a, cq_id=0)

    allocation_between_captures = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
    )
    allocation_between_captures_id = allocation_between_captures.buffer_unique_id()

    trace_b = ttnn.begin_trace_capture(device, cq_id=0)
    trace_b_output = ttnn.neg(trace_input)
    ttnn.end_trace_capture(device, trace_b, cq_id=0)

    try:
        unsafe_for_trace_a = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_a)
        unsafe_for_trace_b = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_b)
        assert allocation_between_captures_id in unsafe_for_trace_a
        assert "trace_storage" not in unsafe_for_trace_a.values()
        assert not unsafe_for_trace_b

        # The allocation happened before trace_b was captured, so trace_b cannot corrupt it through allocator reuse.
        ttnn.execute_trace(device, trace_b, cq_id=0, blocking=True)

        # The same allocation happened after trace_a was captured and must still be rejected for trace_a.
        with expect_error(RuntimeError, rf"Buffer {allocation_between_captures_id}\b"):
            ttnn.execute_trace(device, trace_a, cq_id=0, blocking=True)
    finally:
        ttnn.release_trace(device, trace_a)
        ttnn.release_trace(device, trace_b)

    # Keep captured tensors alive through both checks.
    assert trace_a_output.is_allocated()
    assert trace_b_output.is_allocated()


@skip_for_slow_dispatch()
@pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 at startup"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_trace_allocation_tracking_acknowledgments_and_lifetime(device, expect_error):
    shape = (1, 1, 32, 32)
    trace_input = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    warmup_output = ttnn.neg(trace_input)
    del warmup_output

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_output = ttnn.neg(trace_input)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    try:
        # If the first query finds an unsafe tensor retained only by a Python
        # reference cycle, conditional GC must retire it and allow replay.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            cycle = []
            cyclic_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
            cyclic_tensor_id = cyclic_tensor.buffer_unique_id()
            cycle.extend((cycle, cyclic_tensor))
            del cyclic_tensor
            del cycle
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            assert cyclic_tensor_id not in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)
        finally:
            if gc_was_enabled:
                gc.enable()

        if trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS:
            # Each Python thread must drain only the allocations it dispatched,
            # even when both threads share the same device allocator.
            allocation_barrier = threading.Barrier(2)

            def allocate_and_drain_pending():
                tensor = ttnn._ttnn.operations.core.allocate_tensor_on_device(
                    ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
                )
                buffer_id = tensor.buffer_unique_id()
                allocation_barrier.wait()
                pending = set(ttnn._ttnn.operations.trace.drain_pending_traceback_ids())
                return tensor, buffer_id, pending

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(allocate_and_drain_pending)
                second = executor.submit(allocate_and_drain_pending)
                first_tensor, first_id, first_pending = first.result()
                second_tensor, second_id, second_pending = second.result()

            assert first_id in first_pending
            assert second_id not in first_pending
            assert second_id in second_pending
            assert first_id not in second_pending
            first_tensor.deallocate()
            second_tensor.deallocate()

        marked = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        marked_id = marked.buffer_unique_id()
        assert marked_id in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)
        assert trace_allocation_tracker.acknowledge_corruptible(marked) is None
        assert marked_id not in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)

        with trace_allocation_tracker.corruptible_allocation_scope(device):
            scoped = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        assert scoped.buffer_unique_id() not in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)

        temporary = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        temporary_id = temporary.buffer_unique_id()
        assert temporary_id in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)
        del temporary
        gc.collect()
        assert temporary_id not in ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)

        if trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS:
            from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

            # The aggregate diagnostics registry does not perform per-trace
            # stale-ID cleanup: deallocation must update it eagerly as well.
            assert temporary_id not in ttnn._ttnn.operations.trace.get_all_unsafe_tracked_ids()

            # An ordinary operation boundary, rather than trace replay, must
            # reconcile Python tracebacks with the eagerly updated C++ state.
            with trace_allocation_tracker.corruptible_allocation_scope(device):
                reconciliation_boundary = ttnn.neg(trace_input)
            assert temporary_id not in TraceAllocationTracker._tracebacks
            reconciliation_boundary.deallocate()

        # All three allowed cases remain safe to replay while the acknowledged tensors are still alive.
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

        fast_runtime_mode = ttnn.CONFIG.enable_fast_runtime_mode
        try:
            ttnn.CONFIG.enable_fast_runtime_mode = False
            unsafe_a = ttnn.neg(trace_input)
        finally:
            ttnn.CONFIG.enable_fast_runtime_mode = fast_runtime_mode
        unsafe_map = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)
        assert unsafe_map[unsafe_a.buffer_unique_id()].startswith("ttnn.")
        if trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS:
            # Exercise retirement before the pending ID is drained: bypass the
            # registered-operation wrapper, then deallocate the raw allocation.
            # Its thread-local pending ID is filtered against authoritative
            # allocator accounting at the next operation boundary.
            raw_temporary = ttnn._ttnn.operations.core.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
            )
            raw_temporary_id = raw_temporary.buffer_unique_id()
            del raw_temporary
            gc.collect()
            assert raw_temporary_id not in ttnn._ttnn.operations.trace.get_all_unsafe_tracked_ids()
            assert raw_temporary_id not in ttnn._ttnn.operations.trace.drain_pending_traceback_ids()

        unsafe_b = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        if trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS:
            assert raw_temporary_id not in TraceAllocationTracker._tracebacks
        with expect_error(RuntimeError, "Found 2 device buffer") as error:
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert f"Buffer {unsafe_a.buffer_unique_id()}" in str(error.value)
        assert f"Buffer {unsafe_b.buffer_unique_id()}" in str(error.value)
        assert f"Buffer {unsafe_a.buffer_unique_id()}Buffer" not in str(error.value)
        assert f"Buffer {unsafe_b.buffer_unique_id()}Buffer" not in str(error.value)
    finally:
        ttnn.release_trace(device, trace_id)

    assert trace_output.is_allocated()


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_trace_allocation_tracking_release_trace_is_idempotent(device):
    shape = (1, 1, 32, 32)
    input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    ttnn.neg(input_dev)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = ttnn.neg(input_dev)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # The keyword form must work identically with and without allocation tracking.
    ttnn.execute_trace(mesh_device=device, trace_id=trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    ttnn.release_trace(device, trace_id)

    assert output.is_allocated()
