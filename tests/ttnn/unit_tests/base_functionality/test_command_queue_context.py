# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the TTNN-owned thread-local command queue selection (`ttnn.command_queue`).

The per-thread "current command queue id" stack lives in TTNN (ttnn/core/core.cpp); Metal has no implicit
queue state. These tests check that the stack nests/restores correctly, is per-thread, works without a
device, and that ops dispatched inside `with ttnn.command_queue(1):` really run on command queue 1.
"""

import threading

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _current_cq():
    return ttnn.get_current_command_queue_id_for_thread()


def test_command_queue_context_nests_and_restores_without_device():
    # The stack is owned by TTNN and does not depend on a device or MetalContext existing.
    assert _current_cq() == 0
    with ttnn.command_queue(1):
        assert _current_cq() == 1
        with ttnn.command_queue(0):
            assert _current_cq() == 0
            with ttnn.command_queue(1):
                assert _current_cq() == 1
            assert _current_cq() == 0
        assert _current_cq() == 1
    assert _current_cq() == 0


def test_command_queue_context_restores_on_exception(expect_error):
    assert _current_cq() == 0
    with expect_error(RuntimeError, "boom"):
        with ttnn.command_queue(1):
            assert _current_cq() == 1
            raise RuntimeError("boom")
    assert _current_cq() == 0


def test_command_queue_context_is_per_thread():
    observed = {}

    def worker():
        observed["before"] = _current_cq()
        with ttnn.command_queue(1):
            observed["inside"] = _current_cq()
        observed["after"] = _current_cq()

    with ttnn.command_queue(1):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        # The main thread's selection is untouched by the worker.
        assert _current_cq() == 1
    assert _current_cq() == 0

    # A new thread starts with an empty stack (cq 0) regardless of the main thread's selection.
    assert observed == {"before": 0, "inside": 1, "after": 0}


def test_pop_on_empty_stack_raises(expect_error):
    assert _current_cq() == 0
    with expect_error(RuntimeError, "Current command queue id stack is empty"):
        ttnn.decorators.pop_current_command_queue_id_for_thread()
    assert _current_cq() == 0


@pytest.mark.parametrize("device_params", [{"trace_region_size": 65536, "num_command_queues": 2}], indirect=True)
def test_command_queue_context_dispatches_op_to_selected_queue(device):
    """An op run inside `with ttnn.command_queue(1):` must be dispatched to cq 1.

    Trace capture is used as the observer: a trace captured on cq 1 only records work dispatched to cq 1.
    If the op silently went to cq 0, the trace would be empty and replaying it would not update the output.
    """
    shape = (1, 1, 32, 32)
    torch_input_0 = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_1 = torch.rand(shape, dtype=torch.bfloat16)

    input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input_0, layout=ttnn.TILE_LAYOUT), input_dev, cq_id=0)
    # The upload went out on cq 0 (non-blocking); make sure it has landed before cq 1 consumes it.
    ttnn.synchronize_device(device)

    # Compile the program outside of trace capture, on cq 1 via the context manager.
    with ttnn.command_queue(1):
        assert _current_cq() == 1
        warmup = ttnn.add(input_dev, 1.0)
    ttnn.synchronize_device(device)
    assert_with_pcc(torch_input_0 + 1.0, ttnn.to_torch(warmup), 0.9999)
    assert _current_cq() == 0

    # Capture on cq 1: begin/end_trace_capture without an explicit cq_id resolve to the thread's current cq.
    with ttnn.command_queue(1):
        tid = ttnn.begin_trace_capture(device)
        output = ttnn.add(input_dev, 1.0)
        ttnn.end_trace_capture(device, tid)
        assert _current_cq() == 1
    assert _current_cq() == 0

    # Replace the input, replay the trace on cq 1 and check the output follows the new input.
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input_1, layout=ttnn.TILE_LAYOUT), input_dev, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, tid, cq_id=1, blocking=True)
    ttnn.synchronize_device(device)
    assert_with_pcc(torch_input_1 + 1.0, ttnn.to_torch(output), 0.9999)

    ttnn.release_trace(device, tid)
    assert _current_cq() == 0


@pytest.mark.parametrize("device_params", [{"trace_region_size": 65536, "num_command_queues": 2}], indirect=True)
def test_explicit_queue_id_overrides_context(device):
    """An explicit queue_id keyword wins over the enclosing context, and the context is restored afterwards.

    The override is observed with a trace captured on cq 0: only work dispatched to cq 0 is recorded, so if
    `queue_id=0` were ignored (op dispatched to cq 1), the trace would be empty and replaying it would not
    update the output.
    """
    shape = (1, 1, 32, 32)
    torch_input_0 = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_1 = torch.rand(shape, dtype=torch.bfloat16)

    input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input_0, layout=ttnn.TILE_LAYOUT), input_dev, cq_id=0)
    # The upload went out on cq 0 (non-blocking); make sure it has landed before anything consumes it.
    ttnn.synchronize_device(device)

    # Compile the program outside of trace capture.
    warmup = ttnn.neg(input_dev)
    ttnn.synchronize_device(device)
    assert_with_pcc(-torch_input_0, ttnn.to_torch(warmup), 0.9999)

    with ttnn.command_queue(1):
        assert _current_cq() == 1
        out_ctx = ttnn.neg(input_dev)  # context: cq 1
        # Hand the device over from cq 1 to cq 0 before dispatching work on cq 0: a queue may only run
        # programs on a (sub)device it owns, and the event wait transfers ownership.
        event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, event)

        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out_explicit = ttnn.neg(input_dev, queue_id=0)  # explicit queue_id wins over the context
        ttnn.end_trace_capture(device, tid, cq_id=0)
        assert _current_cq() == 1
    assert _current_cq() == 0

    ttnn.synchronize_device(device)
    assert_with_pcc(-torch_input_0, ttnn.to_torch(out_ctx), 0.9999)

    # Replace the input and replay the cq 0 trace: the output only follows the new input if `neg` was
    # actually captured on cq 0.
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input_1, layout=ttnn.TILE_LAYOUT), input_dev, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    assert_with_pcc(-torch_input_1, ttnn.to_torch(out_explicit), 0.9999)

    ttnn.release_trace(device, tid)
    assert _current_cq() == 0


def _single_core(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))})


def test_selected_queue_out_of_range_raises_and_restores(device, expect_error):
    """On a single-CQ device, selecting cq 1 makes TTNN entry points fail cleanly, and the stack is restored."""
    shape = (1, 1, 32, 32)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    input_dev = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cq_id 1 is out of range"):
        with ttnn.command_queue(1):
            ttnn.neg(input_dev)
    assert _current_cq() == 0

    with expect_error(RuntimeError, "cq_id 1 is out of range"):
        with ttnn.command_queue(1):
            ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    assert _current_cq() == 0

    with expect_error(RuntimeError, "cq_id 1 is out of range"):
        ttnn.neg(input_dev, queue_id=1)
    assert _current_cq() == 0

    with expect_error(RuntimeError, "cq_id 1 is out of range"):
        with ttnn.command_queue(1):
            ttnn.create_global_semaphore(device, _single_core(0, 0), 5)
    with expect_error(RuntimeError, "cq_id 1 is out of range"):
        with ttnn.command_queue(1):
            ttnn.create_global_circular_buffer(
                device, [(ttnn.CoreCoord(0, 0), _single_core(0, 1))], 2048, ttnn.BufferType.L1
            )
    assert _current_cq() == 0
    ttnn.create_global_semaphore(device, _single_core(0, 0), 5)
    ttnn.create_global_circular_buffer(device, [(ttnn.CoreCoord(0, 0), _single_core(0, 1))], 2048, ttnn.BufferType.L1)

    assert_with_pcc(-torch_input, ttnn.to_torch(ttnn.neg(input_dev)), 0.9999)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_io_on_cq1_compute_on_cq0_pipeline(device):
    """The usual two-queue pattern (I/O on cq 1, compute on cq 0, events in between) driven by the context."""
    shape = (1, 1, 64, 64)
    input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    op_event = ttnn.record_event(device, 0)

    for i in range(8):
        torch_input = torch.rand(shape, dtype=torch.bfloat16)
        with ttnn.command_queue(1):
            ttnn.wait_for_event(mesh_event=op_event)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT), input_dev)
            write_event = ttnn.record_event(device)

        ttnn.wait_for_event(0, write_event)
        output = ttnn.add(input_dev, i / 8)
        op_event = ttnn.record_event(device, 0)

        with ttnn.command_queue(1):
            ttnn.wait_for_event(mesh_event=op_event)
            result = ttnn.to_torch(output)
        assert_with_pcc(torch_input + i / 8, result, 0.999)

    ttnn.synchronize_device(device)
    assert _current_cq() == 0


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_concurrent_threads_on_different_queues(device):
    """A worker thread doing I/O under `ttnn.command_queue(1)` runs concurrently with compute + I/O on cq 0 in the
    main thread; each thread's selection stays private and every result is correct."""
    shape = (1, 1, 64, 64)
    iterations = 16
    worker_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    main_input = torch.rand(shape, dtype=torch.bfloat16)
    main_dev = ttnn.from_torch(main_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)

    errors = []

    def worker():
        try:
            with ttnn.command_queue(1):
                for _ in range(iterations):
                    assert _current_cq() == 1
                    data = torch.rand(shape, dtype=torch.bfloat16)
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT), worker_dev)
                    assert torch.equal(ttnn.to_torch(worker_dev), data)
            assert _current_cq() == 0
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    for i in range(iterations):
        assert _current_cq() == 0
        result = ttnn.to_torch(ttnn.add(main_dev, i / iterations))
        assert_with_pcc(main_input + i / iterations, result, 0.999)
    thread.join()
    ttnn.synchronize_device(device)

    assert not errors, errors
    assert _current_cq() == 0
