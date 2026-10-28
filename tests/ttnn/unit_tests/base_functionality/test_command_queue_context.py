# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import threading

import pytest
import torch
import ttnn


def test_command_queue_nesting_and_restore(device):
    assert ttnn.get_current_command_queue_id_for_thread() == 0
    with ttnn.command_queue(1):
        assert ttnn.get_current_command_queue_id_for_thread() == 1
        with ttnn.command_queue(0):
            assert ttnn.get_current_command_queue_id_for_thread() == 0
        assert ttnn.get_current_command_queue_id_for_thread() == 1
    assert ttnn.get_current_command_queue_id_for_thread() == 0


def test_command_queue_restored_after_exception(device, expect_error):
    with expect_error(ValueError, "boom"):
        with ttnn.command_queue(1):
            raise ValueError("boom")
    assert ttnn.get_current_command_queue_id_for_thread() == 0


def test_command_queue_is_per_thread(device):
    seen = []
    with ttnn.command_queue(1):
        t = threading.Thread(target=lambda: seen.append(ttnn.get_current_command_queue_id_for_thread()))
        t.start()
        t.join()
        assert ttnn.get_current_command_queue_id_for_thread() == 1
    assert seen == [0]


def test_command_queue_none_rejected(device, expect_error):
    with expect_error(ValueError, "cq_id cannot be None"):
        with ttnn.command_queue(None):
            pass


@pytest.mark.skipif("TT_METAL_SLOW_DISPATCH_MODE" in os.environ, reason="Queue lookup only happens with fast dispatch")
def test_selected_queue_out_of_range_raises_and_restores(device, expect_error):
    """The default device has a single command queue, so every queue-dependent TTNN entry point must route
    through the selected queue (1) and fail the lookup, rather than silently using queue 0."""
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mapping = [(ttnn.CoreCoord(1, 1), ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 2))}))]
    host = torch.rand(1, 1, 32, 32, dtype=torch.bfloat16)
    # Borrow the address of a regular semaphore (created on queue 0) as caller-owned backing memory.
    backing = ttnn.create_global_semaphore(device, cores, 0)
    backing_address = ttnn.get_global_semaphore_address(backing)

    with ttnn.command_queue(1):
        with expect_error(RuntimeError, "out of range"):
            ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with expect_error(RuntimeError, "out of range"):
            ttnn.create_global_semaphore(device, cores, 0)
        with expect_error(RuntimeError, "out of range"):
            ttnn.create_global_circular_buffer(device, mapping, 3200)
        with expect_error(RuntimeError, "out of range"):
            ttnn.experimental_create_global_semaphore_at_address(device, cores, 1, ttnn.BufferType.L1, backing_address)

    assert ttnn.get_current_command_queue_id_for_thread() == 0
    # Back on queue 0 everything works again.
    ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
