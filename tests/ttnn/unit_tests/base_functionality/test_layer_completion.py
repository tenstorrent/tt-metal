# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Same-process smoke tests for the pipelined-prefill layer_completion module.

The SHM ring + router are exercised from one process (world_size == 1,
master), so no MPI is needed. The router's master path drains the local
ring, reorders by seq, and injects into an InterProcessCounterChannel —
exactly the single-host behaviour, but routed through the new path.
"""

import os
import time

import pytest

import ttnn

# The _layer_completion extension is built only with this model's nanobind module (WITH_PYTHON_BINDINGS)
# and is absent from packaged/wheel runs — skip the module rather than error collection of the suite.
pytest.importorskip("models.demos.common.prefill.runners.pipelined_prefill")
from models.demos.common.prefill.runners.pipelined_prefill import LayerCompletionQueue, LayerCompletionRouter


def _unlink(shm_name: str) -> None:
    try:
        os.unlink("/dev/shm" + shm_name)
    except FileNotFoundError:
        pass


def test_queue_roundtrip():
    name = "/tt_lc_py_queue"
    _unlink(name)
    owner = LayerCompletionQueue.create(name)
    conn = LayerCompletionQueue.connect(name, connect_timeout_ms=5000)

    assert owner.try_push(seq=0, source_rank=1, layer_idx=2, request_id=3)
    popped = conn.try_pop()
    assert popped == (0, 1, 2, 3)
    assert conn.try_pop() is None  # empty
    owner.shutdown()


def test_router_single_rank_orders_into_counter_channel():
    ring = "/tt_lc_py_ring"
    sched = "/tt_lc_py_sched"
    _unlink(ring)
    _unlink(sched)

    router = LayerCompletionRouter(
        rank=0,
        world_size=1,
        master_rank=0,
        ring_shm_name=ring,
        scheduler_channel_shm_name=sched,
    )
    assert router.is_master

    consumer = ttnn.InterProcessCounterChannel.connect(sched, connect_timeout_ms=5000)
    producer = LayerCompletionQueue.connect(ring, connect_timeout_ms=5000)

    # Push 0..5 out of order; the router must inject them in order.
    for s in (0, 2, 1, 4, 3, 5):
        assert producer.try_push(seq=s, source_rank=0, layer_idx=s, request_id=0)

    deadline = time.time() + 5.0
    while router.processed < 6 and time.time() < deadline:
        time.sleep(0.005)
    assert router.processed == 6
    assert consumer.try_consume_all() == 6

    router.stop()
    consumer.shutdown()
    producer.shutdown()
