# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only wiring test for the prefill runner's layer-completion sink.

No device, no model: we test that build_layer_completion_sink() computes a
globally-dense seq and pushes the right fields, and that a single-rank
router carries those pushes into the scheduler counter channel in order.
"""

import os
import time

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.pipelined_prefill import LayerCompletionQueue, LayerCompletionRouter

from models.demos.deepseek_v3_d_p.tt.runners.prefill_runner import build_layer_completion_sink


def _unlink(shm_name: str) -> None:
    try:
        os.unlink("/dev/shm" + shm_name)
    except FileNotFoundError:
        pass


def test_sink_pushes_dense_seq_and_router_orders_them():
    ring = "/tt_prefill_wiring_ring"
    sched = "/tt_prefill_wiring_sched"
    _unlink(ring)
    _unlink(sched)

    router = LayerCompletionRouter(
        rank=0, world_size=1, master_rank=0, ring_shm_name=ring, scheduler_channel_shm_name=sched
    )
    consumer = ttnn.InterProcessCounterChannel.connect(sched, connect_timeout_ms=5000)
    producer = LayerCompletionQueue.connect(ring, connect_timeout_ms=5000)

    num_layers = 4
    sink = build_layer_completion_sink(producer, source_rank=0, num_layers=num_layers)

    # Two requests, each completing layers 0..3 in order. The runtime binds the
    # request/chunk id per prefill() call, so the sink takes it as a second arg.
    for req in range(2):
        for layer in range(num_layers):
            sink(layer, req)

    deadline = time.time() + 5.0
    while router.processed < 8 and time.time() < deadline:
        time.sleep(0.005)
    assert router.processed == 8
    assert consumer.try_consume_all() == 8

    router.stop()
    consumer.shutdown()
    producer.shutdown()
