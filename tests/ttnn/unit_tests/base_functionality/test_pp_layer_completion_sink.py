# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipelined-prefill layer-completion sink.

The seq-math tests use a fake producer and need no device or ttnn build. The
end-to-end test exercises the real ttnn.layer_completion queue + router (single
rank, world_size=1, no MPI) and is skipped until that C++ transport is built in.
"""

import os
import time

import pytest

from models.demos.deepseek_v3_d_p.tt.runners.layer_completion_sink import build_layer_completion_sink


class _FakeProducer:
    """Records try_push calls; full() forces the ring-full path."""

    def __init__(self, full=False):
        self.pushed = []
        self._full = full

    def try_push(self, *, seq, source_rank, layer_idx, request_id):
        if self._full:
            return False
        self.pushed.append((seq, source_rank, layer_idx, request_id))
        return True


def test_seq_is_dense_and_uses_global_layer_count():
    # 2 ranks splitting 4 global layers: rank 0 -> [0,1], rank 1 -> [2,3].
    num_layers = 4
    chunk = [0]
    p0 = _FakeProducer()
    p1 = _FakeProducer()
    sink0 = build_layer_completion_sink(p0, source_rank=0, num_layers=num_layers, get_request_id=lambda: chunk[0])
    sink1 = build_layer_completion_sink(p1, source_rank=1, num_layers=num_layers, get_request_id=lambda: chunk[0])

    # Two chunks; each rank fires completions for its global layer slice.
    for c in (0, 1):
        chunk[0] = c
        sink0(0)
        sink0(1)
        sink1(2)
        sink1(3)

    seqs = sorted(s for (s, *_rest) in (p0.pushed + p1.pushed))
    # Dense, gap-free over both chunks: 0..7.
    assert seqs == list(range(8))
    # Payload carries the global layer index + source rank.
    assert (0, 0, 0, 0) in p0.pushed  # chunk 0, rank 0, layer 0
    assert (6, 1, 2, 1) in p1.pushed  # chunk 1, rank 1, layer 2 -> 1*4+2 = 6


def test_full_ring_raises_rather_than_dropping():
    sink = build_layer_completion_sink(_FakeProducer(full=True), source_rank=0, num_layers=4, get_request_id=lambda: 0)
    with pytest.raises(RuntimeError, match="ring full"):
        sink(0)


def _unlink(shm_name: str) -> None:
    try:
        os.unlink("/dev/shm" + shm_name)
    except FileNotFoundError:
        pass


def test_sink_through_real_router_single_rank():
    """End-to-end with the real transport (world_size=1, master, no MPI).

    Skipped until ttnn.layer_completion (the Step-1 C++ transport) is built into this tree.
    """
    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "layer_completion"):
        pytest.skip("ttnn.layer_completion not built (Step-1 transport not present)")

    ring = "/tt_pp_sink_ring"
    sched = "/tt_pp_sink_sched"
    _unlink(ring)
    _unlink(sched)

    router = ttnn.layer_completion.LayerCompletionRouter(
        rank=0, world_size=1, master_rank=0, ring_shm_name=ring, scheduler_channel_shm_name=sched
    )
    consumer = ttnn.InterProcessCounterChannel.connect(sched, connect_timeout_ms=5000)
    producer = ttnn.layer_completion.LayerCompletionQueue.connect(ring, connect_timeout_ms=5000)

    num_layers = 4
    chunk = [0]
    sink = build_layer_completion_sink(producer, source_rank=0, num_layers=num_layers, get_request_id=lambda: chunk[0])
    for c in (0, 1):
        chunk[0] = c
        for layer in range(num_layers):
            sink(layer)

    deadline = time.time() + 5.0
    while router.processed < 8 and time.time() < deadline:
        time.sleep(0.005)
    assert router.processed == 8
    assert consumer.try_consume_all() == 8

    router.stop()
    consumer.shutdown()
    producer.shutdown()
