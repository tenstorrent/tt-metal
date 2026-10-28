# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layer-completion ring reader (work-conserving consumer).

Host-only: a list-backed fake ring, no ttnn, no device. Covers per-chunk coverage keyed on
identity, eviction and reuse, span summing, side-queue work conservation, producer-bug
detection and the teardown invariant.
"""

import threading
import time
from collections import deque

import pytest

from models.demos.common.prefill.runners.layer_completion_ring_reader import (
    Completion,
    LayerCompletionDrain,
    LayerCompletionRingReader,
    scheduler_shm_name,
    start_layer_completion_drain,
    wait_for_layer_completions,
)

NUM_LAYERS = 4  # small model for tests


class FakeRing:
    def __init__(self, messages=()):
        self._q = deque(messages)

    def try_pop(self):
        return self._q.popleft() if self._q else None


def msg(layer_start, layer_end, *, slot_id=0, pos_start=0, pos_end=128, host_timestamp_ns=0):
    """A wire-order tuple, as LayerCompletionQueue.try_pop() returns."""
    return (slot_id, pos_start, pos_end, layer_start, layer_end, host_timestamp_ns)


def per_layer(layers=range(NUM_LAYERS), **kw):
    return [msg(l, l + 1, **kw) for l in layers]


def test_single_chunk_per_layer_completes_and_is_evicted():
    completed = []
    d = LayerCompletionRingReader(
        FakeRing(per_layer()),
        num_layers=NUM_LAYERS,
        on_chunk_complete=lambda key, cov: completed.append(key),
    )
    assert d.wait_for(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert completed == [(0, 0, 128)]
    assert d.processed == NUM_LAYERS  # per-layer: one message per layer
    assert d.chunks == {}  # evicted on completion
    assert d.completed_chunks == 1


def test_interleaved_chunks_advance_independently():
    completed = []
    ring = FakeRing(per_layer(layers=[0, 1]) + per_layer(layers=[0, 1, 2, 3], slot_id=1) + per_layer(layers=[2, 3]))
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS, on_chunk_complete=lambda key, cov: completed.append(key))
    assert d.wait_for(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    # chunk on slot 1 finished before the one on slot 0: no head-of-line coupling
    assert completed == [(1, 0, 128), (0, 0, 128)]


def test_same_identity_can_come_round_again_after_completion():
    """Slot reuse and multi-turn resumes repeat (slot, pos_start, pos_end); eviction makes that legal."""
    ring = FakeRing(per_layer() + per_layer())  # both on slot 0, positions [0,128)
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS)
    assert d.wait_for(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    assert d.completed_chunks == 2 and d.chunks == {}


def test_out_of_order_spans_tile():
    ring = FakeRing([msg(2, 4), msg(0, 2)])  # two halves, reversed
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS)
    assert d.wait_for(NUM_LAYERS, timeout_s=5) == NUM_LAYERS


def test_wide_span_counts_full_width_as_one_message():
    ring = FakeRing([msg(0, NUM_LAYERS)])  # whole stage in one message
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS)
    assert d.wait_for(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert d.processed == 1


def test_acks_past_the_model_layers_fit_a_widened_bound():
    """MTP and DFlash ack rows past the last model layer; the bound is the widened count."""
    extra = 2
    ring = FakeRing(per_layer(layers=range(NUM_LAYERS + extra)))
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS + extra)
    assert d.wait_for(NUM_LAYERS + extra, timeout_s=5) == NUM_LAYERS + extra
    assert d.completed_chunks == 1


def test_hybrid_chunk_retires_on_its_sparse_coverage_and_comes_round_again():
    """A hybrid stack acks only its KV-writing layers: coverage is chunk_layers, spans stay in [0, num_layers)."""
    kv_layers = [3, 7, 11, 15, 19, 23]
    ring = FakeRing(per_layer(layers=kv_layers) + per_layer(layers=kv_layers))
    d = LayerCompletionRingReader(ring, num_layers=24, chunk_layers=len(kv_layers))
    assert d.wait_for(2 * len(kv_layers), timeout_s=5) == 2 * len(kv_layers)
    assert d.completed_chunks == 2 and d.chunks == {}


def test_chunk_layers_past_the_bound_is_rejected():
    with pytest.raises(ValueError, match="chunk_layers"):  # allow-pytest.raises: host-only, no device error
        LayerCompletionRingReader(FakeRing(), num_layers=NUM_LAYERS, chunk_layers=NUM_LAYERS + 1)


def test_blocked_message_side_queues_without_stalling_others():
    """Slot 0's completion is blocked; slot 1 must still advance (no HoL)."""
    seen = []
    blocked_once = {0: True}  # slot 0 blocked on first sight, ready on retry

    def can_process(c):
        if c.slot_id == 0 and blocked_once[0]:
            return False
        return True

    def on_completion(c):
        seen.append((c.slot_id, c.layer_start))
        if c.slot_id == 1:
            blocked_once[0] = False  # embedder state change unblocks slot 0

    ring = FakeRing([msg(0, 2), *per_layer(slot_id=1), msg(2, 4)])
    d = LayerCompletionRingReader(ring, num_layers=NUM_LAYERS, can_process=can_process, on_completion=on_completion)
    assert d.wait_for(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    assert seen[0][0] == 1  # slot 1 work ran before any slot 0 work
    assert d.side_queued == 1


def test_side_queue_retried_oldest_chunk_first():
    order = []
    ready = {"go": False}
    d = LayerCompletionRingReader(
        FakeRing([msg(0, 2, slot_id=5), msg(0, 2, slot_id=2), msg(2, 4, slot_id=5), msg(2, 4, slot_id=2)]),
        num_layers=NUM_LAYERS,
        can_process=lambda c: ready["go"],
        on_completion=lambda c: order.append(c.slot_id),
    )
    while d.step():
        pass
    assert d.side_queued == 4 and d.processed == 0
    ready["go"] = True
    assert d.step() is True  # ring empty -> retry pass
    assert order == [5, 5, 2, 2]  # the chunk seen first is retried first


def test_overlap_span_raises():
    d = LayerCompletionRingReader(FakeRing([msg(0, 3), msg(2, 4)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        while d.step():
            pass


def test_out_of_bounds_span_raises():
    d = LayerCompletionRingReader(FakeRing([msg(2, NUM_LAYERS + 1)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="out of bounds"):  # allow-pytest.raises: host-only, no device error
        d.step()


def test_finish_raises_on_stranded_side_queue():
    d = LayerCompletionRingReader(
        FakeRing([msg(0, 2)]), num_layers=NUM_LAYERS, can_process=lambda c: False  # never actionable
    )
    while d.step():
        pass
    with pytest.raises(RuntimeError, match="never-actionable"):  # allow-pytest.raises: host-only, no device error
        d.finish()


def test_wait_for_timeout_reports_open_chunks():
    d = LayerCompletionRingReader(FakeRing(per_layer(layers=[0, 1])), num_layers=NUM_LAYERS)
    with pytest.raises(TimeoutError, match="2/4 layers"):  # allow-pytest.raises: host-only, no device error
        d.wait_for(NUM_LAYERS, timeout_s=0.2)


class BoundedRing:
    """Thread-safe ring with the runner's backpressure: push blocks while `capacity` messages are queued."""

    def __init__(self, capacity):
        self._q = deque()
        self._capacity = capacity
        self._lock = threading.Lock()

    def push(self, m, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if len(self._q) < self._capacity:
                    self._q.append(m)
                    return
            if time.monotonic() > deadline:
                raise TimeoutError("ring stayed full: nothing drained it")
            time.sleep(0.001)

    def try_pop(self):
        with self._lock:
            return self._q.popleft() if self._q else None


def test_drain_keeps_a_bounded_ring_moving_while_the_caller_produces():
    """More completions than the ring holds: the producer only finishes because the drain runs alongside it."""
    ring = BoundedRing(capacity=2)
    drain = start_layer_completion_drain(ring, num_layers=NUM_LAYERS, chunk_layers=NUM_LAYERS)
    for slot in range(5):
        for m in per_layer(slot_id=slot):
            ring.push(m)
    assert wait_for_layer_completions(drain, 5 * NUM_LAYERS, timeout_s=5) == 5 * NUM_LAYERS
    assert drain.stop() == 5 * NUM_LAYERS
    assert drain.reader.completed_chunks == 5


def test_drain_surfaces_a_producer_bug_on_the_callers_thread():
    drain = LayerCompletionDrain(FakeRing([msg(0, 3), msg(2, 4)]), num_layers=NUM_LAYERS).start()
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        drain.wait_for(NUM_LAYERS, timeout_s=5)
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        drain.stop()


def test_no_channel_means_no_drain():
    assert start_layer_completion_drain(None, num_layers=NUM_LAYERS, chunk_layers=NUM_LAYERS) is None
    assert wait_for_layer_completions(None, NUM_LAYERS) == 0


def test_scheduler_segment_name():
    """Frozen: tt-d-gen's ack_shm_name configs attach to this name."""
    assert scheduler_shm_name("svc") == "/tt_prefill_layer_completions_svc"
