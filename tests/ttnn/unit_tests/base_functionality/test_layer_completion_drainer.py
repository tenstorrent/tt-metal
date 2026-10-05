# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from collections import deque

import pytest

from models.demos.common.prefill.runners.layer_completion_drainer import (
    BackgroundCompletionDrain,
    Completion,
    LayerCompletionDrainer,
    current_protocol,
)

NUM_LAYERS = 4


class FakeRing:
    def __init__(self, messages=()):
        self._q = deque(messages)

    def try_pop(self):
        return self._q.popleft() if self._q else None


def msg(request_id, layer_start, layer_end, *, seq=None, slot_id=0, pos_start=0, pos_end=128, rank=0, host_ts_ns=0):
    if seq is None:
        seq = request_id * NUM_LAYERS + layer_start
    return (seq, rank, request_id, slot_id, pos_start, pos_end, layer_start, layer_end, host_ts_ns)


def per_layer(request_id, layers=range(NUM_LAYERS), **kw):
    return [msg(request_id, l, l + 1, **kw) for l in layers]


def test_single_request_per_layer_completes():
    completed = []
    d = LayerCompletionDrainer(
        FakeRing(per_layer(0)),
        num_layers=NUM_LAYERS,
        on_request_complete=lambda rid, cov: completed.append(rid),
    )
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert completed == [0]
    assert d.processed == NUM_LAYERS
    assert d.requests[0].is_complete(NUM_LAYERS)


def test_interleaved_requests_advance_independently():
    completed = []
    ring = FakeRing(
        per_layer(0, layers=[0, 1]) + per_layer(1, layers=[0, 1, 2, 3], slot_id=1) + per_layer(0, layers=[2, 3])
    )
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, on_request_complete=lambda rid, cov: completed.append(rid))
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    assert completed == [1, 0]


def test_out_of_order_spans_tile():
    ring = FakeRing([msg(0, 2, 4), msg(0, 0, 2)])
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS


def test_wide_span_counts_full_width_as_one_message():
    ring = FakeRing([msg(0, 0, NUM_LAYERS)])
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert d.processed == 1


def test_blocked_message_side_queues_without_stalling_others():
    seen = []
    blocked_once = {0: True}

    def can_process(c):
        if c.request_id == 0 and blocked_once[0]:
            return False
        return True

    def on_completion(c):
        seen.append((c.request_id, c.layer_start))
        if c.request_id == 1:
            blocked_once[0] = False

    ring = FakeRing([msg(0, 0, 2), *per_layer(1, slot_id=1), msg(0, 2, 4)])
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, can_process=can_process, on_completion=on_completion)
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    r1_done = max(i for i, (rid, _) in enumerate(seen) if rid == 1)
    r0_first = min(i for i, (rid, _) in enumerate(seen) if rid == 0)
    assert r1_done < len(seen) - 1 or seen[r1_done:] == []
    assert seen[r0_first][0] == 0 and r0_first > 0
    assert d.side_queued == 1


def test_side_queue_retried_in_request_order():
    order = []
    ready = {"go": False}
    d = LayerCompletionDrainer(
        FakeRing([msg(5, 0, 2), msg(2, 0, 2), msg(5, 2, 4), msg(2, 2, 4)]),
        num_layers=NUM_LAYERS,
        can_process=lambda c: ready["go"],
        on_completion=lambda c: order.append(c.request_id),
    )
    while d.step():
        pass
    assert d.side_queued == 4 and d.processed == 0
    ready["go"] = True
    assert d.step() is True
    assert order == [2, 2, 5, 5]


def test_overlap_span_raises():
    d = LayerCompletionDrainer(FakeRing([msg(0, 0, 3), msg(0, 2, 4)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        while d.step():
            pass


def test_out_of_bounds_span_raises():
    d = LayerCompletionDrainer(FakeRing([msg(0, 2, NUM_LAYERS + 1)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="out of bounds"):  # allow-pytest.raises: host-only, no device error
        d.step()


def test_inconsistent_identity_raises():
    d = LayerCompletionDrainer(
        FakeRing([msg(0, 0, 2, slot_id=0, pos_start=0, pos_end=128), msg(0, 2, 4, slot_id=1)]),
        num_layers=NUM_LAYERS,
    )
    with pytest.raises(ValueError, match="inconsistent identity"):  # allow-pytest.raises: host-only, no device error
        while d.step():
            pass


def test_expectation_hook_fires_once_per_request():
    registered = {}

    def on_first(request_id, completion, coverage):
        coverage.expectation = ("tile", 0, NUM_LAYERS, completion.pos_start, completion.pos_end)
        registered[request_id] = coverage.expectation

    ring = FakeRing(per_layer(0, layers=[0, 1]) + per_layer(1, layers=[0], slot_id=1) + per_layer(0, layers=[2, 3]))
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, on_first_completion=on_first)
    while d.step():
        pass
    assert set(registered) == {0, 1}
    assert registered[1] == ("tile", 0, NUM_LAYERS, 0, 128)


def test_finish_raises_on_stranded_side_queue():
    d = LayerCompletionDrainer(FakeRing([msg(0, 0, 2)]), num_layers=NUM_LAYERS, can_process=lambda c: False)
    while d.step():
        pass
    with pytest.raises(RuntimeError, match="never-actionable"):  # allow-pytest.raises: host-only, no device error
        d.finish()


def test_drain_blocking_timeout_reports_coverage_snapshot():
    d = LayerCompletionDrainer(FakeRing(per_layer(0, layers=[0, 1])), num_layers=NUM_LAYERS)
    with pytest.raises(TimeoutError, match="req 0: 2/4 layers"):  # allow-pytest.raises: host-only, no device error
        d.drain_blocking(NUM_LAYERS, timeout_s=0.2)


def test_current_protocol(monkeypatch):
    monkeypatch.delenv("PREFILL_LAYER_COMPLETION_PROTOCOL", raising=False)
    assert current_protocol() == 1
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    assert current_protocol() == 2
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "banana")
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only, no device error
        current_protocol()


class FakeCounterChannel:
    def __init__(self, count: int):
        self._count = count

    def try_consume_all(self) -> int:
        n, self._count = self._count, 0
        return n


class BoundedRing:
    def __init__(self, capacity: int):
        self._q = deque()
        self._capacity = capacity
        self._lock = threading.Lock()

    def try_push(self, m) -> bool:
        with self._lock:
            if len(self._q) >= self._capacity:
                return False
            self._q.append(m)
            return True

    def try_pop(self):
        with self._lock:
            return self._q.popleft() if self._q else None


def test_background_drain_v1_counts_acks(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "1")
    drain = BackgroundCompletionDrain(FakeCounterChannel(3 * NUM_LAYERS))
    assert drain.wait(3 * NUM_LAYERS, timeout_s=5) == 3 * NUM_LAYERS
    drain.close()


def test_background_drain_v2_ring(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    monkeypatch.setenv("PREFILL_NUM_LAYERS", str(NUM_LAYERS))
    ring = FakeRing(per_layer(0, layers=[0, 1]) + per_layer(1, slot_id=1) + per_layer(0, layers=[2, 3]))
    drain = BackgroundCompletionDrain(ring)
    assert drain.wait(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    drain.close()


def test_background_drain_keeps_bounded_ring_moving(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    monkeypatch.setenv("PREFILL_NUM_LAYERS", str(NUM_LAYERS))
    ring = BoundedRing(capacity=2)
    drain = BackgroundCompletionDrain(ring)
    num_requests = 50
    for request_id in range(num_requests):
        for m in per_layer(request_id):
            deadline = time.perf_counter() + 5
            while not ring.try_push(m):
                assert time.perf_counter() < deadline, "ring stayed full: nothing drained it"
                time.sleep(0.001)
    assert drain.wait(num_requests * NUM_LAYERS, timeout_s=5) == num_requests * NUM_LAYERS
    drain.close()


def test_background_drain_cumulative_across_waits(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    monkeypatch.setenv("PREFILL_NUM_LAYERS", str(NUM_LAYERS))
    ring = BoundedRing(capacity=64)
    drain = BackgroundCompletionDrain(ring)
    for m in per_layer(0):
        ring.try_push(m)
    assert drain.wait(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    for m in per_layer(1, slot_id=1):
        ring.try_push(m)
    assert drain.wait(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    drain.close()


def test_background_drain_surfaces_drainer_error(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    monkeypatch.setenv("PREFILL_NUM_LAYERS", str(NUM_LAYERS))
    drain = BackgroundCompletionDrain(FakeRing([msg(0, 0, 3), msg(0, 2, 4)]))
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        drain.wait(NUM_LAYERS, timeout_s=5)
    drain.close()


def test_background_drain_none_channel_is_noop(monkeypatch):
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    drain = BackgroundCompletionDrain(None)
    assert drain.wait(NUM_LAYERS, timeout_s=1) == 0
    drain.close()
