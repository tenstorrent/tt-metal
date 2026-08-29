# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v2 layer-completion drainer (work-conserving consumer).

Host-only: a list-backed fake ring, no ttnn, no device. Covers the protocol
semantics: per-request coverage, span summing (a wide span is one message),
side-queue work conservation, producer-bug detection, the expectation hook,
and the teardown invariant.
"""

from collections import deque

import pytest

from models.demos.common.prefill.runners.layer_completion_drainer import (
    Completion,
    LayerCompletionDrainer,
    current_protocol,
)

NUM_LAYERS = 4  # small model for tests


class FakeRing:
    def __init__(self, messages=()):
        self._q = deque(messages)

    def try_pop(self):
        return self._q.popleft() if self._q else None


def msg(request_id, layer_start, layer_end, *, seq=None, slot_id=0, pos_start=0, pos_end=128, rank=0):
    """A v2 wire-order tuple, as LayerCompletionQueueV2.try_pop() returns."""
    if seq is None:
        seq = request_id * NUM_LAYERS + layer_start
    return (seq, rank, request_id, slot_id, pos_start, pos_end, layer_start, layer_end)


def per_layer(request_id, layers=range(NUM_LAYERS), **kw):
    return [msg(request_id, l, l + 1, **kw) for l in layers]


def test_single_request_per_layer_completes():
    completed = []
    d = LayerCompletionDrainer(
        FakeRing(per_layer(0)), num_layers=NUM_LAYERS,
        on_request_complete=lambda rid, cov: completed.append(rid),
    )
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert completed == [0]
    assert d.processed == NUM_LAYERS  # per-layer: one message per layer
    assert d.requests[0].is_complete(NUM_LAYERS)


def test_interleaved_requests_advance_independently():
    completed = []
    ring = FakeRing(
        per_layer(0, layers=[0, 1]) + per_layer(1, layers=[0, 1, 2, 3], slot_id=1) + per_layer(0, layers=[2, 3])
    )
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS,
                               on_request_complete=lambda rid, cov: completed.append(rid))
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    # request 1 finished before request 0 — no head-of-line coupling
    assert completed == [1, 0]


def test_out_of_order_spans_tile():
    ring = FakeRing([msg(0, 2, 4), msg(0, 0, 2)])  # two halves, reversed
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS


def test_wide_span_counts_full_width_as_one_message():
    ring = FakeRing([msg(0, 0, NUM_LAYERS)])  # whole stage in one message
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert d.processed == 1


def test_blocked_message_side_queues_without_stalling_others():
    """Request 0's completion is blocked; request 1 must still advance (no HoL)."""
    seen = []
    blocked_once = {0: True}  # request 0 blocked on first sight, ready on retry

    def can_process(c):
        if c.request_id == 0 and blocked_once[0]:
            return False
        return True

    def on_completion(c):
        seen.append((c.request_id, c.layer_start))
        if c.request_id == 1:
            blocked_once[0] = False  # embedder state change unblocks request 0

    ring = FakeRing([msg(0, 0, 2), *per_layer(1, slot_id=1), msg(0, 2, 4)])
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, can_process=can_process, on_completion=on_completion)
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    # request 1 fully processed BEFORE request 0's second half, despite arriving later
    r1_done = max(i for i, (rid, _) in enumerate(seen) if rid == 1)
    r0_first = min(i for i, (rid, _) in enumerate(seen) if rid == 0)
    assert r1_done < len(seen) - 1 or seen[r1_done:] == []
    assert seen[r0_first][0] == 0 and r0_first > 0  # request 0 processed after some request-1 work
    assert d.side_queued == 1


def test_side_queue_retried_in_request_order():
    """Two blocked requests: when both become ready, retries run oldest-request-first."""
    order = []
    ready = {"go": False}
    # Each request contributes two half-spans: [0,2) and [2,4) — 4 messages total.
    d = LayerCompletionDrainer(
        FakeRing([msg(5, 0, 2), msg(2, 0, 2), msg(5, 2, 4), msg(2, 2, 4)]),
        num_layers=NUM_LAYERS,
        can_process=lambda c: ready["go"],
        on_completion=lambda c: order.append(c.request_id),
    )
    # Nothing actionable: drain would idle — drive steps manually until everything is side-queued.
    while d.step():
        pass
    assert d.side_queued == 4 and d.processed == 0
    ready["go"] = True
    assert d.step() is True  # ring empty → retry pass
    # request 2's messages processed before request 5's
    assert order == [2, 2, 5, 5]


def test_overlap_span_raises():
    d = LayerCompletionDrainer(FakeRing([msg(0, 0, 3), msg(0, 2, 4)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="overlaps"):
        while d.step():
            pass


def test_out_of_bounds_span_raises():
    d = LayerCompletionDrainer(FakeRing([msg(0, 2, NUM_LAYERS + 1)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="out of bounds"):
        d.step()


def test_inconsistent_identity_raises():
    d = LayerCompletionDrainer(
        FakeRing([msg(0, 0, 2, slot_id=0, pos_start=0, pos_end=128), msg(0, 2, 4, slot_id=1)]),
        num_layers=NUM_LAYERS,
    )
    with pytest.raises(ValueError, match="inconsistent identity"):
        while d.step():
            pass


def test_expectation_hook_fires_once_per_request():
    registered = {}

    def on_first(request_id, completion, coverage):
        # The pipelined-prefill rule shape: every layer eventually spans for this
        # request's position range. Recorded, not enforced (future error detection).
        coverage.expectation = ("tile", 0, NUM_LAYERS, completion.pos_start, completion.pos_end)
        registered[request_id] = coverage.expectation

    ring = FakeRing(per_layer(0, layers=[0, 1]) + per_layer(1, layers=[0], slot_id=1) + per_layer(0, layers=[2, 3]))
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, on_first_completion=on_first)
    while d.step():
        pass
    assert set(registered) == {0, 1}
    assert registered[1] == ("tile", 0, NUM_LAYERS, 0, 128)


def test_finish_raises_on_stranded_side_queue():
    d = LayerCompletionDrainer(
        FakeRing([msg(0, 0, 2)]), num_layers=NUM_LAYERS, can_process=lambda c: False  # never actionable
    )
    while d.step():
        pass
    with pytest.raises(RuntimeError, match="never-actionable"):
        d.finish()


def test_drain_blocking_timeout_reports_coverage_snapshot():
    d = LayerCompletionDrainer(FakeRing(per_layer(0, layers=[0, 1])), num_layers=NUM_LAYERS)
    with pytest.raises(TimeoutError, match="req 0: 2/4 layers"):
        d.drain_blocking(NUM_LAYERS, timeout_s=0.2)


def test_current_protocol(monkeypatch):
    monkeypatch.delenv("PREFILL_LAYER_COMPLETION_PROTOCOL", raising=False)
    assert current_protocol() == 1
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    assert current_protocol() == 2
    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "banana")
    with pytest.raises(ValueError):
        current_protocol()


class FakeCounterChannel:
    """v1 scheduler channel stand-in: try_consume_all() destructively drains a count."""

    def __init__(self, count: int):
        self._count = count

    def try_consume_all(self) -> int:
        n, self._count = self._count, 0
        return n


def test_drain_layer_completions_dispatches_v1_count(monkeypatch):
    """v1: the dispatcher drains the bare counter channel (no ring, no coverage)."""
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "1")
    channel = FakeCounterChannel(3 * NUM_LAYERS)
    assert lcd.drain_layer_completions(channel, 3 * NUM_LAYERS, timeout_s=5) == 3 * NUM_LAYERS


def test_drain_layer_completions_dispatches_v2_ring(monkeypatch):
    """v2: the dispatcher routes the ring through the work-conserving drainer."""
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    monkeypatch.setenv("PREFILL_NUM_LAYERS", str(NUM_LAYERS))
    ring = FakeRing(per_layer(0, layers=[0, 1]) + per_layer(1, slot_id=1) + per_layer(0, layers=[2, 3]))
    assert lcd.drain_layer_completions(ring, 2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS


def test_drain_layer_completions_none_channel_is_noop(monkeypatch):
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    assert lcd.drain_layer_completions(None, NUM_LAYERS, timeout_s=1) == 0
