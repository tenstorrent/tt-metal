# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v2 layer-completion drainer (work-conserving consumer).

Host-only: a list-backed fake ring, no ttnn, no device. Covers per-chunk coverage keyed on
identity, eviction and reuse, span summing, side-queue work conservation, producer-bug
detection and the teardown invariant.
"""

from collections import deque

import pytest

from models.demos.common.prefill.runners.layer_completion_drainer import (
    Completion,
    LayerCompletionDrainer,
    current_protocol,
    scheduler_shm_name,
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


def test_single_chunk_per_layer_completes_and_is_evicted():
    completed = []
    d = LayerCompletionDrainer(
        FakeRing(per_layer(0)),
        num_layers=NUM_LAYERS,
        on_chunk_complete=lambda key, cov: completed.append(key),
    )
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert completed == [(0, 0, 128)]
    assert d.processed == NUM_LAYERS  # per-layer: one message per layer
    assert d.chunks == {}  # evicted on completion
    assert d.completed_chunks == 1


def test_interleaved_chunks_advance_independently():
    completed = []
    ring = FakeRing(
        per_layer(0, layers=[0, 1]) + per_layer(1, layers=[0, 1, 2, 3], slot_id=1) + per_layer(0, layers=[2, 3])
    )
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, on_chunk_complete=lambda key, cov: completed.append(key))
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    # chunk on slot 1 finished before the one on slot 0: no head-of-line coupling
    assert completed == [(1, 0, 128), (0, 0, 128)]


def test_chunks_are_keyed_on_identity_not_request_id():
    """The host transport stamps the runner's request id, the D2H transport a per-rank counter:
    the same chunk arrives under two request ids and must still tile once."""
    ring = FakeRing(
        [
            msg(7, 0, 2, slot_id=3, pos_start=5120, pos_end=10240),
            msg(99, 2, 4, slot_id=3, pos_start=5120, pos_end=10240),
        ]
    )
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert d.completed_chunks == 1


def test_same_identity_can_come_round_again_after_completion():
    """Slot reuse and multi-turn resumes repeat (slot, pos_start, pos_end); eviction makes that legal."""
    ring = FakeRing(per_layer(0) + per_layer(1))  # both on slot 0, positions [0,128)
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    assert d.completed_chunks == 2 and d.chunks == {}


def test_out_of_order_spans_tile():
    ring = FakeRing([msg(0, 2, 4), msg(0, 0, 2)])  # two halves, reversed
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS


def test_wide_span_counts_full_width_as_one_message():
    ring = FakeRing([msg(0, 0, NUM_LAYERS)])  # whole stage in one message
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS)
    assert d.drain_blocking(NUM_LAYERS, timeout_s=5) == NUM_LAYERS
    assert d.processed == 1


def test_acks_past_the_model_layers_fit_a_widened_bound():
    """MTP and DFlash ack rows past the last model layer; the bound is the widened count."""
    extra = 2
    ring = FakeRing(per_layer(0, layers=range(NUM_LAYERS + extra)))
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS + extra)
    assert d.drain_blocking(NUM_LAYERS + extra, timeout_s=5) == NUM_LAYERS + extra
    assert d.completed_chunks == 1


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

    ring = FakeRing([msg(0, 0, 2), *per_layer(1, slot_id=1), msg(0, 2, 4)])
    d = LayerCompletionDrainer(ring, num_layers=NUM_LAYERS, can_process=can_process, on_completion=on_completion)
    assert d.drain_blocking(2 * NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS
    assert seen[0][0] == 1  # slot 1 work ran before any slot 0 work
    assert d.side_queued == 1


def test_side_queue_retried_oldest_chunk_first():
    order = []
    ready = {"go": False}
    d = LayerCompletionDrainer(
        FakeRing([msg(5, 0, 2, slot_id=5), msg(2, 0, 2, slot_id=2), msg(5, 2, 4, slot_id=5), msg(2, 2, 4, slot_id=2)]),
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
    d = LayerCompletionDrainer(FakeRing([msg(0, 0, 3), msg(0, 2, 4)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="overlaps"):  # allow-pytest.raises: host-only, no device error
        while d.step():
            pass


def test_out_of_bounds_span_raises():
    d = LayerCompletionDrainer(FakeRing([msg(0, 2, NUM_LAYERS + 1)]), num_layers=NUM_LAYERS)
    with pytest.raises(ValueError, match="out of bounds"):  # allow-pytest.raises: host-only, no device error
        d.step()


def test_finish_raises_on_stranded_side_queue():
    d = LayerCompletionDrainer(
        FakeRing([msg(0, 0, 2)]), num_layers=NUM_LAYERS, can_process=lambda c: False  # never actionable
    )
    while d.step():
        pass
    with pytest.raises(RuntimeError, match="never-actionable"):  # allow-pytest.raises: host-only, no device error
        d.finish()


def test_drain_blocking_timeout_reports_open_chunks():
    d = LayerCompletionDrainer(FakeRing(per_layer(0, layers=[0, 1])), num_layers=NUM_LAYERS)
    with pytest.raises(TimeoutError, match="2/4 layers"):  # allow-pytest.raises: host-only, no device error
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
    assert lcd.drain_layer_completions(channel, 3 * NUM_LAYERS, num_layers=NUM_LAYERS, timeout_s=5) == 3 * NUM_LAYERS


def test_drain_layer_completions_dispatches_v2_ring(monkeypatch):
    """v2: the dispatcher routes the ring through the work-conserving drainer with the caller's bound."""
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    ring = FakeRing(per_layer(0, layers=[0, 1]) + per_layer(1, slot_id=1) + per_layer(0, layers=[2, 3]))
    assert lcd.drain_layer_completions(ring, 2 * NUM_LAYERS, num_layers=NUM_LAYERS, timeout_s=5) == 2 * NUM_LAYERS


def test_drain_layer_completions_surfaces_producer_bugs(monkeypatch):
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    with pytest.raises(ValueError, match="out of bounds"):  # allow-pytest.raises: host-only, no device error
        lcd.drain_layer_completions(
            FakeRing([msg(0, 0, NUM_LAYERS + 1)]), NUM_LAYERS, num_layers=NUM_LAYERS, timeout_s=1
        )


def test_drain_layer_completions_none_channel_is_noop(monkeypatch):
    import models.demos.common.prefill.runners.layer_completion_drainer as lcd

    monkeypatch.setenv("PREFILL_LAYER_COMPLETION_PROTOCOL", "2")
    assert lcd.drain_layer_completions(None, NUM_LAYERS, num_layers=NUM_LAYERS, timeout_s=1) == 0


def test_scheduler_segment_names_are_protocol_specific():
    """The v1 counter channel validates nothing on attach, so the two protocols must never share a name."""
    assert scheduler_shm_name("svc", 1) == "/tt_prefill_layer_acks_svc"  # frozen: existing v1 consumers
    assert scheduler_shm_name("svc", 2) != scheduler_shm_name("svc", 1)
    assert scheduler_shm_name("svc", 2).startswith("/") and "/" not in scheduler_shm_name("svc", 2)[1:]
