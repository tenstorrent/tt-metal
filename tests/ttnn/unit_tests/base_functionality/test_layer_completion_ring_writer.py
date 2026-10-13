# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layer-completion sinks (host-only, fake producer)."""

from models.demos.common.prefill.runners import layer_completion_ring_writer as writer_mod


class FakeProducer:
    """Records every try_push attempt; reports a full ring for the first `fail_first` attempts."""

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.attempts = []

    def try_push(self, **fields) -> bool:
        self.attempts.append(fields)
        return len(self.attempts) > self.fail_first


# One completion event as fired by the runtime's per-chunk closure.
EVENT = dict(layer_start=14, layer_end=15, slot_id=5, pos_start=5120, pos_end=10213)


def make(producer, **kw):
    return writer_mod.LayerCompletionRingWriter(producer, **kw)


def test_sink_pushes_one_message_per_span():
    producer = FakeProducer()
    sink = make(producer)
    sink.layers_completed(**EVENT)
    sink.layers_completed(layer_start=0, layer_end=14, slot_id=1, pos_start=0, pos_end=5120)
    assert producer.attempts == [
        dict(slot_id=5, pos_start=5120, pos_end=10213, layer_start=14, layer_end=15),
        dict(slot_id=1, pos_start=0, pos_end=5120, layer_start=0, layer_end=14),
    ]


def test_sink_waits_on_a_full_ring_and_warns_at_a_cadence(monkeypatch):
    """A full ring is backpressure, not an error: wait, one warning on entry and one per
    LOG_EVERY_S while waiting, never a raise."""
    monkeypatch.setattr(writer_mod, "LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S", 0.02)
    monkeypatch.setattr(writer_mod, "LAYER_COMPLETION_PUSH_SPIN_SLEEP_S", 0.005)
    warnings = []
    monkeypatch.setattr(writer_mod.logger, "warning", lambda msg: warnings.append(msg))
    producer = FakeProducer(fail_first=20)
    make(producer).layers_completed(**EVENT)
    assert len(producer.attempts) == 21
    assert 2 <= len(warnings) <= 8
    assert warnings[0].startswith("[layer-completion] ring full")
    assert all(a == producer.attempts[0] for a in producer.attempts)  # identical payload every retry


def test_sink_shutdown_drops_the_message_and_returns():
    producer = FakeProducer(fail_first=10**9)
    make(producer, is_shutdown=lambda: len(producer.attempts) >= 3).layers_completed(**EVENT)
    assert len(producer.attempts) == 3
