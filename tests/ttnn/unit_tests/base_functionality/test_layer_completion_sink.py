# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layer-completion sinks (v1 count / v2 structured).

Host-only: exercises models.demos.common.prefill.runners.layer_completion_sink
with a fake producer — no ttnn import, no device. Both sinks share the
range-native polymorphic interface

    sink.layers_completed(layer_start, layer_end, request_id, slot_id,
                          actual_start, actual_end)

and map the span onto their protocol's wire format (v1: one message per
covered layer; v2: one self-describing message per span).
"""

import pytest

from models.demos.common.prefill.runners import layer_completion_sink as lcs


class FakeProducer:
    """LayerCompletionQueue/V2 stand-in: records every try_push attempt; can be
    told to report a full ring for the first `fail_first` attempts."""

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.attempts = []  # every try_push payload, accepted or not

    def try_push(self, **fields) -> bool:
        self.attempts.append(fields)
        return len(self.attempts) > self.fail_first


NUM_LAYERS = 61
RANK = 2

# One completion event as fired by the runtime's per-chunk closure:
# (layer_start, layer_end, request_id, slot_id, actual_start, actual_end)
EVENT = dict(layer_start=14, layer_end=15, request_id=7, slot_id=5, actual_start=5120, actual_end=10213)


def test_sinks_are_polymorphic():
    v1 = lcs.build_layer_completion_sink(FakeProducer(), source_rank=RANK, num_layers=NUM_LAYERS)
    v2 = lcs.build_layer_completion_sink_v2(FakeProducer(), source_rank=RANK, num_layers=NUM_LAYERS)
    assert isinstance(v1, lcs.LayerCompletionSink)
    assert isinstance(v2, lcs.LayerCompletionSink)


def test_v1_sink_one_message_per_layer_frozen_fields():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(**EVENT)

    assert producer.attempts == [
        dict(seq=7 * NUM_LAYERS + 14, source_rank=RANK, layer_idx=14, request_id=7)  # no slot/pos on v1 wire
    ]


def test_v1_sink_splits_span_into_dense_per_layer_messages():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(layer_start=14, layer_end=17, request_id=7, slot_id=5, actual_start=5120,
                          actual_end=10213)

    assert producer.attempts == [
        dict(seq=7 * NUM_LAYERS + layer, source_rank=RANK, layer_idx=layer, request_id=7)
        for layer in (14, 15, 16)
    ]


def test_v2_sink_pushes_full_fields():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink_v2(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(**EVENT)

    assert producer.attempts == [
        dict(
            seq=7 * NUM_LAYERS + 14,  # request_id * num_layers + layer_start
            source_rank=RANK,
            request_id=7,
            slot_id=5,
            pos_start=5120,
            pos_end=10213,
            layer_start=14,
            layer_end=15,
        )
    ]


def test_v2_sink_range_passthrough_single_message():
    """A stage-level completion (3 layers at once) is pushed as ONE message — never split."""
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink_v2(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(layer_start=0, layer_end=14, request_id=3, slot_id=1, actual_start=0, actual_end=5120)

    assert len(producer.attempts) == 1
    fields = producer.attempts[0]
    assert fields["layer_start"] == 0
    assert fields["layer_end"] == 14
    assert fields["seq"] == 3 * NUM_LAYERS + 0  # keyed on the first covered layer


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_spins_until_ring_drains(builder):
    """Full ring → spin (identical payload every retry) until the router drains."""
    producer = FakeProducer(fail_first=5)
    sink = builder(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(**EVENT)

    assert len(producer.attempts) == 6  # 5 full-ring refusals, then success
    first = producer.attempts[0]
    assert all(a == first for a in producer.attempts)


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_timeout_raises(monkeypatch, builder):
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S", 0.02)
    producer = FakeProducer(fail_first=10**9)  # never drains
    sink = builder(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    with pytest.raises(RuntimeError, match="router not draining"):
        sink.layers_completed(**EVENT)


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_shutdown_aborts_spin(builder):
    producer = FakeProducer(fail_first=10**9)
    sink = builder(
        producer, source_rank=RANK, num_layers=NUM_LAYERS, is_shutdown=lambda: len(producer.attempts) >= 3
    )

    with pytest.raises(RuntimeError, match="shutdown requested"):
        sink.layers_completed(**EVENT)
    assert len(producer.attempts) == 3  # aborted promptly, not at the timeout
