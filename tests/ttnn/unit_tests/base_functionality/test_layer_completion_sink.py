# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layer-completion sinks (host-only, fake producer)."""

import pytest

from models.demos.common.prefill.runners import layer_completion_sink as lcs


class FakeProducer:
    """Records every try_push attempt; reports a full ring for the first `fail_first` attempts."""

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.attempts = []

    def try_push(self, **fields) -> bool:
        self.attempts.append(fields)
        return len(self.attempts) > self.fail_first


NUM_ACK_LAYERS = 61
RANK = 2
# One completion event as fired by the runtime's per-chunk closure.
EVENT = dict(layer_start=14, layer_end=15, request_id=7, slot_id=5, pos_start=5120, pos_end=10213)


def v1(producer, **kw):
    return lcs.CountOnlyLayerCompletionSink(producer, source_rank=RANK, num_ack_layers=NUM_ACK_LAYERS, **kw)


def v2(producer, **kw):
    return lcs.StructuredLayerCompletionSink(producer, source_rank=RANK, **kw)


def test_sinks_are_polymorphic():
    assert isinstance(v1(FakeProducer()), lcs.LayerCompletionSink)
    assert isinstance(v2(FakeProducer()), lcs.LayerCompletionSink)


def test_v1_sink_one_message_per_layer_frozen_fields():
    producer = FakeProducer()
    v1(producer).layers_completed(**EVENT)
    assert producer.attempts == [dict(seq=7 * NUM_ACK_LAYERS + 14, source_rank=RANK, layer_idx=14, request_id=7)]


def test_v1_sink_splits_span_into_dense_per_layer_messages():
    producer = FakeProducer()
    v1(producer).layers_completed(14, 17, 7, 5, 5120, 10213)
    assert producer.attempts == [
        dict(seq=7 * NUM_ACK_LAYERS + layer, source_rank=RANK, layer_idx=layer, request_id=7) for layer in (14, 15, 16)
    ]


def test_v2_sink_pushes_full_fields_with_emission_order_seq():
    producer = FakeProducer()
    sink = v2(producer)
    sink.layers_completed(**EVENT)
    sink.layers_completed(layer_start=0, layer_end=14, request_id=3, slot_id=1, pos_start=0, pos_end=5120)
    assert producer.attempts[0] == dict(
        seq=0, source_rank=RANK, request_id=7, slot_id=5, pos_start=5120, pos_end=10213, layer_start=14, layer_end=15
    )
    assert producer.attempts[1]["seq"] == 1 and (
        producer.attempts[1]["layer_start"],
        producer.attempts[1]["layer_end"],
    ) == (0, 14)
    assert len(producer.attempts) == 2  # a span is one message, never split


@pytest.mark.parametrize("make", [v1, v2])
def test_sink_waits_on_a_full_ring_and_warns_at_a_cadence(monkeypatch, make):
    """A full ring is backpressure, not an error: wait, one warning on entry and one per
    LOG_EVERY_S while waiting, never a raise."""
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S", 0.02)
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_SLEEP_S", 0.005)
    warnings = []
    monkeypatch.setattr(lcs.logger, "warning", lambda msg: warnings.append(msg))
    producer = FakeProducer(fail_first=20)
    make(producer).layers_completed(**EVENT)
    assert len(producer.attempts) == 21
    assert 2 <= len(warnings) <= 8
    assert warnings[0].startswith("[layer-completion] ring full")
    assert all(a == producer.attempts[0] for a in producer.attempts)  # identical payload every retry


@pytest.mark.parametrize("make", [v1, v2])
def test_sink_shutdown_drops_the_message_and_returns(make):
    producer = FakeProducer(fail_first=10**9)
    make(producer, is_shutdown=lambda: len(producer.attempts) >= 3).layers_completed(**EVENT)
    assert len(producer.attempts) == 3


# A hybrid stack acks only on KV-writing layers: six events per 24-layer Kimi-K3 rank. The v1
# reorder buffer needs a dense seq, so the sink remaps to ACK space; layer_idx stays global.
KIMI_ACK_LAYERS = [3, 7, 11, 15, 19, 23]
KIMI_ACK_IDX = {layer: idx for idx, layer in enumerate(KIMI_ACK_LAYERS)}


def test_v1_sink_remaps_seq_to_ack_space_but_keeps_global_layer_idx():
    producer = FakeProducer()
    sink = lcs.CountOnlyLayerCompletionSink(
        producer, source_rank=RANK, num_ack_layers=len(KIMI_ACK_LAYERS), ack_idx_of_layer=KIMI_ACK_IDX
    )
    for layer in KIMI_ACK_LAYERS:
        sink.layers_completed(layer, layer + 1, 1, 5, 0, 5120)
    assert [a["seq"] for a in producer.attempts] == [len(KIMI_ACK_LAYERS) + i for i in range(len(KIMI_ACK_LAYERS))]
    assert [a["layer_idx"] for a in producer.attempts] == KIMI_ACK_LAYERS


def test_v1_sink_hybrid_layer_outside_the_map_is_a_producer_bug():
    sink = lcs.CountOnlyLayerCompletionSink(
        FakeProducer(), source_rank=RANK, num_ack_layers=len(KIMI_ACK_LAYERS), ack_idx_of_layer=KIMI_ACK_IDX
    )
    with pytest.raises(KeyError):  # allow-pytest.raises: host-only
        sink.layers_completed(4, 5, 1, 5, 0, 5120)


def test_null_sink_emits_nothing():
    sink = lcs.NullLayerCompletionSink()
    assert isinstance(sink, lcs.LayerCompletionSink)
    assert sink.layers_completed(**EVENT) is None
