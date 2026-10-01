# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from models.demos.common.prefill.runners import layer_completion_sink as lcs


class FakeProducer:
    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.attempts = []

    def try_push(self, **fields) -> bool:
        self.attempts.append(fields)
        return len(self.attempts) > self.fail_first


NUM_LAYERS = 61
RANK = 2

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

    assert producer.attempts == [dict(seq=7 * NUM_LAYERS + 14, source_rank=RANK, layer_idx=14, request_id=7)]


def test_v1_sink_splits_span_into_dense_per_layer_messages():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(layer_start=14, layer_end=17, request_id=7, slot_id=5, actual_start=5120, actual_end=10213)

    assert producer.attempts == [
        dict(seq=7 * NUM_LAYERS + layer, source_rank=RANK, layer_idx=layer, request_id=7) for layer in (14, 15, 16)
    ]


def test_v2_sink_pushes_full_fields():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink_v2(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    before_ns = time.time_ns()
    sink.layers_completed(**EVENT)
    after_ns = time.time_ns()

    assert before_ns <= producer.attempts[0].pop("host_ts_ns") <= after_ns
    assert producer.attempts == [
        dict(
            seq=7 * NUM_LAYERS + 14,
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
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink_v2(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(layer_start=0, layer_end=14, request_id=3, slot_id=1, actual_start=0, actual_end=5120)

    assert len(producer.attempts) == 1
    fields = producer.attempts[0]
    assert fields["layer_start"] == 0
    assert fields["layer_end"] == 14
    assert fields["seq"] == 3 * NUM_LAYERS + 0


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_spins_until_ring_drains(builder):
    producer = FakeProducer(fail_first=5)
    sink = builder(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(**EVENT)

    assert len(producer.attempts) == 6
    first = producer.attempts[0]
    assert all(a == first for a in producer.attempts)


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_timeout_raises(monkeypatch, builder):
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lcs, "LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S", 0.02)
    producer = FakeProducer(fail_first=10**9)
    sink = builder(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    with pytest.raises(RuntimeError, match="router not draining"):  # allow-pytest.raises: host-only, no device error
        sink.layers_completed(**EVENT)


@pytest.mark.parametrize("builder", [lcs.build_layer_completion_sink, lcs.build_layer_completion_sink_v2])
def test_sink_shutdown_aborts_spin(builder):
    producer = FakeProducer(fail_first=10**9)
    sink = builder(producer, source_rank=RANK, num_layers=NUM_LAYERS, is_shutdown=lambda: len(producer.attempts) >= 3)

    with pytest.raises(RuntimeError, match="shutdown requested"):  # allow-pytest.raises: host-only, no device error
        sink.layers_completed(**EVENT)
    assert len(producer.attempts) == 3


KIMI_ACK_LAYERS = [3, 7, 11, 15, 19, 23]
KIMI_ACK_IDX = {layer: idx for idx, layer in enumerate(KIMI_ACK_LAYERS)}
NUM_ACK_LAYERS = len(KIMI_ACK_LAYERS)


def test_v1_sink_remaps_seq_to_ack_space_but_keeps_global_layer_idx():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink(
        producer, source_rank=RANK, num_layers=NUM_ACK_LAYERS, ack_idx_of_layer=KIMI_ACK_IDX
    )

    for layer in KIMI_ACK_LAYERS:
        sink.layers_completed(layer, layer + 1, 1, 5, 0, 5120)

    assert [a["seq"] for a in producer.attempts] == [1 * NUM_ACK_LAYERS + i for i in range(NUM_ACK_LAYERS)]
    assert [a["layer_idx"] for a in producer.attempts] == KIMI_ACK_LAYERS


def test_v2_sink_remaps_seq_but_emits_the_global_span():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink_v2(
        producer, source_rank=RANK, num_layers=NUM_ACK_LAYERS, ack_idx_of_layer=KIMI_ACK_IDX
    )

    sink.layers_completed(11, 12, 1, 5, 0, 5120)

    (pushed,) = producer.attempts
    assert pushed["seq"] == 1 * NUM_ACK_LAYERS + KIMI_ACK_IDX[11]
    assert (pushed["layer_start"], pushed["layer_end"]) == (11, 12)


def test_dense_model_needs_no_translation():
    producer = FakeProducer()
    sink = lcs.build_layer_completion_sink(producer, source_rank=RANK, num_layers=NUM_LAYERS)

    sink.layers_completed(**EVENT)

    assert producer.attempts[0]["seq"] == 7 * NUM_LAYERS + 14


def test_null_sink_emits_nothing():
    sink = lcs.NullLayerCompletionSink()
    assert isinstance(sink, lcs.LayerCompletionSink)
    assert sink.layers_completed(**EVENT) is None
