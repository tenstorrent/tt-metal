# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Layer-completion sinks for pipelined prefill (host-callback transport).

The runtime fires `sink.layers_completed(layer_start, layer_end, request_id, slot_id, pos_start,
pos_end)` once per completion event: a half-open global layer range at the model's natural
granularity plus the chunk's request id, cache slot and KV-position range. Each sink maps the
event onto its protocol's wire format (see layer_completion_message.hpp):

* CountOnlyLayerCompletionSink (protocol 1): one message per covered layer, seq dense in ACK
  space (a hybrid stack acks only on KV-writing layers; `ack_idx_of_layer` remaps), layer_idx
  global.
* StructuredLayerCompletionSink (protocol 2): one self-describing message per span; seq is this
  sink's emission order.

TODO(#54632): the v2 span on a hybrid stack is the raw global layer; see LAYER_COMPLETION_OPENS.md §2.
ttnn-free so the sinks are unit-testable.
"""

import time
from abc import ABC, abstractmethod

from loguru import logger

LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001


def _push_with_spin(try_push, *, seq: int, is_shutdown) -> None:
    """Full-ring backpressure: wait for the router to drain, warning on entry and every
    LOG_EVERY_S. Only an operator shutdown ends the wait, dropping the message."""
    start = time.monotonic()
    next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
    logger.warning(f"[layer-completion] ring full (seq={seq}); waiting for the router to drain")
    while not try_push():
        if is_shutdown():
            logger.warning(f"[layer-completion] shutdown while waiting on a full ring; dropped seq={seq}")
            return
        now = time.monotonic()
        if now >= next_log:
            logger.warning(f"[layer-completion] still waiting on a full ring (seq={seq}) after {now - start:.0f}s")
            next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)
    logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed seq={seq}")


class LayerCompletionSink(ABC):
    @abstractmethod
    def layers_completed(
        self, layer_start: int, layer_end: int, request_id: int, slot_id: int, pos_start: int, pos_end: int
    ) -> None:
        """One completion event: global layers [layer_start, layer_end) of the chunk
        (request_id, slot_id, KV positions [pos_start, pos_end))."""


class CountOnlyLayerCompletionSink(LayerCompletionSink):
    """Protocol 1: one 24 B message per covered layer; the master reorders by the dense seq and
    hands the scheduler a count. Slot and positions are not on the v1 wire."""

    def __init__(
        self, producer, *, source_rank: int, num_ack_layers: int, ack_idx_of_layer=None, is_shutdown=lambda: False
    ):
        self._producer = producer
        self._source_rank = source_rank
        self._num_ack_layers = num_ack_layers
        self._ack_idx_of_layer = ack_idx_of_layer
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, pos_start, pos_end) -> None:
        for layer_idx in range(layer_start, layer_end):
            # seq in ACK space, layer_idx global; a KeyError here is a producer bug.
            ack_idx = layer_idx if self._ack_idx_of_layer is None else self._ack_idx_of_layer[layer_idx]
            seq = request_id * self._num_ack_layers + ack_idx
            fields = dict(seq=seq, source_rank=self._source_rank, layer_idx=layer_idx, request_id=request_id)
            if not self._producer.try_push(**fields):
                _push_with_spin(lambda: self._producer.try_push(**fields), seq=seq, is_shutdown=self._is_shutdown)


class StructuredLayerCompletionSink(LayerCompletionSink):
    """Protocol 2: one 40 B self-describing message per span, forwarded as it arrives."""

    def __init__(self, producer, *, source_rank: int, is_shutdown=lambda: False):
        self._producer = producer
        self._source_rank = source_rank
        self._is_shutdown = is_shutdown
        self._emitted = 0

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, pos_start, pos_end) -> None:
        seq = self._emitted
        self._emitted += 1
        fields = dict(
            seq=seq,
            source_rank=self._source_rank,
            request_id=request_id,
            slot_id=slot_id,
            pos_start=pos_start,
            pos_end=pos_end,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        if not self._producer.try_push(**fields):
            _push_with_spin(lambda: self._producer.try_push(**fields), seq=seq, is_shutdown=self._is_shutdown)


class NullLayerCompletionSink(LayerCompletionSink):
    """No-op sink for warm passes (see TtPrefillRuntime.compile)."""

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, pos_start, pos_end) -> None:
        return None
