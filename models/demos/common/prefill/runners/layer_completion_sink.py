# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from abc import ABC, abstractmethod

from loguru import logger

LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001


def _push_with_spin(try_push, *, seq: int, is_shutdown) -> None:
    start = time.monotonic()
    next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
    logger.warning(
        f"[layer-completion] ring full (seq={seq}); spinning up to "
        f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s for router to drain"
    )
    while True:
        if try_push():
            logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed seq={seq}")
            return
        if is_shutdown():
            raise RuntimeError(f"layer-completion ring full (seq={seq}); shutdown requested while spinning")
        now = time.monotonic()
        if now - start >= LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:
            logger.error(f"[layer-completion] gave up after {now - start:.1f}s spinning on full ring (seq={seq})")
            raise RuntimeError(
                f"layer-completion ring full (seq={seq}); router not draining after "
                f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s"
            )
        if now >= next_log:
            logger.warning(f"[layer-completion] still spinning on full ring (seq={seq}) after {now - start:.0f}s")
            next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)


class LayerCompletionSink(ABC):
    @abstractmethod
    def layers_completed(
        self,
        layer_start: int,
        layer_end: int,
        request_id: int,
        slot_id: int,
        actual_start: int,
        actual_end: int,
    ) -> None:
        ...


class CountedLayerCompletionSink(LayerCompletionSink):
    def __init__(
        self, producer, *, source_rank: int, num_layers: int, ack_idx_of_layer=None, is_shutdown=lambda: False
    ):
        self._producer = producer
        self._source_rank = source_rank
        self._num_layers = num_layers
        self._ack_idx_of_layer = ack_idx_of_layer
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, actual_start, actual_end) -> None:
        for layer_idx in range(layer_start, layer_end):
            ack_idx = layer_idx if self._ack_idx_of_layer is None else self._ack_idx_of_layer[layer_idx]
            seq = request_id * self._num_layers + ack_idx
            if self._producer.try_push(
                seq=seq, source_rank=self._source_rank, layer_idx=layer_idx, request_id=request_id
            ):
                continue
            _push_with_spin(
                lambda seq=seq, layer_idx=layer_idx: self._producer.try_push(
                    seq=seq, source_rank=self._source_rank, layer_idx=layer_idx, request_id=request_id
                ),
                seq=seq,
                is_shutdown=self._is_shutdown,
            )


class StructuredLayerCompletionSink(LayerCompletionSink):
    def __init__(
        self, producer, *, source_rank: int, num_layers: int, ack_idx_of_layer=None, is_shutdown=lambda: False
    ):
        self._producer = producer
        self._source_rank = source_rank
        self._num_layers = num_layers
        self._ack_idx_of_layer = ack_idx_of_layer
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, actual_start, actual_end) -> None:
        ack_idx = layer_start if self._ack_idx_of_layer is None else self._ack_idx_of_layer[layer_start]
        seq = request_id * self._num_layers + ack_idx
        fields = dict(
            seq=seq,
            source_rank=self._source_rank,
            request_id=request_id,
            slot_id=slot_id,
            pos_start=actual_start,
            pos_end=actual_end,
            layer_start=layer_start,
            layer_end=layer_end,
            host_ts_ns=time.time_ns(),
        )
        if self._producer.try_push(**fields):
            return
        _push_with_spin(lambda: self._producer.try_push(**fields), seq=seq, is_shutdown=self._is_shutdown)


class NullLayerCompletionSink(LayerCompletionSink):
    def layers_completed(self, layer_start, layer_end, request_id, slot_id, actual_start, actual_end) -> None:
        return None


def build_layer_completion_sink(
    producer, *, source_rank: int, num_layers: int, ack_idx_of_layer=None, is_shutdown=lambda: False
):
    return CountedLayerCompletionSink(
        producer,
        source_rank=source_rank,
        num_layers=num_layers,
        ack_idx_of_layer=ack_idx_of_layer,
        is_shutdown=is_shutdown,
    )


def build_layer_completion_sink_v2(
    producer, *, source_rank: int, num_layers: int, ack_idx_of_layer=None, is_shutdown=lambda: False
):
    return StructuredLayerCompletionSink(
        producer,
        source_rank=source_rank,
        num_layers=num_layers,
        ack_idx_of_layer=ack_idx_of_layer,
        is_shutdown=is_shutdown,
    )
