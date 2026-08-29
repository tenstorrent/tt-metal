# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Layer-completion sinks for pipelined prefill.

The runtime binds one per-chunk closure in prefill_chunk() which fires the
registered sink once per completion EVENT inside model.forward. The sink
interface is range-native and protocol-polymorphic:

    sink.layers_completed(layer_start, layer_end, request_id, slot_id,
                          actual_start, actual_end)

— a half-open GLOBAL layer range at the model's natural completion
granularity (per-layer runtimes fire [l, l+1); a stage-level runtime fires
[first, end)) plus the chunk's request id, cache user slot, and absolute
KV-position range. Each protocol implementation maps the span onto its wire
format:

* v1 (CountedLayerCompletionSink) — the frozen count protocol: ONE message
  per covered layer, {seq, source_rank, layer_idx, request_id}, dense
  seq = request_id*num_layers + layer_idx. The master router reorders by seq
  and emits only a COUNT to the scheduler. (Lives here, not inline in
  prefill_runner.py, so both implementations share the backpressure policy;
  the wire format is unchanged.)

* v2 (StructuredLayerCompletionSink) — the structured protocol (issue
  #54632): ONE self-describing message per span, {seq, source_rank,
  request_id, slot_id, pos_start, pos_end, layer_start, layer_end}, which the
  master forwards as-arrived (no reorder → no per-request head-of-line
  blocking). seq is diagnostic only in v2.

Selected per job by PREFILL_LAYER_COMPLETION_PROTOCOL (1 | 2) in
prefill_runner.py. Kept dependency-light (stdlib + loguru — no ttnn) so the
sinks are unit testable without importing the device stack.
"""

import os
import time
from abc import ABC, abstractmethod

from loguru import logger

# When the completion ring is full, spin waiting for the router to drain rather than
# dropping/failing immediately. Bounded so a genuinely stalled router still surfaces.
LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001  # tiny yield so the spin doesn't peg a core


def _push_with_spin(try_push, *, seq: int, is_shutdown) -> None:
    """Push with bounded full-ring backpressure.

    try_push() is a zero-arg callable returning bool. The ring is sized well
    above in-flight depth; a full ring means the router thread is momentarily
    behind. Spin (don't drop) for up to the timeout, logging on entry, every
    LOG_EVERY_S while waiting, and on exit. Raises if the router never catches
    up, or immediately if is_shutdown() reports an operator stop (SIGTERM).
    """
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
    """Polymorphic completion sink; the runtime binds per-chunk fields and fires
    layers_completed() once per completion event inside model.forward."""

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
        """One completion event covering global layers [layer_start, layer_end)
        for the chunk (request_id, slot_id, KV positions [actual_start, actual_end))."""


class CountedLayerCompletionSink(LayerCompletionSink):
    """v1 — count protocol (frozen wire format): one 24B message PER COVERED LAYER,
    {seq, source_rank, layer_idx, request_id}; the master reorders by the dense seq
    and emits only a count to the scheduler. A multi-layer span is split into
    per-layer messages because that is all the v1 wire can express. The slot and
    position fields are accepted by the interface and intentionally unused."""

    def __init__(self, producer, *, source_rank: int, num_layers: int, is_shutdown=lambda: False):
        self._producer = producer
        self._source_rank = source_rank
        self._num_layers = num_layers  # GLOBAL total (seq stride), NOT this rank's slice
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, actual_start, actual_end) -> None:
        for layer_idx in range(layer_start, layer_end):
            seq = request_id * self._num_layers + layer_idx
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
    """v2 — structured protocol: ONE self-describing 40B message per span, carrying
    the chunk's slot and position range alongside the layer range, forwarded
    as-arrived to the scheduler-facing ring. seq is diagnostic only."""

    def __init__(self, producer, *, source_rank: int, num_layers: int, is_shutdown=lambda: False):
        self._producer = producer
        self._source_rank = source_rank
        self._num_layers = num_layers  # GLOBAL total (diagnostic seq stride)
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, request_id, slot_id, actual_start, actual_end) -> None:
        # Diagnostic-only ordering key (v2 forwards as-arrived); keyed on the first
        # covered layer so multi-layer events stay monotonic within a request.
        seq = request_id * self._num_layers + layer_start
        fields = dict(
            seq=seq,
            source_rank=self._source_rank,
            request_id=request_id,
            slot_id=slot_id,
            pos_start=actual_start,
            pos_end=actual_end,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        if self._producer.try_push(**fields):
            return
        _push_with_spin(lambda: self._producer.try_push(**fields), seq=seq, is_shutdown=self._is_shutdown)


def build_layer_completion_sink(producer, *, source_rank: int, num_layers: int, is_shutdown=lambda: False):
    """v1 sink factory — count protocol (frozen wire format)."""
    return CountedLayerCompletionSink(
        producer, source_rank=source_rank, num_layers=num_layers, is_shutdown=is_shutdown
    )


def build_layer_completion_sink_v2(producer, *, source_rank: int, num_layers: int, is_shutdown=lambda: False):
    """v2 sink factory — structured protocol (issue #54632)."""
    return StructuredLayerCompletionSink(
        producer, source_rank=source_rank, num_layers=num_layers, is_shutdown=is_shutdown
    )
