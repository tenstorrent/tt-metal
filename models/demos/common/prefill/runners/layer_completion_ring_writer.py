# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Layer-completion sinks for the host-callback transport. The runtime fires `sink.layers_completed(layer_start,
layer_end, slot_id, pos_start, pos_end)` once per completion event; LayerCompletionRingWriter pushes one message
per span into the rank's ring. ttnn-free, so unit-testable.
"""

import time
from abc import ABC, abstractmethod

from loguru import logger

LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001


def _push_with_spin(try_push, *, what: str, is_shutdown) -> None:
    """Full-ring backpressure: wait for the router to drain, warning on entry and every LOG_EVERY_S; only a
    shutdown ends the wait, dropping the message."""
    start = time.monotonic()
    next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
    logger.warning(f"[layer-completion] ring full ({what}); waiting for the router to drain")
    while not try_push():
        if is_shutdown():
            logger.warning(f"[layer-completion] shutdown while waiting on a full ring; dropped {what}")
            return
        now = time.monotonic()
        if now >= next_log:
            logger.warning(f"[layer-completion] still waiting on a full ring ({what}) after {now - start:.0f}s")
            next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)
    logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed {what}")


class LayerCompletionSink(ABC):
    @abstractmethod
    def layers_completed(self, layer_start: int, layer_end: int, slot_id: int, pos_start: int, pos_end: int) -> None:
        """Global layers [layer_start, layer_end) of the chunk (slot_id, positions [pos_start, pos_end))."""


class LayerCompletionRingWriter(LayerCompletionSink):
    """One message per span into the rank's ring."""

    def __init__(self, producer, *, is_shutdown=lambda: False):
        self._producer = producer
        self._is_shutdown = is_shutdown

    def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end) -> None:
        fields = dict(
            slot_id=slot_id, pos_start=pos_start, pos_end=pos_end, layer_start=layer_start, layer_end=layer_end
        )
        if not self._producer.try_push(**fields):
            what = f"slot {slot_id} [{pos_start},{pos_end}) layers [{layer_start},{layer_end})"
            _push_with_spin(lambda: self._producer.try_push(**fields), what=what, is_shutdown=self._is_shutdown)


class NullLayerCompletionSink(LayerCompletionSink):
    """No-op sink for warm passes (see TtPrefillRuntime.compile)."""

    def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end) -> None:
        return None
