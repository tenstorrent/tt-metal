# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Consumer side of the layer-completion ring for host tools (prefill_producer, migration drivers, verifiers):

    channel = connect_layer_completion_channel(timeout_s)
    wait_for_layer_completions(channel, expected_layers, num_layers=...)

LayerCompletionRingReader keys each chunk on (slot_id, pos_start, pos_end). A chunk is complete when its disjoint
layer spans tile [0, num_layers) and is then evicted, so the same key can come round again. A completion that
`can_process` rejects waits in its chunk's side queue instead of stalling the ring; finish() raises on anything
still waiting.

ttnn is imported lazily inside the connect helper, so the reader is unit-testable without the device stack. The
ring is any object with `try_pop() -> tuple | None` in the wire order of LayerCompletionQueue.try_pop().
"""

import os
import time
from collections import deque, namedtuple

from loguru import logger

# Wire order of LayerCompletionQueue.try_pop().
Completion = namedtuple("Completion", "slot_id pos_start pos_end layer_start layer_end")


def chunk_key(c: Completion) -> tuple:
    return (c.slot_id, c.pos_start, c.pos_end)


class ChunkCoverage:
    """One in-flight chunk: its disjoint layer spans so far and its side queue."""

    __slots__ = ("key", "intervals", "layers_accounted", "side_queue")

    def __init__(self, key: tuple):
        self.key = key
        self.intervals = []  # sorted, disjoint [start, end)
        self.layers_accounted = 0
        self.side_queue = deque()

    def add_span(self, c: Completion, num_layers: int) -> None:
        """Raises on overlap or out-of-bounds: both are producer bugs."""
        start, end = c.layer_start, c.layer_end
        if not (0 <= start < end <= num_layers):
            raise ValueError(
                f"[ring-reader] chunk {self.key}: layer span [{start},{end}) out of bounds [0,{num_layers}): {c}"
            )
        for iv_start, iv_end in self.intervals:
            if start < iv_end and iv_start < end:
                raise ValueError(
                    f"[ring-reader] chunk {self.key}: layer span [{start},{end}) overlaps already-covered "
                    f"[{iv_start},{iv_end}): {c}"
                )
        self.intervals.append((start, end))
        self.intervals.sort()
        self.layers_accounted += end - start

    def is_complete(self, num_layers: int) -> bool:
        return self.layers_accounted == num_layers


class LayerCompletionRingReader:
    """Work-conserving reader of the scheduler-facing completion ring.

    Args:
        ring: anything with try_pop() -> tuple | None in the wire order.
        num_layers: global ack-layer count per chunk: the model's layers plus the MTP /
            DFlash rows acked past them (the coverage bound).
        can_process(completion) -> bool: readiness predicate; blocked messages are
            side-queued per chunk instead of stalling the ring. Default: always ready.
        on_completion(completion): action for each processed message. Default: none.
        on_chunk_complete(key, coverage): fired when a chunk tiles [0, num_layers), just
            before it is evicted. Default: none.
        poll_idle_s: sleep quantum for wait_for when nothing is actionable.
    """

    def __init__(
        self,
        ring,
        *,
        num_layers: int,
        can_process=None,
        on_completion=None,
        on_chunk_complete=None,
        poll_idle_s: float = 0.001,
    ):
        self._ring = ring
        self._num_layers = num_layers
        self._can_process = can_process or (lambda c: True)
        self._on_completion = on_completion or (lambda c: None)
        self._on_chunk_complete = on_chunk_complete or (lambda key, cov: None)
        self._poll_idle_s = poll_idle_s
        self._chunks = {}  # chunk_key -> ChunkCoverage, insertion-ordered, evicted on completion
        self.total_layers = 0  # span-summed layers processed (NOT message count)
        self.processed = 0
        self.side_queued = 0
        self.completed_chunks = 0

    @property
    def chunks(self):
        return self._chunks

    def _coverage(self, c: Completion) -> ChunkCoverage:
        key = chunk_key(c)
        cov = self._chunks.get(key)
        if cov is None:
            cov = self._chunks[key] = ChunkCoverage(key)
        return cov

    def _process(self, c: Completion) -> None:
        cov = self._coverage(c)
        cov.add_span(c, self._num_layers)
        self._on_completion(c)
        self.total_layers += c.layer_end - c.layer_start
        self.processed += 1
        if cov.is_complete(self._num_layers):
            self._on_chunk_complete(cov.key, cov)
            if cov.side_queue:
                raise RuntimeError(
                    f"[ring-reader] chunk {cov.key} completed with blocked completions: {list(cov.side_queue)}"
                )
            del self._chunks[cov.key]
            self.completed_chunks += 1

    def _retry_side_queues(self) -> bool:
        """Re-test side-queued messages, oldest chunk first. True if any became actionable."""
        progressed = False
        for cov in list(self._chunks.values()):
            sq = cov.side_queue
            while sq and self._can_process(sq[0]):
                self._process(sq.popleft())
                progressed = True
        return progressed

    def step(self) -> bool:
        """One work-conserving iteration. True if anything moved; False means idle."""
        msg = self._ring.try_pop()
        if msg is None:
            return self._retry_side_queues()
        c = Completion._make(msg)
        if self._can_process(c):
            self._process(c)
            self._retry_side_queues()  # processing may have unblocked side-queued messages
        else:
            self._coverage(c).side_queue.append(c)
            self.side_queued += 1
        return True

    def finish(self) -> int:
        """Teardown invariant: no side-queued message may remain. Returns total layers drained."""
        stranded = [(key, list(cov.side_queue)) for key, cov in self._chunks.items() if cov.side_queue]
        if stranded:
            detail = "; ".join(f"chunk {key}: {msgs}" for key, msgs in stranded)
            raise RuntimeError(f"[ring-reader] finish() with side-queued (never-actionable) completions: {detail}")
        return self.total_layers

    def wait_for(self, expected_layers: int, timeout_s: float = 600.0) -> int:
        """Loop until `expected_layers` (span-summed) have been processed, then finish().
        Raises TimeoutError with a coverage snapshot."""
        deadline = time.perf_counter() + timeout_s
        while self.total_layers < expected_layers:
            if not self.step():
                if time.perf_counter() > deadline:
                    snapshot = ", ".join(
                        f"chunk {key}: {cov.layers_accounted}/{self._num_layers} layers"
                        + (f" (+{len(cov.side_queue)} blocked)" if cov.side_queue else "")
                        for key, cov in self._chunks.items()
                    )
                    raise TimeoutError(
                        f"[ring-reader] timed out at {self.total_layers}/{expected_layers} layers "
                        f"after {timeout_s}s; open chunks: [{snapshot}]"
                    )
                time.sleep(self._poll_idle_s)
        return self.finish()


# Channel connect/wait: consumers hold one `completion_channel`.


def scheduler_shm_name(service_id: str) -> str:
    """The scheduler-facing ring the master router publishes. Frozen: tt-d-gen's ack_shm_name
    configs attach to it."""
    return f"/tt_prefill_layer_completions_{service_id}"


def connect_layer_completion_channel(timeout_s: int):
    """Attach (consumer side) to the master router's completion ring. None if unavailable."""
    from ttnn._experimental.layer_completion import LayerCompletionQueue

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = scheduler_shm_name(service_id)
    try:
        ring = LayerCompletionQueue.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[layer-completion] could not connect completion ring {shm_name}: {e}; skipping drain.")
        return None
    logger.info(f"[layer-completion] connected completion ring {shm_name}")
    return ring


def wait_for_layer_completions(
    completion_channel, expected_layers: int, *, num_layers: int, timeout_s: float = 600.0
) -> int:
    """Work-conserving drain until `expected_layers` (span-summed) are accounted; a
    producer-side inconsistency (overlap, out of bounds) raises. `num_layers` is the global
    ack-layer count per chunk (the coverage bound)."""
    if completion_channel is None:
        return 0
    reader = LayerCompletionRingReader(completion_channel, num_layers=num_layers)
    try:
        reader.wait_for(expected_layers, timeout_s=timeout_s)
    except TimeoutError as e:
        logger.warning(f"[layer-completion] {e}")
    logger.info(
        f"[layer-completion] drain: {reader.total_layers}/{expected_layers} layers, "
        f"{reader.completed_chunks} chunk(s) complete, {len(reader.chunks)} open, {reader.processed} messages"
    )
    return reader.total_layers
