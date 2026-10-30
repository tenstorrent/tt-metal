# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Consumer side of the layer-completion ring for host tools (prefill_producer, migration drivers, verifiers):

    channel = connect_layer_completion_channel(timeout_s)
    drain = start_layer_completion_drain(channel, num_layers=..., chunk_layers=...)
    ...submit the chunks...
    wait_for_layer_completions(drain, expected_layers)
    drain.stop()

LayerCompletionRingReader keys each chunk on (slot_id, pos_start, pos_end). Spans must be disjoint and lie in
[0, num_layers); a chunk is complete once they cover chunk_layers layers, and is then evicted, so the same key can
come round again. A hybrid model covers fewer layers than the global bound. A completion that `can_process` rejects
waits in its chunk's side queue instead of stalling the ring; finish() raises on anything still waiting.

The rings are bounded, so a caller submitting input must drain concurrently: LayerCompletionDrain runs the reader
on a thread, which progresses while the caller blocks in an H2D call (those release the GIL).

ttnn is imported lazily inside the connect helper, so the reader is unit-testable without the device stack. The
ring is any object with `try_pop() -> tuple | None` in the wire order of LayerCompletionQueue.try_pop().
"""

import os
import threading
import time
from collections import deque, namedtuple

from loguru import logger

# Wire order of LayerCompletionQueue.try_pop().
Completion = namedtuple("Completion", "slot_id pos_start pos_end layer_start layer_end host_timestamp_ns")


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

    def is_complete(self, chunk_layers: int) -> bool:
        return self.layers_accounted == chunk_layers


class LayerCompletionRingReader:
    """Work-conserving reader of the scheduler-facing completion ring.

    Args:
        ring: anything with try_pop() -> tuple | None in the wire order.
        num_layers: global ack-layer axis: the model's layers plus the MTP / DFlash rows acked
            past them. Every span must lie in [0, num_layers).
        chunk_layers: layers one chunk covers when complete (the ring header's
            completions_per_chunk). Default: num_layers. A hybrid stack acks only its
            KV-writing layers, so it covers fewer than num_layers.
        can_process(completion) -> bool: readiness predicate; blocked messages are
            side-queued per chunk instead of stalling the ring. Default: always ready.
        on_completion(completion): action for each processed message. Default: none.
        on_chunk_complete(key, coverage): fired when a chunk covers chunk_layers layers,
            just before it is evicted. Default: none.
        poll_idle_s: sleep quantum for wait_for when nothing is actionable.
    """

    def __init__(
        self,
        ring,
        *,
        num_layers: int,
        chunk_layers: int | None = None,
        can_process=None,
        on_completion=None,
        on_chunk_complete=None,
        poll_idle_s: float = 0.001,
    ):
        self._ring = ring
        self._num_layers = num_layers
        self._chunk_layers = num_layers if chunk_layers is None else chunk_layers
        if not 0 < self._chunk_layers <= num_layers:
            raise ValueError(f"[ring-reader] chunk_layers={self._chunk_layers} outside (0, {num_layers}]")
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
        if cov.is_complete(self._chunk_layers):
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
                        f"chunk {key}: {cov.layers_accounted}/{self._chunk_layers} layers"
                        + (f" (+{len(cov.side_queue)} blocked)" if cov.side_queue else "")
                        for key, cov in self._chunks.items()
                    )
                    raise TimeoutError(
                        f"[ring-reader] timed out at {self.total_layers}/{expected_layers} layers "
                        f"after {timeout_s}s; open chunks: [{snapshot}]"
                    )
                time.sleep(self._poll_idle_s)
        return self.finish()


class LayerCompletionDrain:
    """Runs a LayerCompletionRingReader on a background thread from start() to finish(), so the runner never
    blocks on a full ring while the caller is submitting or waiting on H2D input."""

    def __init__(self, ring, *, num_layers: int, chunk_layers: int | None = None, poll_idle_s: float = 0.001):
        self.reader = LayerCompletionRingReader(
            ring, num_layers=num_layers, chunk_layers=chunk_layers, poll_idle_s=poll_idle_s
        )
        self._poll_idle_s = poll_idle_s
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="layer-completion-drain", daemon=True)

    def start(self) -> "LayerCompletionDrain":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self.reader.step():
                    time.sleep(self._poll_idle_s)
        except BaseException as e:  # surfaced on the caller's thread by wait_for() / stop()
            self._error = e

    def wait_for(self, expected_layers: int, timeout_s: float = 600.0) -> int:
        """Block until `expected_layers` (span-summed, cumulative since start()) are processed."""
        deadline = time.perf_counter() + timeout_s
        while self.reader.total_layers < expected_layers:
            if self._error is not None:
                raise self._error
            if not self._thread.is_alive():
                raise RuntimeError("[ring-reader] drain thread stopped before the expected completions arrived")
            if time.perf_counter() > deadline:
                snapshot = ", ".join(
                    f"chunk {key}: {cov.layers_accounted}/{self.reader._chunk_layers} layers"
                    for key, cov in list(self.reader.chunks.items())
                )
                raise TimeoutError(
                    f"[ring-reader] timed out at {self.reader.total_layers}/{expected_layers} layers "
                    f"after {timeout_s}s; open chunks: [{snapshot}]"
                )
            time.sleep(0.01)
        return self.reader.total_layers

    def stop(self) -> int:
        """Join the thread, rethrow its failure, then check the teardown invariant. Idempotent."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        if self._error is not None:
            raise self._error
        return self.reader.finish()


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


def start_layer_completion_drain(completion_channel, *, num_layers: int, chunk_layers: int):
    """A started LayerCompletionDrain on the channel, or None when there is no channel."""
    if completion_channel is None:
        return None
    return LayerCompletionDrain(completion_channel, num_layers=num_layers, chunk_layers=chunk_layers).start()


def wait_for_layer_completions(drain, expected_layers: int, *, timeout_s: float = 600.0) -> int:
    """Wait on a running drain until `expected_layers` (span-summed, cumulative) are accounted; a
    producer-side inconsistency (overlap, out of bounds) raises. A timeout is logged, not raised."""
    if drain is None:
        return 0
    reader = drain.reader
    try:
        drain.wait_for(expected_layers, timeout_s=timeout_s)
    except TimeoutError as e:
        logger.warning(f"[layer-completion] {e}")
    logger.info(
        f"[layer-completion] drain: {reader.total_layers}/{expected_layers} layers, "
        f"{reader.completed_chunks} chunk(s) complete, {len(reader.chunks)} open, {reader.processed} messages"
    )
    return reader.total_layers
