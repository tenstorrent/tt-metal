# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Consumer side of the layer-completion protocols (issue #54632).

Channel entry points for any consumer-side tool (prefill_producer, migration
drivers, verifiers) — dispatch on PREFILL_LAYER_COMPLETION_PROTOCOL under the
hood, so callers hold one `completion_channel` and never branch:

    channel = connect_layer_completion_channel(timeout_s)
    drain_layer_completions(channel, expected_layers)   # NUM_LAYERS per chunk

  Each protocol has its own scheduler-facing segment (scheduler_shm_name):
  protocol 1 (default): a counter channel — a bare count; the consumer
      correlates ticks with its own in-order chunk FIFO.
  protocol 2: a structured completion ring — self-describing messages
      drained work-conservingly by LayerCompletionDrainer.

LayerCompletionDrainer (v2) keys every chunk on the identity its messages
carry, (slot_id, pos_start, pos_end): the host transport stamps the runner's
request id into request_id while the D2H transport stamps a per-rank counter,
so request_id is diagnostic only. A chunk is complete when its disjoint layer
spans tile [0, num_layers); it is then evicted, so the same slot and positions
can come round again (slot reuse, multi-turn resumes).

* WORK-CONSERVING: a completion that is not yet actionable (the embedder's
  `can_process` predicate) goes to that chunk's side queue and never stalls the
  ring; side queues are retried after every processed message and whenever the
  ring runs dry.

* TEARDOWN INVARIANT: finish() raises listing any side-queued message that
  never became actionable.

Kept dependency-light at import (stdlib + loguru — ttnn and the
layer_completion bindings are imported lazily inside the connect helpers) so
the drainer is unit testable without the device stack. The v2 ring is any
object with `try_pop() -> tuple | None` in the v2 wire order
(seq, source_rank, request_id, slot_id, pos_start, pos_end, layer_start,
layer_end) — the ttnn._experimental.layer_completion.LayerCompletionQueueV2 binding qualifies.
"""

import os
import time
from collections import deque, namedtuple

from loguru import logger

# Wire order of LayerCompletionQueueV2.try_pop() (ttnn/cpp/ttnn-nanobind/layer_completion.cpp).
Completion = namedtuple("Completion", "seq source_rank request_id slot_id pos_start pos_end layer_start layer_end")


def current_protocol() -> int:
    """The job's completion protocol: PREFILL_LAYER_COMPLETION_PROTOCOL, 1 (default) or 2.

    Mirrors prefill_runner's read (kept separate so consumer-side tools need not import the
    device-heavy runner module).
    """
    try:
        protocol = int(os.environ.get("PREFILL_LAYER_COMPLETION_PROTOCOL", "1").strip())
    except ValueError:
        protocol = -1
    if protocol not in (1, 2):
        raise ValueError(
            f"PREFILL_LAYER_COMPLETION_PROTOCOL must be 1 or 2, got "
            f"{os.environ.get('PREFILL_LAYER_COMPLETION_PROTOCOL')!r}"
        )
    return protocol


def chunk_key(c: Completion) -> tuple:
    return (c.slot_id, c.pos_start, c.pos_end)


class ChunkCoverage:
    """One in-flight chunk: its disjoint layer spans so far and the side queue of
    not-yet-actionable completions."""

    __slots__ = ("key", "request_id", "intervals", "layers_accounted", "side_queue")

    def __init__(self, key: tuple, request_id: int):
        self.key = key
        self.request_id = request_id  # diagnostic: transports stamp different ids
        self.intervals = []  # sorted, disjoint [start, end)
        self.layers_accounted = 0
        self.side_queue = deque()

    def add_span(self, c: Completion, num_layers: int) -> None:
        """Raises on overlap or out-of-bounds: both are producer bugs."""
        start, end = c.layer_start, c.layer_end
        if not (0 <= start < end <= num_layers):
            raise ValueError(
                f"[drainer] chunk {self.key}: layer span [{start},{end}) out of bounds [0,{num_layers}): {c}"
            )
        for iv_start, iv_end in self.intervals:
            if start < iv_end and iv_start < end:
                raise ValueError(
                    f"[drainer] chunk {self.key}: layer span [{start},{end}) overlaps already-covered "
                    f"[{iv_start},{iv_end}): {c}"
                )
        self.intervals.append((start, end))
        self.intervals.sort()
        self.layers_accounted += end - start

    def is_complete(self, num_layers: int) -> bool:
        return self.layers_accounted == num_layers


class LayerCompletionDrainer:
    """Work-conserving drainer for a v2 structured completion ring.

    Args:
        ring: anything with try_pop() -> tuple | None in the v2 wire order.
        num_layers: global ack-layer count per chunk: the model's layers plus the MTP /
            DFlash rows acked past them (the coverage bound).
        can_process(completion) -> bool: readiness predicate; blocked messages are
            side-queued per chunk instead of stalling the ring. Default: always ready.
        on_completion(completion): action for each processed message. Default: none.
        on_chunk_complete(key, coverage): fired when a chunk tiles [0, num_layers), just
            before it is evicted. Default: none.
        poll_idle_s: sleep quantum for drain_blocking when nothing is actionable.
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
            cov = self._chunks[key] = ChunkCoverage(key, c.request_id)
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
                    f"[drainer] chunk {cov.key} completed with blocked completions: {list(cov.side_queue)}"
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
            raise RuntimeError(f"[drainer] finish() with side-queued (never-actionable) completions: {detail}")
        return self.total_layers

    def drain_blocking(self, expected_ack_layers: int, timeout_s: float = 600.0) -> int:
        """Loop until `expected_ack_layers` (span-summed) have been processed, then finish().
        Raises TimeoutError with a coverage snapshot."""
        deadline = time.perf_counter() + timeout_s
        while self.total_layers < expected_ack_layers:
            if not self.step():
                if time.perf_counter() > deadline:
                    snapshot = ", ".join(
                        f"chunk {key}: {cov.layers_accounted}/{self._num_layers} layers"
                        + (f" (+{len(cov.side_queue)} blocked)" if cov.side_queue else "")
                        for key, cov in self._chunks.items()
                    )
                    raise TimeoutError(
                        f"[drainer] timed out at {self.total_layers}/{expected_ack_layers} layers "
                        f"after {timeout_s}s; open chunks: [{snapshot}]"
                    )
                time.sleep(self._poll_idle_s)
        return self.finish()


# ---------------------------------------------------------------------------
# Channel connect/drain — protocol dispatch under the hood, so consumers hold one
# `completion_channel` and never branch on PREFILL_LAYER_COMPLETION_PROTOCOL.
# ---------------------------------------------------------------------------


def scheduler_shm_name(service_id: str, protocol: int) -> str:
    """The scheduler-facing segment for a protocol. Distinct per protocol: the v1 counter channel
    validates nothing on attach, so sharing one name would let a v1 consumer corrupt a v2 ring."""
    if protocol == 2:
        return f"/tt_prefill_layer_completions_{service_id}"
    return f"/tt_prefill_layer_acks_{service_id}"


def _connect_layer_ack_channel(timeout_s: int):
    """v1: attach (consumer side) to the scheduler-facing counter channel. None if unavailable."""
    import ttnn

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = scheduler_shm_name(service_id, 1)
    try:
        channel = ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[layer-completion] could not connect counter channel {shm_name}: {e}; skipping drain.")
        return None
    logger.info(f"[layer-completion] connected counter channel {shm_name}")
    return channel


def _connect_layer_completion_ring(timeout_s: int):
    """v2: attach (consumer side) to the master router's structured completion ring. None if unavailable."""
    from ttnn._experimental.layer_completion import LayerCompletionQueueV2

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = scheduler_shm_name(service_id, 2)
    try:
        ring = LayerCompletionQueueV2.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[layer-completion] could not connect completion ring {shm_name}: {e}; skipping drain.")
        return None
    logger.info(f"[layer-completion] connected completion ring {shm_name}")
    return ring


def connect_layer_completion_channel(timeout_s: int):
    """Attach (consumer side) to this job's layer-completion channel: protocol 1 → the
    counter channel; 2 → the structured ring. None if unavailable."""
    if current_protocol() == 2:
        return _connect_layer_completion_ring(timeout_s)
    return _connect_layer_ack_channel(timeout_s)


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
    """v1: block until `expected` per-layer acks (a bare count) are drained, or timeout.
    Returns the count actually drained."""
    if ack_channel is None:
        return 0
    drained = 0
    last_logged = -1
    start = time.perf_counter()
    while drained < expected:
        drained += ack_channel.try_consume_all()
        if drained != last_logged:
            logger.info(f"[layer-completion] layer acks {drained}/{expected}")
            last_logged = drained
        if drained >= expected:
            break
        if time.perf_counter() - start > timeout_s:
            logger.warning(f"[layer-completion] timed out at {drained}/{expected} acks after {timeout_s}s")
            break
        time.sleep(0.01)
    logger.info(f"[layer-completion] drained {drained}/{expected} layer acks in {(time.perf_counter() - start):.2f}s")
    return drained


def _drain_layer_completion_ring(
    completion_ring, expected_ack_layers: int, *, num_layers: int, timeout_s: float
) -> int:
    """v2: work-conserving drain until `expected_ack_layers` (span-summed) are accounted; a
    producer-side inconsistency (overlap, out of bounds) raises."""
    if completion_ring is None:
        return 0
    drainer = LayerCompletionDrainer(completion_ring, num_layers=num_layers)
    try:
        drainer.drain_blocking(expected_ack_layers, timeout_s=timeout_s)
    except TimeoutError as e:
        logger.warning(f"[layer-completion] {e}")
    logger.info(
        f"[layer-completion] v2 drain: {drainer.total_layers}/{expected_ack_layers} layers, "
        f"{drainer.completed_chunks} chunk(s) complete, {len(drainer.chunks)} open, {drainer.processed} messages"
    )
    return drainer.total_layers


def drain_layer_completions(
    completion_channel, expected_ack_layers: int, *, num_layers: int, timeout_s: float = 600.0
) -> int:
    """Drain `expected_ack_layers` per-layer completions from the channel
    connect_layer_completion_channel() returned. `num_layers` is the global ack-layer count per
    chunk (the v2 coverage bound; unused by v1)."""
    if current_protocol() == 2:
        return _drain_layer_completion_ring(
            completion_channel, expected_ack_layers, num_layers=num_layers, timeout_s=timeout_s
        )
    return _drain_layer_acks(completion_channel, expected_ack_layers, timeout_s)
