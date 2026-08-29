# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Consumer side of the layer-completion protocols (issue #54632).

Channel entry points for any consumer-side tool (prefill_producer, migration
drivers, verifiers) — dispatch on PREFILL_LAYER_COMPLETION_PROTOCOL under the
hood, so callers hold one `completion_channel` and never branch:

    channel = connect_layer_completion_channel(timeout_s)
    drain_layer_completions(channel, expected_layers)   # NUM_LAYERS per chunk

  Both protocols share the ONE scheduler-facing shm name
  (/tt_prefill_layer_acks_<service_id>); the protocol decides the segment's
  layout:
  protocol 1 (default): a counter channel — a bare count; the consumer
      correlates ticks with its own in-order chunk FIFO.
  protocol 2: a structured completion ring — self-describing messages
      drained work-conservingly by LayerCompletionDrainer.

LayerCompletionDrainer (v2):

* WORK-CONSERVING: a completion that is not yet actionable (the embedder's
  `can_process` predicate, e.g. "migration context for this slot exists") goes
  to a per-request SIDE QUEUE and never stalls the input ring — later requests'
  completions keep advancing. Side queues are retried after every processed
  message and whenever the ring runs dry.

* PER-REQUEST COVERAGE: tracks each request's completed layer spans. A request
  is fully covered when its spans total `num_layers` — with overlap enforced
  as an error at insert, that implies an exact tiling of [0, num_layers).

* EXPECTATION HOOK (future error detection): `on_first_completion` fires once
  per request with the first message seen. The embedder may use it to register
  a coverage rule on `RequestCoverage.expectation` — e.g. for pipelined
  prefill, "every layer eventually spans for the position range of the first
  completion received" — evaluated by a future checker. Nothing is enforced
  yet; this is the seam.

* TEARDOWN INVARIANT: finish() requires every side queue to be empty and
  raises listing the stranded (never-actionable) messages otherwise — a lost
  dependency surfaces as a precise, attributed error, not a silent stall.

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
Completion = namedtuple(
    "Completion", "seq source_rank request_id slot_id pos_start pos_end layer_start layer_end"
)


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


class RequestCoverage:
    """Per-request layer-coverage bookkeeping: disjoint spans seen so far, the chunk's
    slot/position identity (cross-checked on every message), and the side queue of
    not-yet-actionable completions for this request."""

    __slots__ = ("request_id", "slot_id", "pos_start", "pos_end", "intervals", "layers_accounted", "side_queue", "expectation")

    def __init__(self, request_id: int):
        self.request_id = request_id
        self.slot_id = None
        self.pos_start = None
        self.pos_end = None
        self.intervals = []  # sorted, disjoint [start, end) — overlap is a producer bug
        self.layers_accounted = 0
        self.side_queue = deque()
        # Registered by the embedder's on_first_completion hook; evaluated by future
        # error-detection rules. Shape for pipelined prefill: "layers eventually tile
        # [0, num_layers) for THIS request's position range".
        self.expectation = None

    def record_identity(self, c: Completion) -> None:
        """Bind slot/pos from the first message of the request; cross-check afterwards."""
        if self.slot_id is None:
            self.slot_id, self.pos_start, self.pos_end = c.slot_id, c.pos_start, c.pos_end
        elif (self.slot_id, self.pos_start, self.pos_end) != (c.slot_id, c.pos_start, c.pos_end):
            raise ValueError(
                f"[drainer] request {self.request_id}: inconsistent identity — first message had "
                f"slot={self.slot_id} pos=[{self.pos_start},{self.pos_end}), now {c}: "
                "producer-side correlation bug"
            )

    def add_span(self, c: Completion, num_layers: int) -> None:
        """Record [c.layer_start, c.layer_end). Raises on overlap or out-of-bounds — both
        are producer bugs, and the self-describing payload makes them attributable."""
        start, end = c.layer_start, c.layer_end
        if not (0 <= start < end <= num_layers):
            raise ValueError(
                f"[drainer] request {c.request_id}: layer span [{start},{end}) out of bounds "
                f"[0,{num_layers}): {c}"
            )
        for iv_start, iv_end in self.intervals:
            if start < iv_end and iv_start < end:
                raise ValueError(
                    f"[drainer] request {c.request_id}: layer span [{start},{end}) overlaps "
                    f"already-covered [{iv_start},{iv_end}): {c}"
                )
        self.intervals.append((start, end))
        self.intervals.sort()
        self.layers_accounted += end - start

    def is_complete(self, num_layers: int) -> bool:
        # Disjointness is enforced at insert, so reaching the full layer count within
        # [0, num_layers) implies an exact tiling.
        return self.layers_accounted == num_layers


class LayerCompletionDrainer:
    """Work-conserving drainer for a v2 structured completion ring.

    Args:
        ring: anything with try_pop() -> tuple | None in the v2 wire order.
        num_layers: GLOBAL model layer count (coverage bound per request).
        can_process(completion) -> bool: readiness predicate; blocked messages are
            side-queued (per request) instead of stalling the ring. Default: always ready.
        on_completion(completion): action for each processed message (e.g. issue a
            migration). Default: coverage accounting only.
        on_first_completion(request_id, completion, coverage): expectation hook, fired
            once per request (see RequestCoverage.expectation). Default: no-op.
        on_request_complete(request_id, coverage): notification when a request tiles
            [0, num_layers). Default: no-op.
        poll_idle_s: sleep quantum for drain_blocking when nothing is actionable.
    """

    def __init__(
        self,
        ring,
        *,
        num_layers: int,
        can_process=None,
        on_completion=None,
        on_first_completion=None,
        on_request_complete=None,
        poll_idle_s: float = 0.001,
    ):
        self._ring = ring
        self._num_layers = num_layers
        self._can_process = can_process or (lambda c: True)
        self._on_completion = on_completion or (lambda c: None)
        self._on_first_completion = on_first_completion or (lambda rid, c, cov: None)
        self._on_request_complete = on_request_complete or (lambda rid, cov: None)
        self._poll_idle_s = poll_idle_s
        self._requests = {}  # request_id -> RequestCoverage
        self.total_layers = 0  # span-summed layers processed (NOT message count)
        self.processed = 0
        self.side_queued = 0

    @property
    def requests(self):
        return self._requests

    def _coverage(self, request_id: int) -> RequestCoverage:
        cov = self._requests.get(request_id)
        if cov is None:
            cov = self._requests[request_id] = RequestCoverage(request_id)
        return cov

    def _process(self, c: Completion) -> None:
        cov = self._coverage(c.request_id)
        first = cov.layers_accounted == 0 and not cov.intervals and cov.slot_id is None
        cov.record_identity(c)
        cov.add_span(c, self._num_layers)
        if first:
            self._on_first_completion(c.request_id, c, cov)
        self._on_completion(c)
        self.total_layers += c.layer_end - c.layer_start
        self.processed += 1
        if cov.is_complete(self._num_layers):
            self._on_request_complete(c.request_id, cov)

    def _retry_side_queues(self) -> bool:
        """Re-test side-queued messages (oldest request first — the only ordering
        preference the consumer keeps). Returns True if any became actionable."""
        progressed = False
        for request_id in sorted(self._requests):
            sq = self._requests[request_id].side_queue
            while sq and self._can_process(sq[0]):
                self._process(sq.popleft())
                progressed = True
        return progressed

    def step(self) -> bool:
        """One work-conserving iteration. Returns True if anything moved (popped,
        processed, or unblocked); False means idle-safe to sleep."""
        msg = self._ring.try_pop()
        if msg is None:
            return self._retry_side_queues()
        c = Completion._make(msg)
        if self._can_process(c):
            self._process(c)
            # Processing may have unblocked side-queued messages (embedder state change).
            self._retry_side_queues()
        else:
            cov = self._coverage(c.request_id)
            cov.record_identity(c)  # identity is known (and checked) even while blocked
            cov.side_queue.append(c)
            self.side_queued += 1
        return True

    def finish(self) -> int:
        """Teardown invariant: every side queue must be empty — a stranded message is a
        lost dependency, reported with its full payload. Returns total layers drained."""
        stranded = [
            (request_id, list(cov.side_queue))
            for request_id, cov in self._requests.items()
            if cov.side_queue
        ]
        if stranded:
            detail = "; ".join(f"request {rid}: {msgs}" for rid, msgs in stranded)
            raise RuntimeError(f"[drainer] finish() with side-queued (never-actionable) completions: {detail}")
        return self.total_layers

    def drain_blocking(self, expected_total_layers: int, timeout_s: float = 600.0) -> int:
        """Work-conserving loop until `expected_total_layers` (span-summed) have been
        processed, then finish(). Raises TimeoutError with a coverage snapshot."""
        deadline = time.perf_counter() + timeout_s
        while self.total_layers < expected_total_layers:
            if not self.step():
                if time.perf_counter() > deadline:
                    snapshot = ", ".join(
                        f"req {rid}: {cov.layers_accounted}/{self._num_layers} layers"
                        + (f" (+{len(cov.side_queue)} blocked)" if cov.side_queue else "")
                        for rid, cov in sorted(self._requests.items())
                    )
                    raise TimeoutError(
                        f"[drainer] timed out at {self.total_layers}/{expected_total_layers} layers "
                        f"after {timeout_s}s; coverage: [{snapshot}]"
                    )
                time.sleep(self._poll_idle_s)
        return self.finish()


# ---------------------------------------------------------------------------
# Channel connect/drain — protocol dispatch under the hood, so consumers hold one
# `completion_channel` and never branch on PREFILL_LAYER_COMPLETION_PROTOCOL.
# ---------------------------------------------------------------------------


def _connect_layer_ack_channel(timeout_s: int):
    """v1: attach (consumer side) to the scheduler-facing counter channel
    (/tt_prefill_layer_acks_<service_id>). None if unavailable."""
    import ttnn

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = f"/tt_prefill_layer_acks_{service_id}"
    try:
        channel = ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[layer-completion] could not connect counter channel {shm_name}: {e}; skipping drain.")
        return None
    logger.info(f"[layer-completion] connected counter channel {shm_name}")
    return channel


def _connect_layer_completion_ring(timeout_s: int):
    """v2: attach (consumer side) to the master router's structured completion ring
    (/tt_prefill_layer_acks_<service_id> — one name for both protocols). None if unavailable."""
    from ttnn._experimental.layer_completion import LayerCompletionQueueV2

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = f"/tt_prefill_layer_acks_{service_id}"
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


def _drain_layer_completion_ring(completion_ring, expected_layers: int, timeout_s: float = 600.0) -> int:
    """v2: work-conserving drain of the structured ring until `expected_layers` (span-summed)
    are accounted, validating per-request coverage along the way."""
    if completion_ring is None:
        return 0
    num_layers = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
    drainer = LayerCompletionDrainer(completion_ring, num_layers=num_layers)
    try:
        drainer.drain_blocking(expected_layers, timeout_s=timeout_s)
    except TimeoutError as e:
        logger.warning(f"[layer-completion] {e}")
    logger.info(
        f"[layer-completion] v2 drain: {drainer.total_layers}/{expected_layers} layers across "
        f"{len(drainer.requests)} request(s), {drainer.processed} messages"
    )
    return drainer.total_layers


def drain_layer_completions(completion_channel, expected_layers: int, timeout_s: float = 600.0) -> int:
    """Drain `expected_layers` (num_layers per chunk) of per-layer completions from the
    channel connect_layer_completion_channel() returned."""
    if current_protocol() == 2:
        return _drain_layer_completion_ring(completion_channel, expected_layers, timeout_s)
    return _drain_layer_acks(completion_channel, expected_layers, timeout_s)
