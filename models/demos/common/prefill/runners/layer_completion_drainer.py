# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections import deque, namedtuple

from loguru import logger

Completion = namedtuple(
    "Completion", "seq source_rank request_id slot_id pos_start pos_end layer_start layer_end host_ts_ns"
)


def current_protocol() -> int:
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
    __slots__ = (
        "request_id",
        "slot_id",
        "pos_start",
        "pos_end",
        "intervals",
        "layers_accounted",
        "side_queue",
        "expectation",
    )

    def __init__(self, request_id: int):
        self.request_id = request_id
        self.slot_id = None
        self.pos_start = None
        self.pos_end = None
        self.intervals = []
        self.layers_accounted = 0
        self.side_queue = deque()
        self.expectation = None

    def record_identity(self, c: Completion) -> None:
        if self.slot_id is None:
            self.slot_id, self.pos_start, self.pos_end = c.slot_id, c.pos_start, c.pos_end
        elif (self.slot_id, self.pos_start, self.pos_end) != (c.slot_id, c.pos_start, c.pos_end):
            raise ValueError(
                f"[drainer] request {self.request_id}: inconsistent identity — first message had "
                f"slot={self.slot_id} pos=[{self.pos_start},{self.pos_end}), now {c}: "
                "producer-side correlation bug"
            )

    def add_span(self, c: Completion, num_layers: int) -> None:
        start, end = c.layer_start, c.layer_end
        if not (0 <= start < end <= num_layers):
            raise ValueError(
                f"[drainer] request {c.request_id}: layer span [{start},{end}) out of bounds " f"[0,{num_layers}): {c}"
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
        return self.layers_accounted == num_layers


class LayerCompletionDrainer:
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
        self._requests = {}
        self.total_layers = 0
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
        progressed = False
        for request_id in sorted(self._requests):
            sq = self._requests[request_id].side_queue
            while sq and self._can_process(sq[0]):
                self._process(sq.popleft())
                progressed = True
        return progressed

    def step(self) -> bool:
        msg = self._ring.try_pop()
        if msg is None:
            return self._retry_side_queues()
        c = Completion._make(msg)
        if self._can_process(c):
            self._process(c)
            self._retry_side_queues()
        else:
            cov = self._coverage(c.request_id)
            cov.record_identity(c)
            cov.side_queue.append(c)
            self.side_queued += 1
        return True

    def finish(self) -> int:
        stranded = [(request_id, list(cov.side_queue)) for request_id, cov in self._requests.items() if cov.side_queue]
        if stranded:
            detail = "; ".join(f"request {rid}: {msgs}" for rid, msgs in stranded)
            raise RuntimeError(f"[drainer] finish() with side-queued (never-actionable) completions: {detail}")
        return self.total_layers

    def drain_blocking(self, expected_total_layers: int, timeout_s: float = 600.0) -> int:
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


def _connect_layer_ack_channel(timeout_s: int):
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
    if current_protocol() == 2:
        return _connect_layer_completion_ring(timeout_s)
    return _connect_layer_ack_channel(timeout_s)


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
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
    if current_protocol() == 2:
        return _drain_layer_completion_ring(completion_channel, expected_layers, timeout_s)
    return _drain_layer_acks(completion_channel, expected_layers, timeout_s)
