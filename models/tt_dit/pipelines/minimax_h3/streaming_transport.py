# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Synchronous, one-chunk transport protocol; wire/authentication is caller-owned."""

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

from .streaming import Acknowledgement
from .streaming_encode import EncodedChunk


@dataclass(frozen=True)
class Delivery:
    session_id: str
    encoded: EncodedChunk

    @property
    def key(self):
        p = self.encoded
        return (self.session_id, p.index, p.sha256, p.video_pts, p.audio_pts, p.frames, p.samples)


@dataclass(frozen=True)
class ReadyReceipt:
    key: tuple
    played_frames: int
    client_ready: bool


class _Endpoint:
    def __init__(self, session_id, max_bytes):
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("bounded unique session identity required")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive integer byte bound required")
        self.session_id, self.max_bytes = session_id, max_bytes
        self.index = self.frames = self.samples = self.played = 0
        self.closed = self.busy = False

    def abort(self):
        self.closed = True

    @contextmanager
    def operation(self):
        if self.closed or self.busy:
            raise RuntimeError("closed or concurrent/reentrant transport")
        self.busy = True
        try:
            yield
        except BaseException:
            self.abort()  # Partial delivery cannot be undone or resumed implicitly.
            raise
        finally:
            self.busy = False

    def verify(self, delivery):
        if not isinstance(delivery, Delivery) or delivery.session_id != self.session_id:
            raise ValueError("wrong session")
        p = delivery.encoded
        if not isinstance(p, EncodedChunk) or type(p.data) is not bytes or not 0 < len(p.data) <= self.max_bytes:
            raise ValueError("immutable payload within byte cap required")
        if any(type(v) is not int or v < 0 for v in (p.index, p.video_pts, p.audio_pts, p.frames, p.samples)):
            raise ValueError("integer counters required")
        if (
            not p.frames
            or p.audio_pts != p.video_pts * 32000 // 24
            or p.samples != (p.video_pts + p.frames) * 32000 // 24 - p.audio_pts
            or hashlib.sha256(p.data).hexdigest() != p.sha256
        ):
            raise ValueError("invalid media timeline or digest")
        return delivery.key

    def expect_next(self, p):
        if (p.index, p.video_pts, p.audio_pts) != (self.index, self.frames, self.samples):
            raise ValueError("duplicate, gap or conflicting sequence")

    def advance(self, p, played):
        if self.closed:
            raise RuntimeError("transport aborted")
        if type(played) is not int or not self.played <= played <= self.frames + p.frames:
            raise ValueError("invalid client playhead")
        self.index += 1
        self.frames += p.frames
        self.samples += p.samples
        self.played = played


class TransportSink(_Endpoint):
    """Session sink: encode once, retry identical delivery until client readiness.

    exchange(delivery, deadline) uses an authenticated, size-bounded wire adapter
    and returns ReadyReceipt only from the client. It must honor the absolute
    monotonic deadline; a process timeout still bounds hung native/network calls.
    Retries stay inside this call. No re-render or retry of a failed Session.
    """

    def __init__(
        self,
        session_id,
        encoder,
        exchange,
        *,
        max_bytes,
        max_attempts=3,
        timeout_s=120,
        clock=time.monotonic,
        cancelled=lambda: False,
    ):
        super().__init__(session_id, max_bytes)
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("positive integer retry bound required")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("positive finite deadline required")
        self.encoder, self.exchange = encoder, exchange
        self.max_attempts, self.timeout_s = max_attempts, timeout_s
        self.clock, self.cancelled = clock, cancelled
        self.last_clock = None
        self.pending_bytes = 0

    def now(self):
        value = self.clock()
        if not math.isfinite(value) or (self.last_clock is not None and value < self.last_clock):
            raise ValueError("finite monotonic clock required")
        self.last_clock = value
        return value

    def check(self, deadline):
        if self.closed or self.cancelled():
            raise RuntimeError("transport aborted")
        if self.now() >= deadline:
            raise TimeoutError("client-ready deadline exceeded")

    def __call__(self, chunk):
        with self.operation():
            try:
                deadline = self.now() + self.timeout_s
                if not math.isfinite(deadline):
                    raise ValueError("finite absolute deadline required")
                self.check(deadline)
                p = self.encoder(chunk, cancelled=lambda: self.closed or self.cancelled() or self.now() >= deadline)
                delivery = Delivery(self.session_id, p)
                key = self.verify(delivery)
                self.expect_next(p)
                if (p.index, p.video_pts, p.audio_pts, p.frames, p.samples) != (
                    chunk.index,
                    chunk.video_pts,
                    chunk.audio_pts,
                    chunk.media.frames,
                    chunk.media.samples,
                ):
                    raise ValueError("encoder changed chunk identity")
                self.pending_bytes = len(p.data)
                for _ in range(self.max_attempts):
                    self.check(deadline)
                    try:
                        ack = self.exchange(delivery, deadline)
                    except (TimeoutError, ConnectionError):
                        continue
                    self.check(deadline)
                    if not isinstance(ack, ReadyReceipt) or ack.client_ready is not True or ack.key != key:
                        raise ValueError("missing, stale or conflicting client-ready receipt")
                    self.advance(p, ack.played_frames)
                    return Acknowledgement(p.index, p.frames, p.samples, ack.played_frames)
                raise TimeoutError("client-ready retry limit exceeded")
            finally:
                self.pending_bytes = 0


class ReadyClient(_Endpoint):
    """One ordered client: accept(payload) decodes/commits once and returns playhead.

    Keep this endpoint across connection retries. Only the latest receipt is
    cached (no media); earlier sequences are rejected. New streams need fresh IDs.
    The accept callback must bound decoder/playout memory and finish before its
    containing job deadline. An ambiguous partial append permanently closes it.
    """

    def __init__(self, session_id, accept, *, max_bytes):
        super().__init__(session_id, max_bytes)
        self.accept, self.last = accept, None

    def __call__(self, delivery):
        with self.operation():
            key = self.verify(delivery)
            if self.last is not None and self.last.key == key:
                return self.last
            p = delivery.encoded
            self.expect_next(p)
            played = self.accept(p)
            self.advance(p, played)
            self.last = ReadyReceipt(key, played, True)
            return self.last
