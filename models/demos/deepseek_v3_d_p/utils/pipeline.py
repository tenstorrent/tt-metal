# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two-chunk pipeline (E3) driver primitives: run two chunk forwards ("streams") with an interleaved op ISSUE order.

The device executes ops in the order the host issues them (per sub-device), so overlapping two chunks needs nothing
more than interleaving the two forwards' op issue at chosen points. Each stream's forward runs unmodified in its own
Python thread; a baton guarantees that exactly one thread issues ttnn ops at any moment, and the model calls
``pipeline_yield(tag)`` at the points where the schedule may hand the baton to the other stream. With no active
runner, ``pipeline_yield`` is a no-op and ``current_stream()`` is 0, so the single-stream model is unchanged.

Stream identity is thread-local; TT_CCL uses it to keep per-stream copies of resources whose lifetime spans a
switch point (see TT_CCL._STREAM_LOCAL).

Ordering contract (what makes the interleave correct):
  * stream 1 (chunk c+1) must not run layer L's attention before stream 0 (chunk c) has issued its layer-L KV write
    (layer L attends to chunk c's layer-L KV); the policies below keep stream 1 at least half a layer behind;
  * anything a stream frees before a switch point must not still be used by one of its ops running on ANOTHER
    sub-device (the other stream may be handed that address); switch points sit where today's sub-device joins are.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

_tls = threading.local()
_active: Optional["TwoStreamRunner"] = None


def current_stream() -> int:
    """0 for the main (or only) stream, 1 for the lagging stream of an active two-stream run."""
    return getattr(_tls, "stream", 0)


def pipeline_active() -> bool:
    return _active is not None


def pipeline_yield(tag: str) -> None:
    """Switch point. No-op unless a TwoStreamRunner is running and its policy says to hand over at `tag`."""
    runner = _active
    if runner is None or getattr(_tls, "runner", None) is not runner:
        return
    runner._maybe_switch(current_stream(), tag)


class PipelineAbort(RuntimeError):
    """Raised in a stream when the other stream failed, so both threads unwind."""


# Policies: (stream, tag, state) -> bool "hand the baton to the other stream now".
def policy_stub(stream: int, tag: str, state: dict) -> bool:
    """Layer granularity, no concurrency: s0 layer L, s1 layer L, s0 layer L+1, ... (s1 starts after s0's layer 0)."""
    return tag == "layer_end"


def policy_pipe(stream: int, tag: str, state: dict) -> bool:
    """Half-layer offset: s0 hands over right after issuing its MoE dispatch window (s1 then issues its layer-L
    attention while s0's dispatch runs on the comm sub-device), s1 hands back at the end of its attention, and so on
    with the roles swapped. Needs the persistent manager (E1) and dispatch on its own sub-device (E1b)."""
    if tag == "moe_window_open":
        return True
    if tag == "attn_end":
        # Stream 1 starts with no MoE to overlap; let it run attention(0) then hand back to stream 0.
        return True
    return False


POLICIES = {"stub": policy_stub, "pipe": policy_pipe}


class TwoStreamRunner:
    """Run fn0 (stream 0) and fn1 (stream 1) with one baton. Stream 0 starts; each pipeline_yield the policy accepts
    hands over; a finished stream hands over for good. Returns (ret0, ret1); re-raises the first exception."""

    def __init__(self, policy: Callable[[int, str, dict], bool] | str = "stub"):
        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        self._cv = threading.Condition()
        self._turn = 0
        self._done = [False, False]
        self._error: Optional[BaseException] = None
        self.state: dict = {}
        self.switches = 0
        self.trace: list = []  # (stream, tag) of every accepted switch, for tests / debugging

    def _maybe_switch(self, stream: int, tag: str) -> None:
        if self._done[1 - stream]:
            return  # the other stream already finished: keep going
        if not self.policy(stream, tag, self.state):
            return
        with self._cv:
            self.trace.append((stream, tag))
            self.switches += 1
            self._turn = 1 - stream
            self._cv.notify_all()
            self._cv.wait_for(lambda: self._turn == stream or self._error is not None or self._done[1 - stream])
            if self._error is not None:
                raise PipelineAbort("other pipeline stream failed") from self._error

    def _thread_main(self, stream: int, fn: Callable, out: list) -> None:
        _tls.stream = stream
        _tls.runner = self
        try:
            with self._cv:
                self._cv.wait_for(lambda: self._turn == stream or self._error is not None)
                if self._error is not None:
                    raise PipelineAbort("other pipeline stream failed") from self._error
            out[stream] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in run()
            with self._cv:
                if self._error is None and not isinstance(e, PipelineAbort):
                    self._error = e
                self._cv.notify_all()
        finally:
            with self._cv:
                self._done[stream] = True
                self._turn = 1 - stream
                self._cv.notify_all()
            _tls.stream = 0
            _tls.runner = None

    def run(self, fn0: Callable, fn1: Callable):
        global _active
        assert _active is None, "nested TwoStreamRunner"
        out = [None, None]
        _active = self
        try:
            threads = [threading.Thread(target=self._thread_main, args=(s, f, out), name=f"pipe-s{s}")
                       for s, f in ((0, fn0), (1, fn1))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            _active = None
        if self._error is not None:
            raise self._error
        return out[0], out[1]
