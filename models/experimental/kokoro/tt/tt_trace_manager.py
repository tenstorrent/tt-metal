# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal-trace capture/replay manager for the Kokoro pipeline (Phase 3 orchestration).

The Kokoro forward splits at the duration readback into two fixed-shape device graphs — Trace A
(``input_ids -> ... -> dur_clipped``) and Trace B (``alignment -> ... -> audio``) — with a host step
(duration readback + alignment build) between them. Each graph is metal-trace-capturable once its
shape bucket is fixed (see docs/generator_perf_optimizations.md; the device graph is trace-clean via
``tt_trace_prep``).

:class:`TraceManager` captures a graph the first time it sees a shape ``key`` and replays it on every
later call with that key. Inputs are held in **persistent device buffers** (fixed addresses the trace
reads); each replay first copies the call's fresh input tensors into those buffers (device->device
``ttnn.copy``), then ``execute_trace``. The output lives in a persistent buffer the replay overwrites.

Usage::

    tm = TraceManager(device)
    audio = tm.run("B/T=748", {"asr": asr_tt, "F0": f0_tt, ...}, forward_fn)
    # forward_fn(persistent: dict[str, ttnn.Tensor]) -> ttnn.Tensor
    # It must read inputs from the dict and NOT deallocate them (clone if the graph consumes them).
    ...
    tm.release()  # before closing the device
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import ttnn

from .tt_trace_prep import clear_trace_weight_prep_cache, set_trace_weight_prep

_DBG = os.environ.get("KOKORO_TRACE_DEBUG") == "1"


def _dbg(msg: str) -> None:
    if _DBG:
        print(f"  [TM] {msg}", flush=True)


@dataclass
class _Entry:
    tid: int
    persistent: dict  # name -> persistent input buffer (fixed address)
    output: object  # persistent output buffer(s) overwritten by each replay; Tensor or tuple[Tensor]
    multi: bool = False  # True if forward_fn returned a tuple/list of tensors


class TraceManager:
    """Per-shape-bucket metal-trace capture/replay with persistent input buffers."""

    def __init__(self, device: ttnn.Device, *, cq_id: int = 0, sync_replay: bool = False) -> None:
        self.device = device
        self.cq_id = cq_id
        # When this trace runs on a non-default CQ, its input tensors are produced/copied on another
        # CQ; a full device sync before execute orders those cross-CQ writes ahead of the replay read.
        # Only needed for the multi-CQ two-trace path; the default (single-CQ) path leaves this off.
        self.sync_replay = sync_replay
        self._entries: dict = {}
        self._pending: dict = {}  # key -> persistent buffers allocated by prepare(), awaiting capture()
        self._prep_enabled = False
        self.captures = 0  # number of first-time captures (cache misses)
        self.replays = 0  # number of trace replays (cache hits)

    def _ensure_prep(self) -> None:
        # Conv/BERT/LSTM weight+id uploads must be cached (not re-uploaded) for capture to be
        # write-free. Enabled once and kept on; the cache persists until release().
        if not self._prep_enabled:
            set_trace_weight_prep(True)
            self._prep_enabled = True

    def has(self, key) -> bool:
        return key in self._entries

    @staticmethod
    def _free(o) -> None:
        if isinstance(o, (tuple, list)):
            for t in o:
                ttnn.deallocate(t)
        else:
            ttnn.deallocate(o)

    def prepare(self, key, inputs: dict, forward_fn: Callable[[dict], object]) -> object:
        """Phase 1 of a multi-trace capture: allocate ``key``'s persistent input buffers and run
        ``forward_fn`` EAGERLY once (compile programs + populate weight-prep caches). No trace is
        captured. Returns the eager (warm) output so a chained caller can read it (e.g. a duration
        readback that sizes the next trace); the CALLER owns and frees that output.

        Critical for coexisting traces: every persistent buffer a trace's replay reads must be
        allocated while **no** trace is resident. Doing all ``prepare`` calls before any ``capture``
        guarantees that — a buffer allocated while another trace is live can land on that trace's
        freed-intermediate addresses and be clobbered when it replays (manifested as a decoder-replay
        hang). See :meth:`capture`.
        """
        with ttnn.command_queue(self.cq_id):
            self._ensure_prep()
            _dbg(f"prepare key={key!r} cq={self.cq_id}: clone {len(inputs)} inputs + eager warmup")
            persistent = {name: ttnn.clone(t) for name, t in inputs.items()}
            warm = forward_fn(persistent)
            ttnn.synchronize_device(self.device)
            self._pending[key] = persistent
            return warm

    def capture(self, key, forward_fn: Callable[[dict], object]) -> None:
        """Phase 2 of a multi-trace capture: RECORD ``key``'s graph into a metal trace, reading the
        persistent buffers allocated by :meth:`prepare`. Does NOT execute (call :meth:`execute`).
        Only trace intermediates are allocated here (during capture, which the framework marks
        allocation-safe, and they are freed before the trace runs) — no persistent buffer is created
        while this or any other trace is resident."""
        persistent = self._pending[key]
        with ttnn.command_queue(self.cq_id):
            _dbg(f"capture key={key!r} cq={self.cq_id}: begin/record/end")
            tid = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
            try:
                output = forward_fn(persistent)
            finally:
                # Always end the capture — leaving the device mid-capture makes close/sync hang.
                ttnn.end_trace_capture(self.device, tid, cq_id=self.cq_id)
            multi = isinstance(output, (tuple, list))
            self._entries[key] = _Entry(
                tid=tid, persistent=persistent, output=tuple(output) if multi else output, multi=multi
            )
            self.captures += 1
        self._pending.pop(key, None)

    def execute(self, key, inputs: dict) -> object:
        """Copy fresh ``inputs`` into ``key``'s persistent buffers (device->device) and replay the
        trace (blocking). Returns the persistent output(s), valid until the next ``execute`` for
        ``key``. Used for both the capture call and every later replay; counters are the caller's
        responsibility (``captures`` counts first-time captures, ``replays`` counts cache-hit calls)."""
        entry = self._entries[key]
        with ttnn.command_queue(self.cq_id):
            _dbg(f"execute key={key!r} cq={self.cq_id}: copy {len(inputs)} inputs + replay")
            for name, t in inputs.items():
                ttnn.copy(t, entry.persistent[name])
            if self.sync_replay:
                ttnn.synchronize_device(self.device)
            ttnn.execute_trace(self.device, entry.tid, cq_id=self.cq_id, blocking=True)
        return entry.output

    def run(self, key, inputs: dict, forward_fn: Callable[[dict], object]) -> object:
        """Single-trace capture/replay: ``prepare`` + ``capture`` on the first call for ``key``,
        then ``execute`` on every call. For chained coexisting traces use ``prepare``/``capture``/
        ``execute`` directly so ALL persistent buffers are allocated before EITHER trace is captured
        (see :meth:`prepare`).

        ``inputs``: name -> the call's fresh device tensor (same shape/dtype/layout for a given key).
        ``forward_fn``: reads its inputs from the passed persistent dict and returns EITHER a single
        output tensor OR a tuple/list of output tensors; it must treat the dict tensors as read-only
        (clone them if the graph consumes/deallocates inputs).
        """
        if key not in self._entries:
            warm = self.prepare(key, inputs, forward_fn)
            self._free(warm)
            self.capture(key, forward_fn)  # captures += 1
            # First call only RECORDED the graph; execute now so the returned output holds real values.
            return self.execute(key, inputs)
        self.replays += 1
        return self.execute(key, inputs)

    def release(self) -> None:
        """Release all captured traces + free persistent buffers. Call before closing the device."""
        for persistent in self._pending.values():  # prepared but never captured (e.g. error path)
            for t in persistent.values():
                ttnn.deallocate(t)
        self._pending.clear()
        for entry in self._entries.values():
            ttnn.release_trace(self.device, entry.tid)
            for t in entry.persistent.values():
                ttnn.deallocate(t)
            if entry.multi:
                for t in entry.output:
                    ttnn.deallocate(t)
            else:
                ttnn.deallocate(entry.output)
        self._entries.clear()
        if self._prep_enabled:
            set_trace_weight_prep(False)
            clear_trace_weight_prep_cache()
            self._prep_enabled = False
