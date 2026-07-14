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

from dataclasses import dataclass
from typing import Callable

import ttnn

from .tt_trace_prep import clear_trace_weight_prep_cache, set_trace_weight_prep


@dataclass
class _Entry:
    tid: int
    persistent: dict  # name -> persistent input buffer (fixed address)
    output: object  # persistent output buffer(s) overwritten by each replay; Tensor or tuple[Tensor]
    multi: bool = False  # True if forward_fn returned a tuple/list of tensors


class TraceManager:
    """Per-shape-bucket metal-trace capture/replay with persistent input buffers."""

    def __init__(self, device: ttnn.Device, *, cq_id: int = 0) -> None:
        self.device = device
        self.cq_id = cq_id
        self._entries: dict = {}
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

    def run(self, key, inputs: dict, forward_fn: Callable[[dict], object]) -> object:
        """Capture (first call for ``key``) or replay ``forward_fn`` with ``inputs`` updated in place.

        ``inputs``: name -> the call's fresh device tensor (same shape/dtype/layout for a given key).
        ``forward_fn``: reads its inputs from the passed persistent dict and returns EITHER a single
        output tensor OR a tuple/list of output tensors; it must treat the dict tensors as read-only
        (clone them if the graph consumes/deallocates inputs). Returns the (persistent) output(s) —
        same shape (single tensor or tuple) as ``forward_fn`` returned — valid until the next ``run``
        for the same key.
        """

        def _warm_free(o):
            if isinstance(o, (tuple, list)):
                for t in o:
                    ttnn.deallocate(t)
            else:
                ttnn.deallocate(o)

        entry = self._entries.get(key)
        if entry is None:
            self._ensure_prep()
            # Persistent input buffers = clones of this call's inputs (fixed addresses for the trace).
            persistent = {name: ttnn.clone(t) for name, t in inputs.items()}
            # Warmup: compile every program + populate the weight-prep caches BEFORE capture
            # (trace capture forbids the on-device writes those would otherwise emit).
            warm = forward_fn(persistent)
            ttnn.synchronize_device(self.device)
            _warm_free(warm)
            # Capture.
            tid = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
            try:
                output = forward_fn(persistent)
            finally:
                # Always end the capture — leaving the device mid-capture makes close/sync hang.
                ttnn.end_trace_capture(self.device, tid, cq_id=self.cq_id)
            # Trace capture only RECORDS the graph — the output buffers hold uncomputed garbage until
            # the trace is executed. Run it once now (against the persistent inputs, = this call's
            # values) so the returned output is valid for the capture call, exactly as a replay would
            # be. Without this a caller that reads the capture output on the host (e.g. a duration
            # readback that then sets a downstream shape) gets garbage and can crash.
            ttnn.execute_trace(self.device, tid, cq_id=self.cq_id, blocking=True)
            multi = isinstance(output, (tuple, list))
            self._entries[key] = _Entry(
                tid=tid, persistent=persistent, output=tuple(output) if multi else output, multi=multi
            )
            self.captures += 1
            return output

        # Replay: copy fresh inputs into the persistent buffers (device->device), then execute.
        for name, t in inputs.items():
            ttnn.copy(t, entry.persistent[name])
        ttnn.execute_trace(self.device, entry.tid, cq_id=self.cq_id, blocking=True)
        self.replays += 1
        return entry.output

    def release(self) -> None:
        """Release all captured traces + free persistent buffers. Call before closing the device."""
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
