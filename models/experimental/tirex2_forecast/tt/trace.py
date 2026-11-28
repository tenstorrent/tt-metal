# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Trace capture and replay of ``TtTiRex2.forward_device``.

The op-level sLSTM loop issues ~50k ops per forward, so eager dispatch is host-bound; the trace replays the
whole forward from stable device inputs. One trace per key (device-row count, variate mode, pad cut); the device must be
opened with a ``trace_region_size`` large enough (about 0.5-1.5 GB for the round-0 graph).

Trace safety (``tech_reports/.../TraceCorrectness.md``; checked with ``TT_METAL_TRACE_ALLOC_TRACKING=1``):
- ``prepare(preps)`` warms every key first (compiles every program, creates the per-device constants and every
  trace input), and only then captures each key, so no program-cache entry or persistent buffer is allocated
  after a capture. A key not prepared before the first capture is refused.
- Replays write only into memory the trace owns; inputs are refreshed with ``copy_host_to_device_tensor``.
- Each trace's output is allocated during its own capture, so a later trace's capture can place it in memory an
  earlier trace reuses: outputs are acknowledged as corruptible and must be read before replaying another trace
  (every caller reads the output right after ``execute``). Inputs are acknowledged for the same reason when
  several keys exist (they are refreshed before every replay).
"""

from __future__ import annotations

import ttnn

from models.experimental.tirex2_forecast.tt import layers


class TracedForward:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self._traces: dict = {}

    @staticmethod
    def key(prep) -> tuple:
        return (prep.device_rows, prep.vmask is not None, prep.cut)

    def prepare(self, preps):
        """Warm every new key of ``preps``, then capture them (all keys must be prepared in one call)."""
        todo = {}
        for p in preps:
            k = self.key(p)
            if k not in self._traces and k not in todo:
                todo[k] = p
        if not todo:
            return
        if self._traces:
            raise RuntimeError(
                f"trace keys {list(todo)} were not prepared before the first capture ({list(self._traces)}): "
                "compiling after a capture can place program-cache buffers where a replay writes"
            )
        staged = {}
        for k, p in todo.items():
            staged[k] = self.model.upload_inputs(p)
            layers.free(self.model.forward_device(staged[k]))  # compile every program and create the constants
        ttnn.synchronize_device(self.device)
        for k, inputs in staged.items():
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            out = self.model.forward_device(inputs)
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            self._traces[k] = (trace_id, inputs, out)
        ttnn.synchronize_device(self.device)
        self._acknowledge()

    def _acknowledge(self):
        from ttnn.tools import trace_allocation_tracker

        if not trace_allocation_tracker.trace_allocation_tracking_enabled():
            return
        several = len(self._traces) > 1
        for _, inputs, out in self._traces.values():
            trace_allocation_tracker.acknowledge_corruptible(out)
            if several:
                for t in (inputs.tokens, inputs.vmask, inputs.keep):
                    if t is not None:
                        trace_allocation_tracker.acknowledge_corruptible(t)

    def upload(self, prep):
        """Refresh the stable device inputs of ``prep``'s trace (preparing it first if it is the first key)."""
        key = self.key(prep)
        if key not in self._traces:
            self.prepare([prep])
        _, inputs, _ = self._traces[key]
        host = self.model.upload_inputs(prep, host=True)
        for dst, src in ((inputs.tokens, host.tokens), (inputs.vmask, host.vmask), (inputs.keep, host.keep)):
            if dst is not None:
                ttnn.copy_host_to_device_tensor(src, dst)

    def execute(self, prep):
        """Replay the trace for ``prep``'s key (inputs must be uploaded). Returns the persistent output; read it
        before replaying another trace."""
        trace_id, _, out = self._traces[self.key(prep)]
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
        return out

    def __call__(self, prep):
        self.upload(prep)
        return self.execute(prep)

    def release(self):
        for trace_id, inputs, out in self._traces.values():
            ttnn.release_trace(self.device, trace_id)
            self.model.deallocate_inputs(inputs)
            layers.free(out)
        self._traces.clear()
