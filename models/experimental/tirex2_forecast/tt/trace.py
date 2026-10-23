# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Trace capture and replay of ``TtTiRex2.forward_device``.

The op-level sLSTM loop issues ~50k ops per forward, so eager dispatch is host-bound; the trace replays the
whole forward from stable device inputs. One trace per device-row count and variate mode; the device must be
opened with a ``trace_region_size`` large enough for it (about 0.5 GB for the round-0 graph).
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
        return (prep.device_rows, prep.vmask is not None)

    def _host_inputs(self, prep):
        return self.model.upload_inputs(prep, host=True)

    def upload(self, prep):
        """Refresh the stable device inputs of ``prep``'s trace (captures it first if needed)."""
        key = self.key(prep)
        if key not in self._traces:
            self._capture(prep)
            return
        _, inputs, _ = self._traces[key]
        host = self._host_inputs(prep)
        for dst, src in ((inputs.tokens, host.tokens), (inputs.vmask, host.vmask), (inputs.keep, host.keep)):
            if dst is not None:
                ttnn.copy_host_to_device_tensor(src, dst)

    def execute(self, prep):
        """Replay the trace for ``prep``'s shape (inputs must be uploaded). Returns the persistent output."""
        trace_id, _, out = self._traces[self.key(prep)]
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
        return out

    def __call__(self, prep):
        self.upload(prep)
        return self.execute(prep)

    def _capture(self, prep):
        inputs = self.model.upload_inputs(prep)
        layers.free(self.model.forward_device(inputs))  # compile every program and create the constants
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        out = self.model.forward_device(inputs)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._traces[self.key(prep)] = (trace_id, inputs, out)

    def release(self):
        for trace_id, inputs, out in self._traces.values():
            ttnn.release_trace(self.device, trace_id)
            self.model.deallocate_inputs(inputs)
            layers.free(out)
        self._traces.clear()
