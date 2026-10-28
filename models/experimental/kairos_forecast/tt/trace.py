# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Traced forward_device, one trace per batch size (N is padded to 64, so the batch fixes every shape).

PLAN v1.21: nothing may be allocated while a trace exists, or it can alias the trace's freed intermediates. So
there is one active trace at a time: a new batch size releases it, allocates that batch's persistent inputs (kept
for later), warms up eagerly, and captures again. Each replay refreshes the inputs in place with
copy_host_to_device_tensor. Capturing a second trace while the first is alive allocates its intermediates under
an active trace (ttnn warns), which is why the runner does not keep several.
"""
from __future__ import annotations

import torch


def _leaves(x):
    if isinstance(x, dict):
        for k in sorted(k for k in x if k != "B"):
            yield from _leaves(x[k])
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


class TracedKairos:
    """Drop-in for TtKairos.forward (front -> head [B, 2, 576]), traced per batch size, one trace alive at a time."""

    def __init__(self, model):
        self.model, self.device, self.W = model, model.device, model.W
        self.inputs, self.B, self.tid, self.output = {}, None, None, None
        self.captures = 0

    def _switch(self, front):
        import ttnn

        B = front["attn_mask"].shape[0]
        if B == self.B:
            return
        self.release()
        if B not in self.inputs:
            self.inputs[B] = self.model.prepare(front)
        self.model.forward_device(self.inputs[B])  # compile / warm up, no trace alive
        ttnn.synchronize_device(self.device)
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output = self.model.forward_device(self.inputs[B])
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.B = B
        self.captures += 1

    def load(self, front):
        """Copy one forward's inputs into the persistent buffers of its batch (capturing first if needed)."""
        import ttnn

        self._switch(front)
        host = self.model.prepare_host(front)
        dev = {k: v for k, v in self.inputs[self.B].items() if k not in ("reg", "dec_in", "causal")}
        for h, d in zip(_leaves(host), _leaves(dev), strict=True):
            ttnn.copy_host_to_device_tensor(h, d)
        return self.B

    def run(self):
        import ttnn

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.output

    def forward(self, front):
        import ttnn

        B = self.load(front)
        out = ttnn.to_torch(self.run()).float()
        return out.view(B, -1, out.shape[-1])[:, :2]

    def release(self):
        import ttnn

        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)
            self.tid, self.B, self.output = None, None, None
