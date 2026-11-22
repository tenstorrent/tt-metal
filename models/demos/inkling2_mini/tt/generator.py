# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 greedy generation for Inkling2Model: eager or traced whole 48-application decode steps (one trace per
short-conv ring phase, pos % 4).

v1 prefill = decode token by token (design.md §6)."""

from __future__ import annotations

import time

import ttnn
from models.demos.inkling2_mini.tt.residual import ShortConv


class Generator:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.trace = None  # [phase] -> trace id
        self.out = None  # [phase] -> output dict

    def capture(self):
        """Compile eagerly once (program cache), then capture one full decode step per conv phase."""
        m = self.model
        m.set_inputs(0, 0)
        m.step()  # compile
        ttnn.synchronize_device(self.device)
        self.trace, self.out = [], []
        for ph in range(ShortConv.PHASES):
            m.set_inputs(0, ph)
            self.trace.append(ttnn.begin_trace_capture(self.device, cq_id=0))
            self.out.append(m.step())
            ttnn.end_trace_capture(self.device, self.trace[-1], cq_id=0)
        ttnn.synchronize_device(self.device)

    def release(self):
        for t in self.trace or []:
            ttnn.release_trace(self.device, t)
        self.trace = None

    def decode(self, token, pos, traced=True):
        """One step. Returns device result dict (logits, token)."""
        self.model.set_inputs(token, pos)
        if traced:
            ph = pos % ShortConv.PHASES
            ttnn.execute_trace(self.device, self.trace[ph], cq_id=0, blocking=False)
            return self.out[ph]
        return self.model.step()

    @staticmethod
    def read_token(out):
        return int(ttnn.to_torch(out["token"]).reshape(-1)[0].item())

    def generate(self, prompt_ids, max_new, traced=True, on_token=None):
        """Token-by-token prefill of prompt_ids, then greedy max_new tokens. Returns (new ids, per-step seconds)."""
        self.model.reset()
        ids = list(prompt_ids)
        pos = 0
        out = None
        for t in ids:
            out = self.decode(t, pos, traced)
            pos += 1
        nxt = self.read_token(out)
        new, times = [nxt], []
        for _ in range(max_new - 1):
            t0 = time.perf_counter()
            out = self.decode(nxt, pos, traced)
            nxt = self.read_token(out)
            times.append(time.perf_counter() - t0)
            pos += 1
            new.append(nxt)
            if on_token:
                on_token(nxt)
        return new, times
