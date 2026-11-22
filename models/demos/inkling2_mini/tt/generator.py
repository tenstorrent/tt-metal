# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 greedy generation for Inkling2Model: eager or traced whole 48-application decode steps (one trace per
short-conv ring phase, pos % 4).

v1 prefill = decode token by token."""

from __future__ import annotations

import os
import time

import ttnn
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv as ShortConv


# Weight staging (tt/mk/overlap.py) from capture() to release(); Generator(model, overlap=False) turns it off.
# Decode only: its staging tensor holds ~535 KB of L1 on every core, which prefill programs need, so a session that
# runs Inkling2Prefill between capture() and release() builds Generator(model, overlap=False).
# shortcut: such sessions decode without the staging (~0.47 ms/token); upgrade: free and re-capture around prefill.
# experts=True (default: INKLING_EXPERT_PREFETCH=1) also streams each MoE step's guessed experts into L1 ahead of it
# (tt/mk/expert_overlap.py; needs a PF-layout cache, ~0.47 ms/token at 4K). Once per process: its 300 KB L1 ring is
# not given back by release().
class Generator:
    def __init__(self, model, overlap=True, experts=None):
        self.model = model
        self.use_overlap = overlap
        self.use_experts = os.environ.get("INKLING_EXPERT_PREFETCH") == "1" if experts is None else experts
        self.experts = None  # tt/mk/expert_overlap.ExpertOverlap while installed
        self.device = model.device
        self.trace = None  # [phase] -> trace id
        self.out = None  # [phase] -> output dict
        self.overlap = None  # tt/mk/overlap.Overlap while installed (capture .. release)
        self.router_stage = None  # tt/mk/router_stage.RouterStage, same lifetime

    def capture(self):
        """Compile eagerly once (program cache), then capture one full decode step per conv phase."""
        m = self.model
        m.set_inputs(0, 0)
        m.step()  # compile
        if self.use_overlap and self.overlap is None:
            # stage the next app's w_all into L1 during the single-core residual programs (tt/mk/overlap.py); the
            # attach pattern is periodic from the 2nd eager step on, which the capture needs
            from models.demos.inkling2_mini.tt.mk.overlap import install

            self.overlap = m.overlap = install(m)
            if self.use_experts:
                from models.demos.inkling2_mini.tt.mk.expert_overlap import ExpertOverlap

                self.experts = ExpertOverlap(m)
            from models.demos.inkling2_mini.tt.mk import router_stage

            self.router_stage = router_stage.install(m)  # next app's router weights into L1 (rides on the conv)
            m.step()
            m.step()
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
        if self.router_stage is not None:
            self.router_stage.uninstall()
            self.router_stage = None
        if self.experts is not None:
            self.experts.uninstall(self.model)
            self.experts = None
        if self.overlap is not None:
            self.overlap.uninstall(self.model)
            self.overlap = self.model.overlap = None

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
