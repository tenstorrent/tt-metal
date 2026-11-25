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


def _pf_cache(model):
    """True when every MoE layer's experts were loaded in the PF layout (the expert prefetch needs it)."""
    return all(getattr(getattr(getattr(L, "moe", None), "fused", None), "pf", False) for L in model.layers)


# Weight staging (tt/mk/overlap.py) from capture() to release(); Generator(model, overlap=False) turns it off.
# Decode only: its staging tensor holds ~535 KB of L1 on every core, which prefill programs need, so a session that
# runs Inkling2Prefill between capture() and release() builds Generator(model, overlap=False).
# shortcut: such sessions decode without the staging (~0.47 ms/token); upgrade: free and re-capture around prefill.
# experts=True also streams each MoE step's guessed experts into L1 ahead of it (tt/mk/expert_overlap.py; needs a
# PF-layout cache, ~0.3 ms/token at 4K). Default: on when the cache has the PF expert layout (with overlap on);
# INKLING_EXPERT_PREFETCH=0 / 1 forces it. Once per process: its L1 ring is not given back by release().
class Generator:
    def __init__(self, model, overlap=True, experts=None):
        self.model = model
        self.use_overlap = overlap
        if experts is None:
            env = os.environ.get("INKLING_EXPERT_PREFETCH")
            experts = env == "1" if env is not None else _pf_cache(model)
        self.use_experts = experts
        self.experts = None  # tt/mk/expert_overlap.ExpertOverlap while installed
        self.device = model.device
        self.trace = None  # [phase] -> trace id
        self.out = None  # [phase] -> output dict
        self.overlap = None  # tt/mk/overlap.Overlap while installed (capture .. release)
        self.router_stage = None  # tt/mk/router_stage.RouterStage, same lifetime

    def capture(self):
        """Compile eagerly once (program cache), then capture one full decode step per conv phase."""
        m = self.model
        # compile, and every conv phase once: the modules' own L1 buffers (some are per phase) are allocated before
        # the staging's, so that release() leaves the L1 below them free again (a chunked prefill needs it)
        for ph in range(ShortConv.PHASES):
            m.set_inputs(0, ph)
            m.step()
        self._install(settle=2)
        ttnn.synchronize_device(self.device)
        self.trace, self.out = [], []
        for ph in range(ShortConv.PHASES):
            m.set_inputs(0, ph)
            self.trace.append(ttnn.begin_trace_capture(self.device, cq_id=0))
            self.out.append(m.step())
            ttnn.end_trace_capture(self.device, self.trace[-1], cq_id=0)
        ttnn.synchronize_device(self.device)

    def _install(self, settle):
        """The weight staging (once, until release()), then `settle` eager steps."""
        m = self.model
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
            try:
                for _ in range(settle):
                    m.step()
            except Exception:
                if self.experts is not None:  # a queued prefetch request would hang the device close
                    self.experts.abort(m)
                    self.experts = None
                raise
            return True
        return False

    # Capture in the middle of a sequence. A chunked prefill (tt/prefill_model.py) needs the L1 the staging holds,
    # so a process that decodes with the staging calls release() before it; and no step may run afterwards but the
    # sequence's own (a step can not be undone: the conv histories shift, the local rings lose a position). So the
    # traces come from the sequence's next steps themselves:
    #     gen.release(); prefill; gen.begin_live(); then gen.step_live(token, pos) for every step while gen.live
    # The staging goes in again and its two settle steps are real decode steps, run eagerly (same outputs as a
    # replay); each of the next four steps captures its conv phase's trace and runs it; from then on step_live
    # replays.
    def begin_live(self):
        self._settle = 2 if self._install(settle=0) else 0
        self.trace, self.out = [None] * ShortConv.PHASES, [None] * ShortConv.PHASES

    @property
    def live(self):
        """True while a conv phase has no trace yet."""
        return self.trace is not None and any(t is None for t in self.trace)

    def step_live(self, token, pos):
        """One decode step of the sequence: eager while the staging settles, then captured for its conv phase,
        a replay once the phase has a trace. Returns the output dict."""
        m, ph = self.model, pos % ShortConv.PHASES
        m.set_inputs(token, pos)
        if self._settle:
            self._settle -= 1
            return m.step()
        if self.trace[ph] is None:  # (a capture records the step's programs; the replay below runs them)
            ttnn.synchronize_device(self.device)
            self.trace[ph] = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.out[ph] = m.step()
            ttnn.end_trace_capture(self.device, self.trace[ph], cq_id=0)
        ttnn.execute_trace(self.device, self.trace[ph], cq_id=0, blocking=False)
        return self.out[ph]

    def release(self):
        for t in self.trace or []:
            if t is not None:
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
        # the expert steps' last output rows (L1, one per layer): no trace writes them any more, and a chunked
        # prefill needs their L1
        for lay in self.model.layers:
            free = getattr(getattr(lay.moe, "fused", None), "release_last", None)
            if free is not None:
                free()

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
