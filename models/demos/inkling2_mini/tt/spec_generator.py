# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decoding for Inkling-2-Mini with the DSpark drafter (tt/mtp.py), batch 1.

`TappedModel` = Inkling2Model whose decode step also commits the drafter's context K/V for the position it
decodes (one trace for both). `SpecGenerator.generate`:
  prompt token by token (as v1) -> anchor x_p -> draft 7 tokens (traced) -> verify the first k.
shortcut: verify = k+1 single-token decode steps that stop at the first mismatch (no multi-token target step
yet; PF1's chunked prefill is the planned entry point). Outputs are exactly greedy and every step's features are
committed, so the measured acceptance is real; the per-pass cost is not (see E9-report.md).
"""

from __future__ import annotations

import time

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import APPS, Inkling2Model
from models.demos.inkling2_mini.tt.mtp import TAP_APPS, DSparkDrafter

TAP_LP = tuple(APPS[a] for a in TAP_APPS)  # (layer, pass) of the tapped applications
# the apps whose output is a tapped app's input: their row form must be written (decode writes it only on collect)
PRE_TAP_LP = tuple(APPS[a - 1] for a in TAP_APPS)


class TappedModel(Inkling2Model):
    """Records the drafter taps during step(); `drafter` must be attached before the first step."""

    drafter: DSparkDrafter | None = None
    tapping = True  # False: plain Inkling2Model.step (A/B checks in one process)

    def _app(self, L, p, h, stack, n_sealed, msgs, row_out=True):
        if not self.tapping:
            return super()._app(L, p, h, stack, n_sealed, msgs, row_out=row_out)
        if (L, p) in TAP_LP:
            self._taps[(L, p)] = ttnn.clone(h)  # h is a module buffer that later loop passes overwrite
        return super()._app(L, p, h, stack, n_sealed, msgs, row_out=row_out or (L, p) in PRE_TAP_LP)

    def _rms(self, row, w):
        y = super()._rms(row, w)
        if self.tapping and w is self.final_norm:
            self._taps["final"] = y
        return y

    def step(self, force=None, collect=False):
        if not self.tapping:
            return super().step(force=force, collect=collect)
        self._taps = {}
        res = super().step(force=force, collect=collect)
        rows = [self._taps[k] for k in TAP_LP] + [self._taps["final"]]
        feats = ttnn.pad(ttnn.concat(rows, dim=2), [(0, 0), (0, 0), (0, 28), (0, 0)], 0.0)
        if collect:
            res["feats"] = feats
        self.last_feats = feats
        self.drafter.commit(feats, self.positions.pos)
        ttnn.copy(ttnn.reshape(res["token"], [1, 1]), self.drafter.anchor)  # next anchor = this step's argmax
        self._taps = {}
        return res

    def set_inputs(self, token, pos):
        super().set_inputs(token, pos)
        if self.tapping:
            self.drafter.set_commit_pos(pos)


class SpecGenerator(Generator):
    def __init__(self, model: TappedModel, drafter: DSparkDrafter):
        super().__init__(model)
        self.drafter = drafter

    def capture(self):
        """Compile target step and draft first, then capture both traces (no program first built after a capture)."""
        m, d = self.model, self.drafter
        m.set_inputs(0, 0)
        m.step()
        d.set_draft_pos(1)
        d.draft(d.anchor)
        ttnn.synchronize_device(self.device)
        super().capture()
        d.capture(compile=False)

    def release(self):
        self.drafter.release()
        super().release()

    def generate_spec(self, prompt_ids, max_new, k=4, traced=True):
        """Greedy tokens (identical to Generator.generate) + per-pass stats.

        Returns (new ids, stats) with stats = {"passes", "accepted" (per pass), "draft_s", "verify_s"}."""
        self.model.reset()
        pos = 0
        out = None
        for t in prompt_ids:
            out = self.decode(t, pos, traced)
            self.read_token(out)  # shortcut: sync per prompt step; an unsynced traced prefill hung on card
            pos += 1
        nxt = self.read_token(out)
        new = [nxt]
        acc, t_draft, t_ver = [], 0.0, 0.0
        while len(new) < max_new:
            t0 = time.perf_counter()
            drafts = self.drafter.run(pos) if traced else self._draft_eager(pos)
            t1 = time.perf_counter()
            a = 0
            tok = nxt
            for j in range(k + 1):
                out = self.decode(tok, pos, traced)
                pos += 1
                tok = self.read_token(out)
                new.append(tok)
                if len(new) >= max_new or j == k or drafts[j] != tok:
                    break
                a += 1
            nxt = tok
            t_draft += t1 - t0
            t_ver += time.perf_counter() - t1
            acc.append(a)
        stats = {"passes": len(acc), "accepted": acc, "draft_s": t_draft, "verify_s": t_ver}
        return new[:max_new], stats

    def _draft_eager(self, pos):
        d = self.drafter
        d.set_draft_pos(pos)
        return [int(x) for x in ttnn.to_torch(d.draft(d.anchor)).reshape(-1)[:7].tolist()]


def build(device, max_ctx, log=print, mtp_root=None, k=7, **kw):
    """TappedModel + its drafter (Inkling2Model.__init__ never steps, so the drafter can be attached after)."""
    model = TappedModel(device, max_ctx=max_ctx, log=log, **kw)
    model.drafter = DSparkDrafter(model, root=mtp_root, max_ctx=max_ctx, k=k)
    return model, model.drafter
