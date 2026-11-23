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

import os
import time

import torch

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
    def __init__(self, model: TappedModel, drafter: DSparkDrafter, overlap=True, verify=None):
        """verify: a tt/verify.VerifyStep (R = drafter.k + 1) for generate_rows; None: generate_spec only."""
        super().__init__(model, overlap=overlap)
        self.drafter = drafter
        self.verify = verify
        self.vtrace = {}  # (conv phase, shared far call) -> (trace id, outputs)
        self.ctrace = {}  # (conv phase, accepted rows) -> trace id

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
        for tid in [t for t, _ in self.vtrace.values()] + list(self.ctrace.values()):
            ttnn.release_trace(self.device, tid)
        self.vtrace, self.ctrace = {}, {}
        super().release()

    # ------------------------------------------------------------ multi-row verify (tt/verify.py)
    @staticmethod
    def _shared(p0, R):
        from models.demos.inkling2_mini.tt.attention import WINDOW
        from models.demos.inkling2_mini.tt.attention_fast import FAR_CHUNK
        from models.demos.inkling2_mini.tt.verify import SHARED_FAR

        return SHARED_FAR and p0 >= WINDOW and (p0 - WINDOW) // FAR_CHUNK == (p0 + R - 1 - WINDOW) // FAR_CHUNK

    def _feed(self):
        """Device-side verify inputs: token 0 = the anchor, tokens 1..k = the drafts (no host round trip)."""
        V, d = self.verify, self.drafter
        ttnn.copy(ttnn.reshape(d.anchor, [1, 1]), V.tokens[0])
        for t in range(1, V.R):
            ttnn.copy(ttnn.reshape(ttnn.slice(d.tokens, [0, 0, 0, t - 1], [1, 1, 1, t]), [1, 1]), V.tokens[t])

    def _set_anchor(self, n):
        """After the accept of n rows: the next anchor is verify token n - 1 (device copy)."""
        tok = ttnn.slice(self.vtok, [0, 0, 0, n - 1], [1, 1, 1, n])
        ttnn.copy(ttnn.reshape(tok, [1, 1]), self.drafter.anchor)

    def capture_rows(self, p_shared, p_split):
        """Plain traces (Generator.capture), the draft trace, and per conv phase the verify pass (far call shared or
        per token) and the commit of n = 1 .. R accepted rows. p_shared / p_split: positions >= 1024 whose
        pass does / does not share the far chunk (only the host-side choice depends on them). Every program is
        compiled eagerly first (a program first compiled after a capture corrupts that trace's scratch)."""
        m, d, V = self.model, self.drafter, self.verify
        R = V.R
        say = print if os.environ.get("S6_STAGES") else (lambda *a, **k: None)
        assert not self._shared(p_split, R)
        # verify tokens of the last pass (host reads them; the commit trace copies the next anchor from them):
        # allocated before any capture so no trace's scratch can overlap it
        self.vtok = ttnn.from_torch(torch.zeros(1, 1, 1, R, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)  # fmt: skip
        m.set_inputs(0, 0)
        m.step()
        if self.use_overlap and self.overlap is None:
            # decode staging first (as Generator.capture), so every verify program is compiled in its final form
            # before any capture; Generator.capture below then only settles and captures
            from models.demos.inkling2_mini.tt.mk import router_stage
            from models.demos.inkling2_mini.tt.mk.overlap import install

            self.overlap = m.overlap = install(m)
            self.router_stage = router_stage.install(m)
            m.step()
            m.step()
        d.set_draft_pos(1)
        d.tokens = d.draft(d.anchor)
        for p in (p_shared, p_split):
            V.set_inputs([0] * R, p)
            self._feed()
            ttnn.copy(V.step()["token"][0], self.vtok)
            ttnn.synchronize_device(self.device)
            say("S6 eager verify", p, flush=True)
        for n in range(1, R + 1):
            V.commit(n)
            self._set_anchor(n)
            ttnn.synchronize_device(self.device)
            say("S6 eager commit", n, flush=True)
        if self.overlap is not None:  # settle the staging attach pattern again after the verify passes
            m.set_inputs(0, 0)
            m.step()
            m.step()
        super().capture()
        say("S6 plain captured", flush=True)
        d.capture(compile=False)
        say("S6 draft captured", flush=True)
        for ph in range(4):
            for shared in (True, False):
                # a position of conv phase ph in this far-chunk case (a chunk-crossing pass exists only at some
                # phases: R tokens cross a 128-key chunk only from its last R-1 offsets)
                p = next((q for q in range(4, p_shared + 512) if q % 4 == ph and self._shared(q, R) == shared),
                         None)  # fmt: skip
                if p is None:
                    continue
                V.set_positions(p)
                tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                self._feed()
                out = V.step()
                ttnn.copy(out["token"][0], self.vtok)
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                self.vtrace[(ph, shared)] = (tid, out)
                say("S6 verify captured", ph, shared, flush=True)
            for n in range(1, R + 1):
                V.set_inputs([0] * R, p_shared - p_shared % 4 + ph)
                tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                V.commit(n)
                self._set_anchor(n)
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                self.ctrace[(ph, n)] = tid
        ttnn.synchronize_device(self.device)

    def generate_rows(self, prompt_ids, max_new, on_pass=None):
        """Greedy tokens through draft (k = R - 1) + one R-row verify pass + commit per pass; output identical to
        Generator.generate. Returns (new ids, stats)."""
        import time as _t

        m, d, V = self.model, self.drafter, self.verify
        R, k = V.R, V.R - 1
        assert d.k >= k
        m.reset()
        out = None
        for pos, t in enumerate(prompt_ids):
            out = self.decode(t, pos, True)
            if pos % 64 == 63:
                self.read_token(out)
        nxt = self.read_token(out)
        pos = len(prompt_ids)
        new, acc, t_draft, t_ver = [nxt], [], 0.0, 0.0
        while len(new) < max_new:
            # one host sync per pass: the anchor (from the last commit / prompt step) and the drafts reach the
            # verify pass on device; the host reads the verify tokens and the drafts after it
            t0 = _t.perf_counter()
            d.set_draft_pos(pos)
            ttnn.execute_trace(self.device, d.trace, cq_id=0, blocking=False)
            V.set_positions(pos)  # host writes while the draft runs
            t1 = _t.perf_counter()
            ph = pos % 4
            ttnn.execute_trace(self.device, self.vtrace[(ph, self._shared(pos, R))][0], cq_id=0, blocking=False)
            tgt = [int(x) for x in ttnn.to_torch(self.vtok).reshape(-1)[:R].tolist()]
            drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()]
            a = 0
            while a < k and drafts[a] == tgt[a]:
                a += 1
            ttnn.execute_trace(self.device, self.ctrace[(ph, a + 1)], cq_id=0, blocking=False)
            t_draft += t1 - t0
            t_ver += _t.perf_counter() - t1
            new += tgt[: a + 1]
            acc.append(a)
            nxt, pos = tgt[a], pos + a + 1
            if on_pass:
                on_pass(a)
        ttnn.synchronize_device(self.device)
        stats = {"passes": len(acc), "accepted": acc, "draft_s": t_draft, "verify_s": t_ver}
        return new[:max_new], stats

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
