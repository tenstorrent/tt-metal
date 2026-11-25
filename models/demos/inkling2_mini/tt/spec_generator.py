# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decoding for Inkling-2-Mini with the DSpark drafter (tt/mtp.py), batch 1.

`TappedModel` = Inkling2Model whose decode step also commits the drafter's context K/V for the position it
decodes (one trace for both). `SpecGenerator.generate`:
  prompt token by token (as v1) -> anchor x_p -> draft 7 tokens (traced) -> verify the first k.
shortcut: verify = k+1 single-token decode steps that stop at the first mismatch (no multi-token target step
yet; the chunked prefill is the planned entry point). Outputs are exactly greedy and every step's features are
committed, so the measured acceptance is real; the per-pass cost is not.
"""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import APPS, Inkling2Model
from models.demos.inkling2_mini.tt.mtp import STASH, TAP_APPS, DSparkDrafter

TAP_LP = tuple(APPS[a] for a in TAP_APPS)  # (layer, pass) of the tapped applications
# the apps whose output is a tapped app's input: their row form must be written (decode writes it only on collect)
PRE_TAP_LP = tuple(APPS[a - 1] for a in TAP_APPS)


class TappedModel(Inkling2Model):
    """Records the drafter taps during step(); `drafter` must be attached before the first step."""

    drafter: DSparkDrafter | None = None
    tapping = True  # False: plain Inkling2Model.step (A/B checks in one process)
    # True: the step only stashes its taps (slot = pos % STASH) and the step of the last slot commits the group
    # in one pass (DSparkDrafter.stash_commit): an eighth of the per-step commit's small programs. For runs of plain
    # steps (a prompt feed, text the drafter does not predict): a draft needs every position before it committed,
    # so a group cut short is committed by stash_commit(n) first (SpecGenerator.ftrace).
    stash = False
    # True: the step leaves the drafter alone (no taps, no commit): the plain step for positions the drafter does
    # not cover (its max_ctx is below the model's), beside the drafter's and the verify pass's programs.
    quiet = False

    def _app(self, L, p, h, stack, n_sealed, msgs, row_out=True):
        if not self.tapping:
            return super()._app(L, p, h, stack, n_sealed, msgs, row_out=row_out)
        tap = not self.quiet
        if tap and (L, p) in TAP_LP:  # h is a module buffer that later loop passes overwrite
            j = TAP_LP.index((L, p))
            self._taps[(L, p)] = self.drafter.stash_put(self.slot, j, h) if self.stash else ttnn.clone(h)
        res = super()._app(L, p, h, stack, n_sealed, msgs, row_out=row_out or (tap and (L, p) in PRE_TAP_LP))
        self._release()  # the app before this one: its expert output was read by this app's first program
        self._expert = getattr(self.layers[L].moe.fused, "release_last", None) if self.free_rows else None
        return res

    # Each expert step's output row (L1, kept alive for its twin) is freed once its consumer has run. An L1 buffer
    # takes its bytes from the circular buffers of every core, and the 24 layers keep one row each (48 KB): room
    # the expert prefetch ring needs beside the drafter's and the verify pass's programs (SpecGenerator).
    free_rows = True
    _expert = None

    def _release(self):
        if self._expert is not None:
            self._expert()
            self._expert = None

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
        self._release()
        if self.quiet:
            self._taps = {}
            return res
        if self.stash:
            d = self.drafter
            d.stash_put(self.slot, len(TAP_LP), self._taps["final"])
            if self.slot == STASH - 1:
                d.stash_commit(STASH)
            self._taps = {}  # (no anchor copy: the stash steps' caller writes the next draft's anchor itself)
            return res
        rows = [self._taps[k] for k in TAP_LP] + [self._taps["final"]]
        feats = ttnn.pad(ttnn.concat(rows, dim=2), [(0, 0), (0, 0), (0, 28), (0, 0)], 0.0)
        if collect:
            res["feats"] = feats
        self.last_feats = feats
        self.drafter.commit(feats, self.positions.pos)
        ttnn.copy(ttnn.reshape(res["token"], [1, 1]), self.drafter.anchor)  # next anchor = this step's argmax
        self._taps = {}
        return res

    slot = 0  # stash slot of the step: pos % STASH

    def set_inputs(self, token, pos):
        super().set_inputs(token, pos)
        self.slot = pos % STASH
        if self.tapping and pos < self.drafter.max_ctx:
            self.drafter.set_commit_pos(pos)


class SpecGenerator(Generator):
    def __init__(self, model: TappedModel, drafter: DSparkDrafter, overlap=True, verify=None, stash=False,
                 quiet=None):  # fmt: skip
        """verify: a tt/verify.VerifyStep (R <= drafter.k + 1) for generate_rows, or a list of them (early stop:
        one per row count); None: generate_spec only. stash: capture_rows also captures the stash steps (strace,
        sout: as trace, out) and the commit of a group's first n positions (ftrace[n]); see TappedModel.stash.
        quiet: capture_rows also captures the steps that leave the drafter alone (qtrace, qout: as trace, out;
        TappedModel.quiet), for positions from drafter.max_ctx on; None = when the drafter covers less than the
        model."""
        super().__init__(model, overlap=overlap)
        self.quiet = drafter.max_ctx < model.max_ctx if quiet is None else quiet
        self.qtrace, self.qout = None, None
        self.stash, self.strace, self.sout, self.ftrace = stash, None, None, {}
        self.drafter = drafter
        vs = verify if isinstance(verify, (list, tuple)) else [verify] if verify is not None else []
        self.verifies = {V.R: V for V in vs}
        self.verify = max(vs, key=lambda V: V.R) if vs else None  # the largest pass
        self.vtrace = {}  # (rows, conv phase, shared far call) -> (trace id, outputs)
        self.ctrace = {}  # (rows, conv phase, accepted rows) -> trace id
        self.vtok = {}  # rows -> verify tokens of the last pass with that many rows

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
        for tid in ([t for t, _ in self.vtrace.values()] + list(self.ctrace.values()) + (self.strace or [])
                    + (self.qtrace or [])):  # fmt: skip
            ttnn.release_trace(self.device, tid)
        self.qtrace = None
        for tid in self.ftrace.values():
            ttnn.release_trace(self.device, tid)
        self.vtrace, self.ctrace, self.strace, self.ftrace = {}, {}, None, {}
        super().release()

    # ------------------------------------------------------------ multi-row verify (tt/verify.py)
    @staticmethod
    def _shared(p0, R):
        from models.demos.inkling2_mini.tt.attention import WINDOW
        from models.demos.inkling2_mini.tt.attention_fast import FAR_CHUNK

        return p0 >= WINDOW and (p0 - WINDOW) // FAR_CHUNK == (p0 + R - 1 - WINDOW) // FAR_CHUNK

    def _feed(self, V):
        """Device-side verify inputs: token 0 = the anchor, tokens 1..k = the drafts (no host round trip)."""
        d = self.drafter
        ttnn.copy(ttnn.reshape(d.anchor, [1, 1]), V.tokens[0])
        for t in range(1, V.R):
            ttnn.copy(ttnn.reshape(ttnn.slice(d.tokens, [0, 0, 0, t - 1], [1, 1, 1, t]), [1, 1]), V.tokens[t])

    def _set_anchor(self, R, n):
        """After the accept of n rows of an R-row pass: the next anchor is verify token n - 1 (device copy)."""
        tok = ttnn.slice(self.vtok[R], [0, 0, 0, n - 1], [1, 1, 1, n])
        ttnn.copy(ttnn.reshape(tok, [1, 1]), self.drafter.anchor)

    def capture_rows(self, p_shared, p_split=None, before_capture=None):
        try:
            self._capture_rows(p_shared, before_capture)
        except Exception:
            if self.experts is not None:  # a queued prefetch request would hang the device close
                self.experts.abort(self.model)
                self.experts = None
            raise

    def _capture_rows(self, p_shared, before_capture=None):
        """Plain traces (Generator.capture), the draft trace, and per verify step (row count R) and conv phase the
        verify pass (far call shared or per token) and the commit of n = 1 .. R accepted rows. p_shared: a position
        >= 1024 whose pass shares the far chunk; the chunk-crossing position of each R is found from it (p_split:
        unused, kept for callers). Every program is compiled eagerly first (a program first compiled after a capture
        corrupts that trace's scratch); before_capture() runs after that, for the caller's own programs."""
        m, d = self.model, self.drafter
        say = print if os.environ.get("INKLING_SPEC_STAGES") else (lambda *a, **k: None)
        Vs = list(self.verifies.values())
        for V in Vs:
            assert self._shared(p_shared, V.R)
            # verify tokens of the last pass (host reads them; the commit trace copies the next anchor from them):
            # allocated before any capture so no trace's scratch can overlap it
            self.vtok[V.R] = ttnn.from_torch(torch.zeros(1, 1, 1, V.R, dtype=torch.int32), dtype=ttnn.uint32,
                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)  # fmt: skip
        m.set_inputs(0, 0)
        m.step()
        for V in Vs:  # one eager pass per row count before the staging: the L1 buffers a pass keeps from its
            V.set_inputs([0] * V.R, p_shared)  # first run on sit above the staging, not below it (a chunked
            V.step()  # prefill between two steps needs the L1 below the staging for its circular buffers)
        ttnn.synchronize_device(self.device)
        if self.use_overlap and self.overlap is None:
            # decode staging first (as Generator.capture), so every verify program is compiled in its final form
            # before any capture; Generator.capture below then only settles and captures
            from models.demos.inkling2_mini.tt.mk import router_stage
            from models.demos.inkling2_mini.tt.mk.overlap import install

            self.overlap = m.overlap = install(m)
            if self.use_experts:
                # guessed-expert prefetch, as Generator.capture (the verify pass keeps the ring protocol). Its
                # 240 KB L1 ring fits beside the drafter's and the verify pass's programs because the expert
                # output rows are freed after use (TappedModel.free_rows, from the first step on: the staging
                # tensor then sits 48 KB higher) and the multi-row expert step holds one gate_up column, not two
                from models.demos.inkling2_mini.tt.mk.expert_overlap import ExpertOverlap

                assert m.free_rows, "the expert ring needs TappedModel.free_rows from the first step on"
                for lay in m.layers:
                    lay.moe.fused.mr_cols = 1
                self.experts = ExpertOverlap(m)
            self.router_stage = router_stage.install(m)
            m.step()
            m.step()
        d.set_draft_pos(1)
        d.tokens = d.draft(d.anchor)
        snap = self._stage_snap()  # every verify pass, eager or captured, starts from this staging state, so the
        for V in Vs:  # fills its host programs carry (their program variants) are the same in both
            R = V.R
            p_sp = next(q for q in range(p_shared, p_shared + 256) if not self._shared(q, R))
            for p in (p_shared, p_sp):
                V.set_inputs([0] * R, p)
                self._stage_begin(snap)
                self._feed(V)
                ttnn.copy(V.step()["token"][0], self.vtok[R])
                self._stage_topup(snap)
                self._stage_restore(snap)
                ttnn.synchronize_device(self.device)
                say("spec eager verify", R, p, flush=True)
            for n in range(1, R + 1):
                V.commit(n)
                self._set_anchor(R, n)
                ttnn.synchronize_device(self.device)
                say("spec eager commit", R, n, flush=True)
        if self.stash:  # the stash steps (from the same staging state as their captures) and the short commits
            m.stash = True
            d.set_stash_pos(0)
            for ph in range(STASH):
                m.set_inputs(0, ph)
                self._stage_restore(snap)
                m.step()
                self._stage_topup(snap)
                self._stage_restore(snap)
            for n in range(1, STASH):
                d.stash_commit(n)
            m.stash = False
            ttnn.synchronize_device(self.device)
            say("spec eager stash", flush=True)
        if self.quiet:
            m.quiet = True
            for ph in range(4):
                m.set_inputs(0, ph)
                self._stage_restore(snap)
                m.step()
                self._stage_topup(snap)
                self._stage_restore(snap)
            m.quiet = False
            ttnn.synchronize_device(self.device)
        if before_capture is not None:
            before_capture()
        if self.overlap is not None:  # settle the staging attach pattern again after the verify passes
            m.set_inputs(0, 0)
            m.step()
            m.step()
        super().capture()
        say("spec plain captured", flush=True)
        snap = self._stage_snap(snap)  # the same state after the plain captures (asserted)
        d.capture(compile=False)
        self._stage_restore(snap)
        say("spec draft captured", flush=True)
        if self.stash:
            # as the verify passes: every stash step starts from the plain step's staging state and ends in it
            m.stash = True
            self.strace, self.sout = [], []
            for ph in range(STASH):
                m.set_inputs(0, ph)
                self._stage_restore(snap)
                self.strace.append(ttnn.begin_trace_capture(self.device, cq_id=0))
                self.sout.append(m.step())
                self._stage_topup(snap)
                ttnn.end_trace_capture(self.device, self.strace[-1], cq_id=0)
                self._stage_snap(snap)
                self._stage_restore(snap)
            m.stash = False
            for n in range(1, STASH):
                self.ftrace[n] = ttnn.begin_trace_capture(self.device, cq_id=0)
                d.stash_commit(n)
                ttnn.end_trace_capture(self.device, self.ftrace[n], cq_id=0)
            say("spec stash captured", flush=True)
        if self.quiet:  # as the stash steps
            m.quiet = True
            self.qtrace, self.qout = [], []
            for ph in range(4):
                m.set_inputs(0, ph)
                self._stage_restore(snap)
                self.qtrace.append(ttnn.begin_trace_capture(self.device, cq_id=0))
                self.qout.append(m.step())
                self._stage_topup(snap)
                ttnn.end_trace_capture(self.device, self.qtrace[-1], cq_id=0)
                self._stage_snap(snap)
                self._stage_restore(snap)
            m.quiet = False
            say("spec quiet captured", flush=True)
        for V in Vs:
            R = V.R
            for ph in range(4):
                for shared in (True, False):
                    # a position of conv phase ph in this far-chunk case (a chunk-crossing pass exists only at some
                    # phases: R tokens cross a 128-key chunk only from its last R-1 offsets)
                    p = next((q for q in range(4, p_shared + 512) if q % 4 == ph and self._shared(q, R) == shared),
                             None)  # fmt: skip
                    if p is None:
                        continue
                    V.set_positions(p)
                    self._stage_begin(snap)
                    tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                    self._feed(V)
                    out = V.step()
                    ttnn.copy(out["token"][0], self.vtok[R])
                    self._stage_topup(snap)
                    ttnn.end_trace_capture(self.device, tid, cq_id=0)
                    self._stage_restore(snap)
                    self.vtrace[(R, ph, shared)] = (tid, out)
                    say("spec verify captured", R, ph, shared, flush=True)
                for n in range(1, R + 1):
                    V.set_inputs([0] * R, p_shared - p_shared % 4 + ph)
                    tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                    V.commit(n)
                    self._set_anchor(R, n)
                    ttnn.end_trace_capture(self.device, tid, cq_id=0)
                    self._stage_restore(snap)
                    self.ctrace[(R, ph, n)] = tid
        ttnn.synchronize_device(self.device)

    # ------------------------------------------------------------ decode staging across traces (tt/mk/overlap.py)
    # A staged projection reads the prefix of the shared staging tensor that the host programs before it filled, as
    # recorded at capture. The spec loop replays plain, draft, verify and commit traces in any order, so every trace
    # starts from the state a plain step ends with (snap: app 0's prefix filled up to a fraction, its fill queued),
    # and a verify trace whose own tail filled less of app 0 fills the rest at its end (_stage_topup).
    def _stage_snap(self, like=None):
        ov = self.overlap
        if ov is None:
            return None
        snap = [(t, t.ptr, t.filled, t.filled_for) for t in ov.tracks], list(ov.queue)
        if like is not None:  # same app, same queued items; the filled fraction may differ (a runtime arg)
            assert [(t, p, f) for t, p, _, f in snap[0]] == [(t, p, f) for t, p, _, f in like[0]]
            assert [id(q[0]) for q in snap[1]] == [id(q[0]) for q in like[1]], "staging queue differs"
        return snap

    def _stage_begin(self, snap):
        """Before a verify pass (eager or captured)."""
        from models.demos.inkling2_mini.tt import verify as VF

        self._stage_restore(snap)
        if snap is not None and not VF.STAGE:  # raw-bank verify (A/B): its programs carry no fills
            self.overlap.queue = []

    def _stage_restore(self, snap):
        if snap is None:
            return
        tracks, queue = snap
        for t, ptr, filled, filled_for in tracks:
            assert t.ptr == ptr, ("staging: app order out of step after a trace", t.name, t.ptr, ptr)
            t.filled, t.filled_for = filled, filled_for
        self.overlap.queue = list(queue)

    def _stage_topup(self, snap):
        if snap is None:
            return
        from models.demos.inkling2_mini.tt import verify as VF

        if not VF.STAGE:
            return
        for t, ptr, filled, proxy in snap[0]:
            if proxy is None:
                continue
            # always the same program (a pass that filled enough tops up zero bytes)
            have = min(filled, t.filled if t.filled_for is proxy else 0.0)
            prog = ttnn.ProgramDescriptor(kernels=proxy.st.fill_kernels((have, filled)), semaphores=[], cbs=[])
            ttnn.generic_op(proxy.st.fill_io, prog)

    def generate_rows(self, prompt_ids, max_new, on_pass=None, early_stop=None):
        """Greedy tokens through draft + one verify pass + commit per pass; output identical to Generator.generate.
        early_stop=None: every pass verifies the largest verify step's R rows (k = R - 1 drafts). early_stop=thr
        (needs drafter.top_probs and verify steps for each row count): keep draft i while the product of the drafts'
        top probs up to i is >= thr; 0 kept runs a plain step. Returns (new ids, stats)."""
        self.model.reset()
        out = None
        for pos, t in enumerate(prompt_ids):
            out = self.decode(t, pos, True)
            if pos % 64 == 63:
                self.read_token(out)
        return self.generate_rows_from(self.read_token(out), len(prompt_ids), max_new, on_pass, early_stop)

    def generate_rows_from(self, nxt, pos, max_new, on_pass=None, early_stop=None):
        """generate_rows from the model's present state (after decode steps, or a chunked prefill with the drafter
        feed): nxt = the token at position pos. Returns ([nxt] + the new ids, stats)."""
        import time as _t

        m, d = self.model, self.drafter
        # the anchor of the first draft: a decode step leaves its argmax there, a prefill does not
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[nxt]], dtype=torch.int32), dtype=ttnn.uint32), d.anchor
        )
        kmax = max(self.verifies) - 1
        assert d.k >= kmax and (early_stop is None or d.top_probs)
        new, acc, rows, t_draft, t_ver = [nxt], [], [], 0.0, 0.0
        while len(new) < max_new:
            t0 = _t.perf_counter()
            d.set_draft_pos(pos)
            ttnn.execute_trace(self.device, d.trace, cq_id=0, blocking=False)
            for V in self.verifies.values():
                V.set_positions(pos)  # host writes while the draft runs
            k = kmax
            if early_stop is not None:
                # one extra host sync per pass: the row count depends on the drafts' probs
                pr = ttnn.to_torch(d.probs).float()[0, 0, 0, :kmax].tolist()
                k, c = 0, 1.0
                while k < kmax and c * pr[k] >= early_stop:
                    c *= pr[k]
                    k += 1
            t1 = _t.perf_counter()
            ph = pos % 4
            if k == 0:  # plain step (writes the next anchor itself)
                o = self.decode(nxt, pos, True)
                tgt, drafts, a = [self.read_token(o)], [], 0
            else:
                R = k + 1
                ttnn.execute_trace(self.device, self.vtrace[(R, ph, self._shared(pos, R))][0], cq_id=0,
                                   blocking=False)  # fmt: skip
                tgt = [int(x) for x in ttnn.to_torch(self.vtok[R]).reshape(-1)[:R].tolist()]
                drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()]
                a = 0
                while a < k and drafts[a] == tgt[a]:
                    a += 1
                ttnn.execute_trace(self.device, self.ctrace[(R, ph, a + 1)], cq_id=0, blocking=False)
            t_draft += t1 - t0
            t_ver += _t.perf_counter() - t1
            new += tgt[: a + 1]
            acc.append(a)
            rows.append(k + 1)
            nxt, pos = tgt[a], pos + a + 1
            if on_pass:
                on_pass(a)
        ttnn.synchronize_device(self.device)
        stats = {"passes": len(acc), "accepted": acc, "rows": rows, "draft_s": t_draft, "verify_s": t_ver}
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


# DRAM free (GiB) after the model and the drafter that the verify steps need with their traces: 2-row passes;
# 2- and 3-row passes. The far KV and the drafter context grow with max_ctx (19,584 + 12,288 B per token).
VERIFY_NEED_GIB = (0.46, 0.74)


# The drafter beside a built model: its weights and head copy, and per covered position the context K / V (one
# full-length layer; the two local layers are rings) and the RoPE rows.
DRAFTER_GIB = (0.83, 4608 / 2**30)  # measured at 143360 positions: 0.55 + 0.28 (BFP4 head copy) + 0.62 GiB


def spec_fits(free_gib, spec_ctx):
    """True when the drafter for spec_ctx positions and the 2-row verify step fit in free_gib of DRAM."""
    return free_gib >= DRAFTER_GIB[0] + DRAFTER_GIB[1] * spec_ctx + VERIFY_NEED_GIB[0]


def dram_free_gib(device):
    v = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_free_per_bank / 2**30


def verify_rows(device, max_ctx):
    """Row counts of the verify steps that fit beside a built model + drafter: [2, 3], or [2] when DRAM is short
    for the 3-row step. ValueError when not even the 2-row step fits."""
    free = dram_free_gib(device)
    if free < VERIFY_NEED_GIB[0]:
        raise ValueError(
            f"speculative decode does not fit at max_ctx {max_ctx}: {free:.2f} GiB DRAM free after the model and "
            f"the drafter, the verify step needs {VERIFY_NEED_GIB[0]}"
        )
    return [2, 3] if free >= VERIFY_NEED_GIB[1] else [2]


def build(device, max_ctx, log=print, mtp_root=None, k=7, spec_ctx=None, **kw):
    """TappedModel + its drafter (Inkling2Model.__init__ never steps, so the drafter can be attached after).
    spec_ctx: the positions the drafter covers when that is less than max_ctx (drafts and verify passes below it,
    TappedModel.quiet steps from it on)."""
    model = TappedModel(device, max_ctx=max_ctx, log=log, **kw)
    model.drafter = DSparkDrafter(model, root=mtp_root, max_ctx=min(max_ctx, spec_ctx or max_ctx), k=k)
    return model, model.drafter
