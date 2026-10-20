# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of deferred serving in the spec engine (QWEN36_SPEC_CHAIN + QWEN36_SPEC_FUSED, stage S4b, work/notes/63 section 2):
``decode_forward(defer_read=True)`` enqueues a fused step (the chain step for ``chain_rows``), enqueues non-blocking reads, records an
event and returns a FusedHandle with no synchronize; the reader thread's ``handle.wait()`` blocks on the event and the engine thread's
``propose_draft_tokens(hidden=handle)`` does the bookkeeping later. Fake device, stream-ordered fake host copies, call log (the
approach of test_spec_chain_engine_host.py, whose rig this one extends). No device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_async_engine_host.py
"""

import threading

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine_host import (
    _GOLDEN_FLAG_OFF_DIGEST,
    CHG,
    CHS,
    _ChainRig,
    _flag_off_scenario,
    _Flow,
    _strip_kv,
)
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    FA,
    NB,
    VFY,
    K,
    T,
    _block,
    _ids_for,
    _pt,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tt.spec_chain import OVR_WORDS, fused_fit_mode_bounds
from models.demos.blackhole.qwen36.tt.spec_engine import FusedHandle
from models.demos.blackhole.qwen36.tt.spec_fused import fused_fit_mode


class _Copy:
    """A non-blocking D2H copy: the data is snapshotted when the copy is ENQUEUED (stream order: later fake traces, which run at
    enqueue time here, must not change it) and handed to the host only when an event / synchronize that follows the enqueue is waited.
    """

    def __init__(self, t, seq):
        self.name, self.snap, self.seq, self.data = t.name, (None if t.data is None else t.data.clone()), seq, None


class _AsyncRig(_ChainRig):
    """_ChainRig + events + stream-ordered host copies + a verify trace that 'computes' its argmax ids / logits from the token
    buffer it finds when it runs (``queue``: ("acc", accept vector) or ("logits", [R, V]) per verify, consumed in order), so a chained
    step (whose tokens only the device knows) gets its ids too."""

    def __init__(self, monkeypatch, B, chain=True):
        super().__init__(monkeypatch, B, chain=chain)
        monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")  # these tests defer sampled steps (stage S4b)
        self.queue, self.copies, self.n_copies, self.threads = [], [], 0, set()
        log = self.log

        def from_device(t, blocking=True):
            log.append(("from_device", t.name, blocking))
            if blocking:
                return t
            c = _Copy(t, self.n_copies)
            self.n_copies += 1
            self.copies.append(c)
            return c

        def fill(upto):
            for c in [c for c in self.copies if c.seq < upto]:
                c.data = c.snap
                self.copies.remove(c)

        def record_event(dev, cq_id=0):
            log.append(("record_event",))
            return ("event", self.n_copies)

        def event_synchronize(ev):
            log.append(("event_sync",))
            self.threads.add(threading.get_ident())
            fill(ev[1])

        def sync(dev):
            log.append(("sync",))
            fill(self.n_copies)

        monkeypatch.setattr(ttnn, "from_device", from_device)
        monkeypatch.setattr(ttnn, "record_event", record_event)
        monkeypatch.setattr(ttnn, "event_synchronize", event_synchronize)
        monkeypatch.setattr(ttnn, "synchronize_device", sync)

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        super()._exec(dev, tid, cq_id, blocking)
        if tid == VFY and self.queue:
            kind, v = self.queue.pop(0)
            if kind == "logits":
                self.set_logits(v)
            else:
                tok = self.m._vfy_token_buf.data.reshape(self.B, T)
                aux = self.fa.aux_in.data
                nv = aux[0, : self.B] if aux is not None else torch.zeros(self.B, dtype=torch.int32)
                self.set_ids(_ids_for(tok, nv, v))

    def syncs(self):
        return [x for x in self.log if x[0] == "sync"]


class _Pipe:
    """The plugin side of a deferred pipeline: it builds the first block from host state, chain blocks (placeholders, provisional
    positions = applied length + K+1 per verify in flight, counts -1) without the previous step's commit, reads a step (reader thread's
    handle.wait()) and proposes from its handle (engine thread)."""

    def __init__(self, r, B, nv, base=50, stride=10):
        self.r, self.B = r, B
        self.tok, self.pos, self.nvd = _block(B, nv, base=base, stride=stride)
        self.acc, self.pt = torch.ones(B, dtype=torch.int32), _pt(B)
        self.up = self.pos[:, 0].clone().to(torch.int64)  # upper bound of the next column-0 position
        self.live = [int(p) >= 0 for p in self.pos[:, 0]]
        self.tips = []

    def enq(self, prev=None, sampling=None, absent=(), nv_hint=None, pt=None, finishing=None, remap=None):
        B, r = self.B, self.r
        kw = {} if sampling is None else {"spec_sampling": sampling}
        if (
            remap is not None
        ):  # the plugin's slot_remap: slot i takes the request that sat at slot remap[i]; the block is post-remap
            kw["slot_remap"] = list(remap)
            self.live = [self.live[int(j)] for j in remap]
            self.up = self.up[torch.tensor(list(remap), dtype=torch.int64)]
        if finishing is not None:
            kw["finishing_slots"] = list(finishing)
        pt = self.pt if pt is None else pt
        if prev is None:
            tok, pos, nvd, acc = self.tok, self.pos, self.nvd, self.acc
        else:
            self.up = self.up + T
            tok = torch.full((B, T), -1, dtype=torch.int32)
            pos = torch.full((B, T), -1, dtype=torch.int32)
            nvd = torch.zeros(B, dtype=torch.int32)
            acc = torch.ones(B, dtype=torch.int32)
            cr = torch.zeros(B, dtype=torch.bool)
            for u in range(B):
                if self.live[u] and u not in absent:
                    cr[u] = True
                    pos[u] = int(self.up[u]) + torch.arange(T, dtype=torch.int32)
                    nvd[u] = K if nv_hint is None else int(nv_hint[u])
                    acc[u] = -1
                else:
                    tok[u] = 0
            kw.update(chain_rows=cr, chain_tip=prev.tip_id)
        res = r.e.decode_forward(
            tok,
            pos,
            num_valid_drafts=nvd,
            accepted_counts=acc,
            page_table=pt,
            spec_mode="fused_sample",
            defer_read=True,
            **kw,
        )
        assert isinstance(res.handle, FusedHandle) and res.argmax_ids is None
        return res.handle

    def resolve(self, h, release=()):
        """Reader thread wait, then the engine thread's propose from the handle (with what the plugin would pass)."""
        ids, counts, p0 = h.wait()
        cpos = torch.full_like(ids, -1)
        for u in range(self.B):
            if int(p0[u]) >= 0:
                cpos[u, : int(counts[u])] = int(p0[u]) + 1 + torch.arange(int(counts[u]), dtype=torch.int32)
        dr = self.r.e.propose_draft_tokens(K, ids.clone(), cpos, counts.clone(), hidden=h)
        for u in release:
            self.r.e.release(u)
            self.live[u] = False
        return ids, counts, p0, dr


def _run_pipe(mp, B, nv, queue, sampling=None, finish=None, absent=None, chain=True):
    """enqueue i (chained on i-1), then resolve i-1: two steps in flight. ``finish`` = (step, row): the request ends at that step, its
    row leaves from step+2 (it ran one wasted lane-step in step+1). ``absent`` = (step, row): a live row is left out of that step (park).
    """
    r = _AsyncRig(mp, B, chain=chain)
    r.queue = list(queue)
    p = _Pipe(r, B, nv)
    out, pend = [], []
    n = len(queue)
    for i in range(n):
        ab = (absent[1],) if absent is not None and i == absent[0] else ()
        if absent is not None and i > absent[0]:
            p.live[absent[1]] = False
        h = p.enq(pend[-1] if pend else None, sampling=None if sampling is None else sampling[i], absent=ab)
        pend.append(h)
        if len(pend) == 2:
            h0 = pend.pop(0)
            j = len(out)
            out.append(p.resolve(h0, release=(finish[1],) if finish is not None and j == finish[0] else ()))
    while pend:
        out.append(p.resolve(pend.pop(0)))
    return r, p, out


def _run_sync(mp, B, nv, queue, sampling=None, finish=None, leave=None):
    r = _AsyncRig(mp, B)
    r.queue = list(queue)
    f = _Flow(r, B, list(nv)[:B])
    out = []
    for i in range(len(queue)):
        f.step(sampling=None if sampling is None else sampling[i])
        out.append((f.res.accepted_token_ids.clone(), f.res.accepted_counts.clone()))
        dr = f.propose()
        out[-1] += (dr,)
        f.next_block(
            finish=(finish[1],) if finish is not None and i == finish[0] else (),
            leave=(leave[1],) if leave is not None and i == leave[0] else (),
        )
    return r, f, out


def _accepts(B, n, seed):
    g = torch.Generator().manual_seed(seed)
    return [("acc", torch.randint(0, 4, (B,), generator=g).tolist()) for _ in range(n)]


def _logits_q(B, n, seed):
    return [("logits", _smooth_logits(B * T, seed * 100 + i)) for i in range(n)]


def _samplings(B, n):
    temps = [[0.8, 0.0, 1.2, 0.9][:B], [0.9, 0.9, 0.0, 0.7][:B], [0.0, 1.1, 0.8, 0.8][:B]]
    return [_sampling(B, [K] * B, temps[i % 3], seed=i) for i in range(n)]


# --------------------------------------------------------------------------- #
# (i) the deferred call logs: no synchronize, an event at the end; the read does an event wait and host conversions only
# --------------------------------------------------------------------------- #
def test_deferred_step_call_logs_have_no_synchronize_and_end_with_the_event(monkeypatch):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 3, 1)
    p = _Pipe(r, B, [3, 2, 1])
    del r.log[:]
    h1 = p.enq()
    log1 = r.log[:]
    assert ("sync",) not in log1 and log1[-1] == ("record_event",)
    assert [x[:2] for x in log1] == [
        ("h2d", "fa_aux"),
        ("h2d", "fa_ptrows"),
        ("h2d", "mtp"),
        ("h2d", "kvpt"),
        ("h2d", "kvpt_users"),
        ("h2d", "kvpt1"),
        ("h2d", "token"),
        ("h2d", "aux"),
        ("trace", VFY),
        ("trace", FA),
        ("trace", DRAFT),
        ("from_device", "fa_result"),
        ("from_device", "fa_aux"),
        *[("from_device", f"out{k}") for k in range(K)],
        ("from_device", "ids"),  # the S1 greedy check's read (QWEN36_SPEC_FUSED_AUDIT, default on)
        ("record_event",),
    ], log1
    assert all(x[2] is False for x in log1 if x[0] == "from_device") and h1.tip_id == 1
    del r.log[:]
    h2 = p.enq(h1)
    log2 = r.log[:]
    assert ("sync",) not in log2 and log2[-1] == ("record_event",)
    assert [x[:2] for x in _strip_kv(log2)] == [
        ("h2d", "ovr_g"),  # same table, same declined rows: only the overlay
        ("trace", CHG),
        ("from_device", "token"),  # the chain audit copies, right behind the chain trace
        ("from_device", "aux"),
        ("from_device", "fa_aux"),
        ("trace", VFY),
        ("trace", FA),
        ("trace", DRAFT),
        ("from_device", "fa_result"),
        ("from_device", "fa_aux"),
        *[("from_device", f"out{k}") for k in range(K)],
        ("from_device", "ids"),
        ("record_event",),
    ], log2
    assert not {"token", "aux"} & set(r.names("h2d")) and h2.tip_id == 2 and r.e.chain_steps == 1
    # the reads: an event wait, then host conversions only (no device call, no engine state)
    snapshot = (r.e._offered.copy(), r.e.mtp_extra_steps, r.e.chain_audits, r.e._no_draft.copy(), r.e._last_pt.clone())
    del r.log[:]
    ro = h1.wait()
    assert r.log[0] == ("event_sync",) and {x[0] for x in r.log[1:]} == {"to_torch"}, r.log
    assert ("sync",) not in r.log
    assert (r.e._offered, r.e.mtp_extra_steps, r.e.chain_audits, r.e._no_draft) == snapshot[:4] and torch.equal(
        r.e._last_pt, snapshot[4]
    )
    assert [t.dtype for t in ro] == [torch.int32] * 3 and ro[0].shape == (B, T) and ro[1].shape == ro[2].shape == (B,)
    del r.log[:]
    assert h1.wait() is ro or all(torch.equal(a, b) for a, b in zip(h1.wait(), ro))
    assert r.log == [], "a second wait reads the per-handle cache"
    # the engine thread's propose does the bookkeeping, no device call at all
    del r.log[:]
    dr = r.e.propose_draft_tokens(K, ro[0], torch.zeros_like(ro[0]), ro[1], hidden=h1)
    assert r.log == [] and dr is r.e.propose_draft_tokens(K, ro[0], torch.zeros_like(ro[0]), ro[1], hidden=h1)
    h2.wait()
    r.e.propose_draft_tokens(K, ro[0], torch.zeros_like(ro[0]), ro[1], hidden=h2)
    assert r.e.chain_audits == 1


def test_reads_happen_in_stream_order_not_after_later_steps(monkeypatch):
    """A step's host copies are filled by ITS event, and show what the device held when they were enqueued, not what a later
    enqueued step wrote."""
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 3, 2)
    p = _Pipe(r, B, [3, 2, 1])
    h1 = p.enq()
    h2 = p.enq(h1)
    h3 = p.enq(h2)
    ro1 = h1.wait()
    assert h2._data is None and h3._data is None
    ro2, ro3 = h2.wait(), h3.wait()
    assert not torch.equal(ro1[2], ro2[2]) and not torch.equal(ro2[2], ro3[2])  # p0 advanced between the steps


# --------------------------------------------------------------------------- #
# (ii) the pipeline equals the sync chain path
# --------------------------------------------------------------------------- #
def _compare(out_p, out_s, rows_of=lambda i: None, upto=None):
    n = len(out_p)
    for i in range(n):
        ids_p, cnt_p, p0_p, dr_p = out_p[i]
        ids_s, cnt_s, dr_s = out_s[i]
        for u in range(ids_p.shape[0]) if rows_of(i) is None else rows_of(i):
            assert int(cnt_p[u]) == int(cnt_s[u]), f"step {i} row {u} counts"
            c = int(cnt_p[u])
            assert torch.equal(ids_p[u, :c], ids_s[u, :c]), f"step {i} row {u} tokens"
            n_dr = int(dr_s.num_valid[u])
            assert int(dr_p.num_valid[u]) == n_dr, f"step {i} row {u} num_valid"
            assert torch.equal(
                dr_p.draft_token_ids[u, :n_dr], dr_s.draft_token_ids[u, :n_dr]
            ), f"step {i} row {u} drafts"


def _same_bookkeeping(rp, rs):
    a, b = rp.e, rs.e
    assert (a._offered, a._no_draft, a._speculated, a._withheld_streak) == (
        b._offered,
        b._no_draft,
        b._speculated,
        b._withheld_streak,
    )
    assert (a.mtp_extra_steps, a.fused_steps, a.chain_steps, a.chain_audits) == (
        b.mtp_extra_steps,
        b.fused_steps,
        b.chain_steps,
        b.chain_audits,
    )
    assert torch.equal(a._last_pt, b._last_pt) and torch.equal(a._row_active, b._row_active)
    assert rp.m._vfy_gdn[0]._spec_win_par == rs.m._vfy_gdn[0]._spec_win_par


@pytest.mark.parametrize("B,nv", [(2, (3, 2)), (3, (3, 2, 1)), (4, (3, 3, 3, 3))])
def test_pipeline_equals_the_sync_chain_path_greedy(monkeypatch, B, nv):
    n = 10
    q = _accepts(B, n, B)
    rp, p, out_p = _run_pipe(monkeypatch, B, nv, q)
    rs, f, out_s = _run_sync(monkeypatch, B, nv, q)
    _compare(out_p, out_s)
    _same_bookkeeping(rp, rs)
    assert rp.e.chain_breaks == {} and rs.e.chain_breaks == {"first": 1}
    assert len(rp.syncs()) == 0, "no synchronize_device in the whole pipeline"
    assert rp.e.chain_steps == n - 1 and rp.e.chain_audits == n - 1
    assert (rp.m._vfy_aux_buf.data.to(torch.int64) & 0xFFFFFFFF).equal(
        rs.m._vfy_aux_buf.data.to(torch.int64) & 0xFFFFFFFF
    )


@pytest.mark.parametrize("B", [2, 3, 4])
def test_pipeline_equals_the_sync_chain_path_sampled(monkeypatch, B):
    n = 9
    q = _logits_q(B, n, B)
    smp = _samplings(B, n)
    rp, p, out_p = _run_pipe(monkeypatch, B, (3, 2, 1, 3)[:B], q, sampling=smp)
    rs, f, out_s = _run_sync(monkeypatch, B, (3, 2, 1, 3)[:B], q, sampling=smp)
    _compare(out_p, out_s)
    _same_bookkeeping(rp, rs)
    assert rp.e.chain_steps == n - 1 and rp.e.fused_audited_steps == rs.e.fused_audited_steps == n
    assert rp.e.fused_sampled_steps == n and len(rp.syncs()) == 0
    assert {CHG, CHS} & set(rp.ch_runs) and rp.e.chain_breaks == {}


def test_pipeline_with_a_finishing_row_overlay(monkeypatch):
    """Row 1's request ends at step 3: step 4 (already enqueued, chained) runs it for nothing, step 5 has it as a host (overlay)
    row and the chain goes on; every other row's stream and drafts equal the sync run in which row 1 left after step 3.
    """
    B, n = 3, 10
    q = _accepts(B, n, 7)
    rp, p, out_p = _run_pipe(monkeypatch, B, (3, 2, 1), q, finish=(3, 1))
    rs, f, out_s = _run_sync(monkeypatch, B, (3, 2, 1), q, finish=(3, 1))
    _compare(out_p, out_s, rows_of=lambda i: [0, 1, 2] if i <= 3 else [0, 2])
    assert rp.e.chain_breaks == {} and rp.e.chain_steps == n - 1 and len(rp.syncs()) == 0
    assert 1 not in rp.e._active and not bool(rp.e._row_active[1])


def test_a_park_breaks_the_chain_materializes_and_the_chain_restarts(monkeypatch):
    """Row 2 is left out of step 4 without a release: its state must be parked on the host, which needs the device idle. The chain
    step cannot chain: the engine synchronizes, builds the chain rows' host inputs from the pending step, runs it as an ordinary
    step (the park runs inside verify), and the next step chains again. The other rows' streams equal the sync run."""
    B, n = 3, 9
    q = _accepts(B, n, 3)
    rp, p, out_p = _run_pipe(monkeypatch, B, (3, 2, 1), q, absent=(4, 2))
    rs, f, out_s = _run_sync(monkeypatch, B, (3, 2, 1), q, leave=(4, 2))
    _compare(out_p, out_s, rows_of=lambda i: [0, 1, 2] if i < 4 else [0, 1])
    assert rp.e.chain_breaks == {"park": 1}, rp.e.chain_breaks
    assert 2 in rp.e._parked and len(rp.syncs()) == 1  # the one synchronize is the materialize
    assert rp.e.chain_steps == n - 2  # every step but the first and the broken one chained
    assert rs.e.chain_breaks.get("park") == 1


# --------------------------------------------------------------------------- #
# (iii) chain_tip validation
# --------------------------------------------------------------------------- #
def test_wrong_chain_tip_raises(monkeypatch, expect_error):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 4, 4)
    p = _Pipe(r, B, [3, 2, 1])
    h1 = p.enq()
    h2 = p.enq(h1)
    with expect_error(ValueError, "chain_tip"):  # h1 is not the newest fused step any more
        p.enq(h1)
    p.up = p.up - T
    n_logs = len(r.log)
    assert n_logs and r.e._def_last is h2
    cr = torch.ones(B, dtype=torch.bool)
    tok, pos = torch.full((B, T), -1, dtype=torch.int32), p.up[:, None] + 2 * T + torch.arange(T)[None, :]
    kw = dict(
        num_valid_drafts=torch.full((B,), K, dtype=torch.int32),
        accepted_counts=torch.full((B,), -1, dtype=torch.int32),
        page_table=p.pt,
        spec_mode="fused_sample",
        defer_read=True,
    )
    for tip in (None, 99, h2.tip_id + 1):
        with expect_error(ValueError, "chain_tip"):
            r.e.decode_forward(tok, pos.to(torch.int32), chain_rows=cr, chain_tip=tip, **kw)
    with expect_error(ValueError, "chain_tip without chain_rows"):
        r.e.decode_forward(tok, pos.to(torch.int32), chain_tip=h2.tip_id, **kw)
    with expect_error(ValueError, "need defer_read"):
        r.e.decode_forward(tok, pos.to(torch.int32), chain_rows=cr, chain_tip=h2.tip_id, **{**kw, "defer_read": False})
    with expect_error(ValueError, "placeholder"):  # a chain row with real tokens
        r.e.decode_forward(
            torch.ones(B, T, dtype=torch.int32), pos.to(torch.int32), chain_rows=cr, chain_tip=h2.tip_id, **kw
        )
    assert r.e._def_last is h2, "a refused call enqueued nothing"
    p.up = p.up + T
    h3 = p.enq(h2)  # the right tip works
    assert h3.tip_id == h2.tip_id + 1


def test_a_chain_step_after_a_fallback_step_builds_its_inputs_from_the_host_result(monkeypatch):
    """A fallback step (here: not captured, argmax verify + host walk) runs synchronously and hands back a ready handle; a step
    chained on it cannot chain on the device: the engine proposes from the host result and runs the step host-staged."""
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 4, 5)
    e = r.e
    p = _Pipe(r, B, [3, 2, 1])
    saved = e._fa_trace
    e._fa_trace = None  # not_captured: the first step falls back
    h1 = p.enq()
    assert not h1.device and h1.ready and len(r.syncs()) == 1 and e.fused_fallbacks == {"not_captured": 1}
    e._fa_trace = saved
    ro1 = h1.wait()
    assert h1.wait() is not None and [t.dtype for t in ro1] == [torch.int32] * 3
    h2 = p.enq(h1)
    assert e.chain_breaks == {"tip_not_chainable": 1} and h2.device and h2.cmask is None
    # the ready handle's propose ran inline (the ordinary propose): the plugin's own propose returns the same result
    dr1 = p.resolve(h1)[3]
    assert dr1 is h1.draft_result
    p.resolve(h2)
    h3 = p.enq(h2)  # a device step: the next one chains
    assert h3.cmask is not None and e.chain_steps == 1


def test_mixed_fit_chain_step_materializes(monkeypatch):
    """Rows near the end of the page table: whether the next step's drafts fit depends on how many drafts the pending step accepted
    (fused_fit_mode_bounds says mixed), so the step cannot be enqueued blind."""
    B = 2
    r = _AsyncRig(monkeypatch, B)
    e = r.e
    cap = NB * e.block_size
    r.queue = _accepts(B, 3, 6)
    p = _Pipe(r, B, [3, 3], base=cap - 8, stride=0)
    h1 = p.enq()
    h2 = p.enq(h1)
    assert (
        e.chain_breaks.get("mixed_fit") == 1 and h2.cmask is None and len(r.syncs()) >= 1
    )  # the materialize (+ the fallback verify if the exact start is itself mixed)
    assert [h1.wait()[1].dtype, h2.wait()[1].dtype] == [torch.int32] * 2


# --------------------------------------------------------------------------- #
# (iv) parity at enqueue time
# --------------------------------------------------------------------------- #
def test_parity_flips_once_per_replay_at_enqueue_and_the_overlay_carries_it(monkeypatch):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 7, 8)
    p = _Pipe(r, B, [3, 2, 1])
    par = lambda: {dn._spec_win_par for dn in r.m._vfy_gdn}
    seen, prev, pend = [], None, []
    for i in range(6):
        before = par()
        h = p.enq(prev)
        assert par() == {(i + 1) % 2} and before == {i % 2}, "flipped at enqueue, not at the read"
        if i:
            assert (
                int(r.e._ch_g.ovr_in.data[OVR_WORDS, 0]) == i % 2
            )  # the overlay's parity word = the parity the replay reads
        assert r.m._vfy_aux_buf.data[2, 0].item() == i % 2  # the ctrl page the replay read
        seen.append((before, par()))
        pend.append(h)
        prev = h
        if len(pend) == 2:
            p.resolve(pend.pop(0))
            assert par() == {(i + 1) % 2}, "a read or a propose never flips"
    while pend:
        p.resolve(pend.pop(0))
    assert par() == {0}
    # a synchronous step after the deferred ones continues the sequence
    assert r.m.vfy_gdn_parity() == 0


# --------------------------------------------------------------------------- #
# (v) the deferred chain audit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("what,match", [("token", "verify token buffer differs"), ("aux", "verify aux row")])
def test_deferred_chain_audit_catches_a_corrupted_device_word(monkeypatch, what, match, expect_error):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 3, 9)
    p = _Pipe(r, B, [3, 2, 1])
    h1 = p.enq()
    p.resolve(h1)
    r.corrupt_ch = what
    h2 = p.enq(h1)  # enqueue-only: nothing is checked yet
    with expect_error(RuntimeError, match):
        h2.wait()
    assert h2._data is None


def test_deferred_chain_audit_every_n_and_off(monkeypatch):
    B = 3
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "3")
    r, p, out = _run_pipe(monkeypatch, B, (3, 2, 1), _accepts(B, 8, 10))
    assert r.e.chain_steps == 7 and r.e.chain_audits == 3  # chained steps 0, 3, 6
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "0")
    r, p, out = _run_pipe(monkeypatch, B, (3, 2, 1), _accepts(B, 5, 10))
    assert r.e.chain_audits == 0 and r.e.chain_steps == 4
    assert not [x for x in r.log if x[0] == "from_device" and x[1] in ("token", "aux")]


def test_greedy_check_at_read_time_catches_a_wrong_device_accept(monkeypatch, expect_error):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 2, 11)
    r.corrupt = True  # the fake accept trace writes a wrong count
    p = _Pipe(r, B, [3, 2, 1])
    h = p.enq()
    with expect_error(RuntimeError, "fused accept != host greedy walk"):
        h.wait()


def test_sampled_overflow_in_deferred_mode_fails_closed(monkeypatch, expect_error):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    lg = _smooth_logits(B * T, 3)
    lg[0, :] = 1.0  # a flat row: the top-k tie group reaches the last union candidate
    r.queue = [("logits", lg)]
    p = _Pipe(r, B, [3, 2, 1])
    h = p.enq(sampling=_sampling(B, p.nvd, [0.8, 0.0, 1.2]))
    with expect_error(RuntimeError, "tie group"):
        h.wait()


def test_deferred_sampled_step_needs_the_chain_sampled_flag(monkeypatch, expect_error):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    monkeypatch.delenv("QWEN36_SPEC_CHAIN_SAMPLED", raising=False)
    r.queue = _accepts(B, 2, 1)
    p = _Pipe(r, B, [3, 2, 1])
    with expect_error(ValueError, "QWEN36_SPEC_CHAIN_SAMPLED"):
        p.enq(sampling=_sampling(B, p.nvd, [0.8, 0.0, 1.2]))
    h = p.enq()  # no spec_sampling: a greedy-only step still defers
    h.wait()


# --------------------------------------------------------------------------- #
# threads
# --------------------------------------------------------------------------- #
def test_reads_run_on_a_reader_thread_while_the_engine_thread_goes_on(monkeypatch):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 6, 12)
    p = _Pipe(r, B, [3, 2, 1])
    h1 = p.enq()
    h2 = p.enq(h1)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("ro", (h1.wait(), h2.wait())))
    t.start()
    t.join(10)
    assert not t.is_alive() and len(r.threads) == 1 and threading.get_ident() not in r.threads
    ro1, ro2 = box["ro"]
    dr1 = r.e.propose_draft_tokens(
        K, ro1[0], torch.zeros_like(ro1[0]), ro1[1], hidden=h1
    )  # engine thread, after the read
    dr2 = r.e.propose_draft_tokens(K, ro2[0], torch.zeros_like(ro2[0]), ro2[1], hidden=h2)
    h3 = p.enq(h2)
    assert h3.cmask is not None and dr1.num_valid.shape == (B,) and dr2 is h2.draft_result


# --------------------------------------------------------------------------- #
# bounds, window invariant, flags off
# --------------------------------------------------------------------------- #
def test_fused_fit_mode_bounds_equals_brute_force_and_the_exact_forms():
    for cap in (40, 64):
        for K_ in (3, 7):
            for a in range(0, cap + 3):
                for b in range(a, min(cap + 3, a + 4 * K_ + 4)):
                    got = fused_fit_mode_bounds([a], [b], cap, [True], K_)[0]
                    fits = [s + m + 1 + K_ < cap for s in range(a, b + 1) for m in range(K_ + 1)]
                    want = "fits_all" if all(fits) else "fits_none" if not any(fits) else "mixed"
                    assert got == want, (cap, K_, a, b)
                    if a == b:
                        assert got == fused_fit_mode([a], cap, [True], K_)[0]
    assert fused_fit_mode_bounds([5], [5], 64, [False], 3) == ["fits_none"]


def test_window_block_invariant(monkeypatch):
    r = _AsyncRig(monkeypatch, 3)
    e = r.e
    pt = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=torch.int32)
    act, nv = torch.ones(3, dtype=torch.bool), torch.full((3,), 3)
    lo = torch.tensor([60, 10, 10])
    hi = torch.tensor([70, 20, 20])
    assert e._def_window_blocks_disjoint(pt, act, lo, hi, nv)
    pt2 = pt.clone()
    pt2[1, 0] = 12  # row 1's only window block is row 0's second block
    assert not e._def_window_blocks_disjoint(pt2, act, lo, hi, nv)
    assert e._def_window_blocks_disjoint(
        pt2, torch.tensor([True, False, True]), lo, hi, nv
    )  # an inactive row writes nothing
    pt3 = pt.clone()
    pt3[1, 1] = 12  # shared, but outside the blocks row 1 may write (a prefix-cache block): fine
    assert e._def_window_blocks_disjoint(pt3, act, lo, hi, nv)
    pt3[1, 0] = 0  # a null block is never a conflict
    pt3[0, 1] = 0
    assert e._def_window_blocks_disjoint(pt3, act, lo, hi, nv)


def test_flag_off_call_logs_are_the_pre_chain_call_logs_and_decode_forward_unchanged(monkeypatch, expect_error):
    import hashlib

    for n in ("QWEN36_SPEC_CHAIN", "QWEN36_SPEC_CHAIN_AUDIT"):
        monkeypatch.delenv(n, raising=False)
    got = _flag_off_scenario(monkeypatch)
    assert hashlib.sha256(repr(got).encode()).hexdigest() == _GOLDEN_FLAG_OFF_DIGEST
    # a sync step on the async rig (no defer_read) has the old shape: one synchronize, results on the host
    r = _AsyncRig(monkeypatch, 3, chain=False)
    f = _Flow(r, 3, [3, 2, 1])
    r.queue = _accepts(3, 1, 1)
    res = f.step()
    assert len(r.syncs()) == 1 and res.handle is None and res.accepted_counts is not None and r.e._def_last is None
    with expect_error(ValueError, "need defer_read"):
        r.e.decode_forward(
            f.tok,
            f.pos,
            num_valid_drafts=f.nvd,
            accepted_counts=f.acc,
            page_table=f.pt,
            spec_mode="fused_sample",
            chain_tip=1,
        )
    with expect_error(ValueError, "fused_sample"):
        r.e.decode_forward(
            f.tok,
            f.pos,
            num_valid_drafts=f.nvd,
            accepted_counts=f.acc,
            page_table=f.pt,
            spec_mode="argmax_ids",
            defer_read=True,
        )


# --------------------------------------------------------------------------- #
# the vLLM adapter: capability, decode_forward arguments, read_decode_output
# --------------------------------------------------------------------------- #
def _adapter(engine_calls, caps):
    import types

    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    cls = qv.Qwen36MTPContractForCausalLM
    handle = object()
    eng = types.SimpleNamespace(
        decode_forward=lambda *a, **kw: engine_calls.append(kw)
        or types.SimpleNamespace(spec_mode="fused_sample", handle=handle)
    )
    fake = types.SimpleNamespace(
        _SPEC_K=K,
        model_capabilities=caps,
        _ensure_ready=lambda: eng,
        _table=lambda t: t,
        _check_spec_sampling=lambda ss, mode, B: ss,
    )
    return cls, fake, handle


def test_adapter_passes_defer_read_and_chain_arguments_and_gates_them_on_the_capability(monkeypatch, expect_error):
    B = 3
    calls = []
    cls, fake, handle = _adapter(calls, {"supports_async_spec_decode": True})
    tok, pos = torch.full((B, T), -1, dtype=torch.int32), torch.zeros(B, T, dtype=torch.int32)
    nv, acc, pt = (
        torch.full((B,), K, dtype=torch.int32),
        torch.tensor([-1, -1, 2], dtype=torch.int32),
        torch.zeros(B, NB, dtype=torch.int32),
    )
    cr = torch.tensor([True, True, False])
    out = cls.decode_forward(
        fake,
        tok,
        pos,
        pt,
        num_valid_drafts=nv,
        accepted_counts=acc,
        spec_mode="fused_sample",
        defer_read=True,
        chain_rows=cr,
        chain_tip=4,
    )
    assert out.handle is handle and out.accepted_token_ids is None and out.spec_mode == "fused_sample"
    assert calls[0]["defer_read"] is True and calls[0]["chain_tip"] == 4 and calls[0]["chain_rows"] is cr
    out = cls.decode_forward(
        fake,
        tok,
        pos,
        pt,
        num_valid_drafts=nv,
        accepted_counts=torch.ones(B, dtype=torch.int32),
        spec_mode="fused_sample",
        defer_read=True,
    )
    assert out.handle is handle and "chain_rows" not in calls[1]
    with expect_error(ValueError, "accepted_counts -1"):  # a chain row must carry -1
        cls.decode_forward(
            fake,
            tok,
            pos,
            pt,
            num_valid_drafts=nv,
            accepted_counts=torch.ones(B, dtype=torch.int32),
            spec_mode="fused_sample",
            defer_read=True,
            chain_rows=cr,
            chain_tip=4,
        )
    with expect_error(ValueError, "entries must lie"):  # -1 is not valid on a row that does not chain
        cls.decode_forward(
            fake,
            tok,
            pos,
            pt,
            num_valid_drafts=nv,
            accepted_counts=torch.tensor([-1, -1, -1], dtype=torch.int32),
            spec_mode="fused_sample",
            defer_read=True,
            chain_rows=cr,
            chain_tip=4,
        )
    with expect_error(ValueError, "fused_sample"):
        cls.decode_forward(
            fake,
            tok,
            pos,
            pt,
            num_valid_drafts=nv,
            accepted_counts=torch.ones(B, dtype=torch.int32),
            spec_mode="argmax_ids",
            defer_read=True,
        )
    cls, fake_off, _ = _adapter([], {"supports_async_spec_decode": False})
    with expect_error(ValueError, "supports_async_spec_decode"):
        cls.decode_forward(
            fake_off,
            tok,
            pos,
            pt,
            num_valid_drafts=nv,
            accepted_counts=torch.ones(B, dtype=torch.int32),
            spec_mode="fused_sample",
            defer_read=True,
        )


def test_adapter_read_decode_output_returns_a_fused_readout_from_the_handle(monkeypatch):
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 2, 13)
    p = _Pipe(r, B, [3, 2, 1])
    h = p.enq()
    from vllm_tt_plugin.spec_decode import FusedReadout

    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    ro = qv.Qwen36MTPContractForCausalLM.read_decode_output(object(), h)
    assert (
        isinstance(ro, FusedReadout)
        and ro.accepted_token_ids.dtype == ro.accepted_counts.dtype == ro.p0.dtype == torch.int32
    )
    assert ro.p0.tolist() == [50, 60, 70]


def test_capability_is_off_without_the_flags():
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv
    from models.demos.blackhole.qwen36.tt.tp_common import spec_chain_enabled, spec_fused_enabled

    want = bool(spec_chain_enabled() and spec_fused_enabled())
    assert qv.Qwen36MTPContractForCausalLM.model_capabilities["supports_async_spec_decode"] is want


# --------------------------------------------------------------------------- #
# finishing slots (decode_forward(finishing_slots=...)) and the batched park / unpark
# --------------------------------------------------------------------------- #
class _BatchedGdn:
    """GDN layer stand-in with the batched park API: enqueue (non-blocking copy, logged), finish (after the synchronize), load enqueue.
    The saved state of (layer, row) is a pure function of both, so two engines can be compared byte for byte."""

    def __init__(self, li, log, T):
        self.li, self.log, self.T = li, log, T

    @staticmethod
    def state(li, u, T):
        return torch.full((T * 2, 1, 2, 2), 10.0 * li + u), torch.full((1, 3 + T, 4), 100.0 * li + u)

    def spec_users_save_enqueue(self, users):
        self.log.append(("gdn_enq", self.li, tuple(users)))
        return list(users)

    def spec_users_save_finish(self, pend):
        self.log.append(("gdn_fin", self.li, tuple(pend)))
        return {u: self.state(self.li, u, self.T) for u in pend}

    def spec_users_load_enqueue(self, states):
        self.log.append(("gdn_load", self.li, tuple(sorted(states))))
        self.loaded = states


class _LegacyGdn:
    """Only the per-user blocking API (the pre-batching engine contract)."""

    def __init__(self, li, log, T):
        self.li, self.log, self.T = li, log, T

    def save_spec_user(self, u):
        self.log.append(("gdn_save1", self.li, u))
        return _BatchedGdn.state(self.li, u, self.T)

    def load_spec_users(self, states):
        self.log.append(("gdn_load1", self.li, tuple(sorted(states))))


def _gdn_layers(r, cls, n=3):
    r.e._gdn = [cls(li, r.log, T) for li in range(n)]


def _calls(r, kinds):
    return [x for x in r.log if x[0] in kinds]


def test_finishing_slot_is_freed_without_a_park_and_the_chain_goes_on(monkeypatch, expect_error):
    """Row 2 is left out of step 4 because its in-flight step ends it: decode_forward(finishing_slots=[2]) frees the row like a
    finished request. No save_spec_user / enqueue call, no synchronize, no break, the chain continues; the later ordinary release is a
    no-op; the row cannot carry a request again before a prefill."""
    B, n = 3, 9
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, n, 3)
    _gdn_layers(r, _BatchedGdn)
    p = _Pipe(r, B, (3, 2, 1))
    pend = []
    for i in range(n):
        ab = (2,) if i >= 4 else ()
        if i >= 4:
            p.live[2] = False
        h = p.enq(pend[-1] if pend else None, absent=ab, finishing=[2] if i == 4 else None)
        pend.append(h)
        if len(pend) == 2:
            p.resolve(pend.pop(0))
    while pend:
        p.resolve(pend.pop(0))
    e = r.e
    assert not _calls(r, {"gdn_enq", "gdn_fin", "gdn_save1"}), r.log
    assert e.chain_breaks == {} and e.chain_steps == n - 1 and not r.syncs()
    assert 2 not in e._parked and 2 not in e._active and e._finishing == {2} and e.finishing_freed == 1
    assert "finishing rows freed 1" in e._park_note_text()
    before = (set(e._active), dict(e._parked), set(e._finishing))
    e.release(2)  # the ordinary release of the finished request arrives later
    e.release(2)
    assert (set(e._active), dict(e._parked), set(e._finishing)) == before
    # the row cannot come back without a prefill, whichever way it is offered
    p.live[2] = True
    with expect_error(RuntimeError, "without a prefill"):
        p.enq(None, finishing=None)
    # a finishing slot that is also scheduled in the step is a caller bug
    r2 = _AsyncRig(monkeypatch, B)
    r2.queue = _accepts(B, 2, 3)
    p2 = _Pipe(r2, B, (3, 2, 1))
    with expect_error(RuntimeError, "also scheduled"):
        p2.enq(None, finishing=[1])
    with expect_error(ValueError, "out of range"):
        _Pipe(_AsyncRig(monkeypatch, B), B, (3, 2, 1)).enq(None, finishing=[7])


def test_finishing_slot_matches_a_release_in_the_sync_run(monkeypatch):
    """Streams of the other rows with a finishing slot equal the run in which the row was simply released."""
    B, n = 3, 8
    q = _accepts(B, n, 5)
    rp, p, out_p = _run_pipe(monkeypatch, B, (3, 2, 1), q, finish=(3, 1))
    rf = _AsyncRig(monkeypatch, B)
    rf.queue = list(q)
    pf = _Pipe(rf, B, (3, 2, 1))
    out_f, pend = [], []
    for i in range(n):
        if i >= 5:
            pf.live[1] = False
        h = pf.enq(pend[-1] if pend else None, absent=(1,) if i >= 5 else (), finishing=[1] if i == 5 else None)
        pend.append(h)
        if len(pend) == 2:
            h0 = pend.pop(0)
            out_f.append(pf.resolve(h0, release=(1,) if len(out_f) == 3 else ()))
    while pend:
        out_f.append(pf.resolve(pend.pop(0)))
    assert rf.e.chain_breaks == {} and 1 not in rf.e._active
    for a, b in zip(out_p, out_f):
        for u in (0, 2):
            assert torch.equal(a[0][u], b[0][u]) and int(a[1][u]) == int(b[1][u])


def _finish_remap_run(monkeypatch, B, nv, n, fin_step, fin_slots, remap, seed=5):
    """Deferred pipeline (two steps in flight). At step ``fin_step`` the requests in ``fin_slots`` (PRE-remap slots) are left out
    because their in-flight step ends them; the plugin pops them BEFORE it computes ``remap`` (so remap may move another request
    into a finishing slot number). Returns (rig, outputs per step)."""
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, n, seed)
    _gdn_layers(r, _BatchedGdn)
    p = _Pipe(r, B, nv)
    out, pend = [], []
    for i in range(n):
        if i == fin_step:
            for s_ in fin_slots:
                p.live[s_] = False
        absent = tuple(u for u in range(B) if i >= fin_step and not p.live[u]) if remap is None else ()
        h = p.enq(
            pend[-1] if pend else None,
            absent=absent,
            finishing=list(fin_slots) if i == fin_step else None,
            remap=remap if i == fin_step else None,
        )
        pend.append(h)
        if len(pend) == 2:
            out.append(p.resolve(pend.pop(0)))
    while pend:
        out.append(p.resolve(pend.pop(0)))
    return r, out


def _check_finish_remap(monkeypatch, B, nv, fin_slots, remap, fin_step=5, n=9):
    r, out = _finish_remap_run(monkeypatch, B, nv, n, fin_step, fin_slots, remap)
    rp, out_p = _finish_remap_run(
        monkeypatch, B, nv, n, fin_step, fin_slots, None
    )  # plain release: same finishing, no remap
    e = r.e
    ident = list(range(B))
    idx = ident if remap is None else list(remap)
    phys_before = list(range(B))
    fin_rows = {phys_before[s_] for s_ in fin_slots}  # engine rows, pre-remap map
    assert e._finishing == fin_rows and e.finishing_freed == len(fin_slots)
    assert not (fin_rows & set(e._active)) and not (fin_rows & set(e._parked))
    assert not _calls(r, {"gdn_enq", "gdn_fin", "gdn_save1"}), r.log  # nothing parked
    assert e.chain_breaks == {} and e.chain_steps == n - 1 and not r.syncs()
    assert set(e._active) == set(range(B)) - fin_rows  # the moved requests keep their own engine rows
    # post-remap runner slot u holds the request that sat at slot idx[u]: its stream is the plain run's stream of that request
    for j, (a, b) in enumerate(zip(out, out_p)):
        for u in range(B):
            old = idx[u] if j >= fin_step else u
            if old in fin_slots and j >= fin_step:
                continue
            assert torch.equal(a[0][u], b[0][old]) and int(a[1][u]) == int(b[1][old]), (j, u, old)


def test_finishing_slot_reused_by_the_remap_of_the_same_step(monkeypatch):
    """The server crash: B rows, the request in slot 1 finishes and the plugin's remap moves the request of slot 2 into slot 1 in
    the same step. finishing_slots=[1] names the PRE-remap slot, so engine row 1 is freed (no park) and the request moved into slot 1
    (engine row 2) keeps its row and state; no 'also scheduled' error; the chain goes on; streams equal the plain release run.
    """
    _check_finish_remap(monkeypatch, 3, (3, 2, 1), [1], [0, 2, 1])


def test_finishing_without_a_remap_and_of_the_last_slot(monkeypatch):
    _check_finish_remap(monkeypatch, 3, (3, 2, 1), [1], None)
    _check_finish_remap(monkeypatch, 3, (3, 2, 1), [2], None)
    _check_finish_remap(monkeypatch, 3, (3, 2, 1), [2], [0, 1, 2])  # identity remap


def test_two_finishing_slots_with_a_condense_remap(monkeypatch):
    """B=4, slots 1 and 2 finish, the remap condenses: slot 1 takes the request of slot 3 (reuses finishing slot number 1)."""
    _check_finish_remap(monkeypatch, 4, (3, 2, 1, 2), [1, 2], [0, 3, 1, 2])
    _check_finish_remap(monkeypatch, 4, (3, 2, 1, 2), [1, 2], [0, 3, 2, 1])


def test_finishing_row_scheduled_after_the_remap_is_still_an_error(monkeypatch, expect_error):
    """Truly inconsistent: slot 1 finishes (engine row 1) and the remap leaves that request, still active, in a scheduled slot."""
    B = 3
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, 3, 3)
    p = _Pipe(r, B, (3, 2, 1))
    with expect_error(RuntimeError, "without a prefill"):
        p.enq(None, finishing=[1], remap=[0, 2, 1])  # row 1 moved to slot 2, whose block row is live


def _park_scenario(monkeypatch, cls):
    B, n = 3, 9
    r = _AsyncRig(monkeypatch, B)
    r.queue = _accepts(B, n, 3)
    _gdn_layers(r, cls)
    p = _Pipe(r, B, (3, 2, 1))
    pend = []
    for i in range(n):
        if i > 4:
            p.live[2] = False
        h = p.enq(pend[-1] if pend else None, absent=(2,) if i == 4 else ())
        pend.append(h)
        if len(pend) == 2:
            p.resolve(pend.pop(0))
    while pend:
        p.resolve(pend.pop(0))
    return r


def test_batched_park_saves_the_same_bytes_with_one_synchronize(monkeypatch):
    rb = _park_scenario(monkeypatch, _BatchedGdn)
    rl = _park_scenario(monkeypatch, _LegacyGdn)
    sb, sl = rb.e._parked[2], rl.e._parked[2]
    assert len(sb) == len(sl) == 3
    for (ra, ea), (rb_, eb) in zip(sb, sl):
        assert torch.equal(ra, rb_) and torch.equal(ea, eb) and ra.dtype == rb_.dtype and ea.dtype == eb.dtype
    # call order: every layer enqueues, ONE synchronize, then every layer finishes (the other sync is the chain materialize)
    kinds = [x[0] for x in rb.log if x[0] in {"gdn_enq", "gdn_fin", "sync"}]
    i = kinds.index("gdn_enq")
    assert kinds[i : i + 3] == ["gdn_enq"] * 3 and kinds[i + 3] == "sync" and kinds[i + 4 : i + 7] == ["gdn_fin"] * 3
    assert kinds.count("sync") == 2 and not _calls(rb, {"gdn_save1"})
    assert len(_calls(rl, {"gdn_save1"})) == 3
    n, ms = rb.e.park_stats["park"]
    assert n == 1 and ms >= 0.0
    assert "park n=1" in rb.e._park_note_text() and "deferred breaks 1 (park=1)" in rb.e._park_note_text()


def test_batched_unpark_one_synchronize_same_loaded_states(monkeypatch):
    from models.demos.blackhole.qwen36.tests.test_spec_engine_host import _bare_engine
    from models.demos.blackhole.qwen36.tt.spec_engine import parked_rows_host

    B, Kk = 3, 2
    Tt = Kk + 1
    logs = {}
    loaded = {}
    for name, cls in (("batched", _BatchedGdn), ("legacy", _LegacyGdn)):
        log = logs[name] = []
        gdn = [cls(li, log, Tt) for li in range(3)]
        for g in gdn:
            if name == "legacy":
                g.load_spec_users = lambda st, g=g: loaded.setdefault(name, []).append((g.li, st))
            else:
                g.spec_users_load_enqueue = lambda st, g=g, log=log: (
                    log.append(("gdn_load", g.li)),
                    loaded.setdefault(name, []).append((g.li, st)),
                )
        model = types_ns(B, Tt)
        e = _bare_engine(B, Kk, captured=True, nb=4, _timing=False, _gdn=gdn, model=model, mesh="mesh")
        monkeypatch.setattr(ttnn, "synchronize_device", lambda d, log=log: log.append(("sync",)))
        tok, pos, pt = (
            torch.ones(B, Tt, dtype=torch.int32),
            torch.full((B,), 5, dtype=torch.int32),
            torch.zeros(B, 4, dtype=torch.int32),
        )
        e.verify(tok, pos, torch.ones(B, dtype=torch.int32), pt, active=[True, False, True])
        assert set(e._parked) == {1}
        e.verify(tok, pos, torch.tensor([1, 3, 1], dtype=torch.int32), pt)
        assert not e._parked and e.park_stats["unpark"][0] == 1 and e.park_stats["park"][0] == 1
    assert [x for x in logs["batched"] if x[0] in {"gdn_enq", "gdn_fin", "gdn_load", "sync"}] == (
        [("gdn_enq", li, (1,)) for li in range(3)]
        + [("sync",)]
        + [("gdn_fin", li, (1,)) for li in range(3)]
        + [("gdn_load", li) for li in range(3)]
        + [("sync",)]
    )
    for (la, sa), (lb, sb) in zip(loaded["batched"], loaded["legacy"]):
        assert la == lb and list(sa) == list(sb) == [1]
        assert torch.equal(sa[1][0], sb[1][0]) and torch.equal(sa[1][1], sb[1][1])
        want = parked_rows_host(*_BatchedGdn.state(la, 1, Tt), 2, Tt)
        assert torch.equal(sa[1][0], want[0]) and torch.equal(sa[1][1], want[1])


def types_ns(B, Tt):
    import types

    return types.SimpleNamespace(
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32),
        verify_traced=lambda *a, **kw: (list(range(B * Tt)), "feed", None),
    )


def test_adapter_passes_finishing_slots_and_gates_them_on_the_capability(monkeypatch, expect_error):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    assert qv.Qwen36MTPContractForCausalLM.model_capabilities["supports_finishing_slots"] is True
    B = 3
    calls = []
    cls, fake, handle = _adapter(calls, {"supports_async_spec_decode": True, "supports_finishing_slots": True})
    tok, pos = torch.full((B, T), -1, dtype=torch.int32), torch.zeros(B, T, dtype=torch.int32)
    nv, pt = torch.full((B,), K, dtype=torch.int32), torch.zeros(B, NB, dtype=torch.int32)
    ones = torch.ones(B, dtype=torch.int32)
    cls.decode_forward(
        fake,
        tok,
        pos,
        pt,
        num_valid_drafts=nv,
        accepted_counts=ones,
        spec_mode="fused_sample",
        defer_read=True,
        finishing_slots=[2],
    )
    assert calls[0]["finishing_slots"] == [2]
    cls.decode_forward(
        fake, tok, pos, pt, num_valid_drafts=nv, accepted_counts=ones, spec_mode="fused_sample", defer_read=True
    )
    assert "finishing_slots" not in calls[1]
    cls, fake_off, _ = _adapter([], {"supports_async_spec_decode": True})
    with expect_error(ValueError, "supports_finishing_slots"):
        cls.decode_forward(
            fake_off,
            tok,
            pos,
            pt,
            num_valid_drafts=nv,
            accepted_counts=ones,
            spec_mode="fused_sample",
            defer_read=True,
            finishing_slots=[1],
        )
