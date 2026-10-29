# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the CHAINED split step (phase 3 stage P6, design note 92 section B3 T1): ``decode_forward_begin(chain_rows,
chain_tip)`` enqueues the chain trace, the audit reads and the forward-only main trace; ``decode_forward_finish`` enqueues the mask,
topk, accept, drafter and the reads. The call logs, equality with the one-call chained step, the begin / finish order rules, the reader
thread, the late counter rebased to the previous step's end, ``FusedHandle.drafts()`` and the adapter readout, the chain breaks after
a split step, overflow, the certificate row and the kill switch. Fake device (the rigs of the async and mask tests), no device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_mask_chain_engine_host.py
"""

import dataclasses
import threading
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import NB, _Pipe
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine_host import CHS
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    TOPK,
    VFY,
    K,
    T,
    _pt,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_mask_engine_host import (
    FSC,
    MASKT,
    ONES,
    _err,
    _mask,
    _MaskDefer,
    _traces,
)
from models.demos.blackhole.qwen36.tt import spec_engine as se_mod

TEMPS = [0.8, 0.0, 1.2]


class _SplitPipe(_Pipe):
    """The deferred pipe whose steps go through ``decode_forward_begin`` (the finish is the caller's)."""

    def begin(self, prev=None, sampling=None, **kw):
        e = self.r.e
        orig = e.decode_forward

        def route(*a, **k):  # the pipe calls decode_forward; begin calls it again from inside (split flag set)
            return orig(*a, **k) if e._split_begin else e.decode_forward_begin(*a, **k)

        e.decode_forward = route
        try:
            return self.enq(prev, sampling=sampling, **kw)
        finally:
            del e.decode_forward

    def finish(self, h, mask=None):
        return self.r.e.decode_forward_finish(h, mask)


def _rig(monkeypatch, B=3, n=3, seed=10, **kw):
    r = _MaskDefer(monkeypatch, B, **kw)
    r.queue = [("logits", _smooth_logits(B * T, seed + i)) for i in range(n)]
    return r


def _ss(B, seed, temps=TEMPS):
    return _sampling(B, torch.full((B,), K), temps[:B], topk=3, seed=seed)


def _kinds(log, names):
    return [x for x in log if x[0] in names]


def _strip(log):
    """The call log without the page-table staging of verify_stage_and_enqueue."""
    return [x for x in log if len(x) < 2 or x[1] not in ("kvpt", "kvpt_users", "kvpt1")]


# --------------------------------------------------------------------------- #
# a. call log order
# --------------------------------------------------------------------------- #
def test_split_chain_call_log_order(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 2)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    del r.log[:]
    h1 = p.begin(h0, sampling=_ss(B, 2))
    assert h1.pending and h1.event is None and r.ch_runs == [CHS] and r.e.chain_steps == 1
    log = _strip(r.log)
    assert [x[:2] for x in log] == [
        ("h2d", "ovr_s"),  # the overlay of the chain trace
        ("trace", CHS),
        ("from_device", "token"),  # the chain audit copies, right behind the chain trace
        ("from_device", "aux"),
        ("from_device", "fa_aux"),
        ("trace", VFY),  # the main trace only: no mask, topk, accept, drafter, no event
    ], log
    assert not _kinds(r.log, ("record_event", "sync", "event_sync")) and r.m._vfy_aux_staged is None
    assert "fs_samp" not in [x[1] for x in r.log if x[0] == "h2d"] and "token" not in [
        x[1] for x in r.log if x[0] == "h2d"
    ]
    del r.log[:]
    out = p.finish(h1, _mask(B, {(0, 0): [5, 9], (2, 2): [7, 8, 60]}))
    assert out.handle is h1 and not h1.pending and h1.post is None
    log = _strip(r.log)
    assert [x[:2] for x in log] == [
        ("h2d", "fs_samp"),
        ("h2d", "mask_in"),
        ("trace", MASKT),
        ("trace", TOPK),
        ("trace", FSC),  # the certificate accept: a masked step
        ("trace", DRAFT),
        ("from_device", "fs_result"),
        ("from_device", "fa_aux"),
        *[("from_device", f"out{k}") for k in range(K)],
        ("from_device", "ids"),  # the audit reads (QWEN36_SPEC_FUSED_AUDIT)
        ("from_device", "uvals"),
        ("from_device", "uids"),
        ("record_event",),
    ], log
    assert "sync" not in [x[0] for x in r.log]
    # the finish did not chain again: one chain trace in all, and the audit is the one the begin made
    assert r.ch_runs == [CHS] and h1.audit is not None
    assert r.e.mask_chain_steps == 1 and r.e.mask_steps == 1


# --------------------------------------------------------------------------- #
# b. split chain == one-call chain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("masked", [False, True])
def test_split_chain_equals_one_call_chain(monkeypatch, masked):
    B = 3
    mask = _mask(B, {(0, 0): [5, 9], (1, 1): range(0, 64, 2), (2, 2): [7, 8, 60]}) if masked else None
    res = []
    for split in (False, True):
        r = _rig(monkeypatch, B, 3, seed=20)
        p = _SplitPipe(r, B, [3, 2, 1])
        h0 = p.enq(None, sampling=_ss(B, 1))
        if split:
            h1 = p.begin(h0, sampling=_ss(B, 2))
            p.finish(h1, mask)
            assert h1.cmask is not None
        else:  # a one-call masked step is not chained (reason "mask"): the same stream through a materialize
            h1 = p.enq(h0, sampling={**_ss(B, 2), **({} if mask is None else {"mask": mask})})
            assert (h1.cmask is None) == masked
        h2 = p.enq(h1, sampling=_ss(B, 3))
        out = []
        for h in (h0, h1, h2):
            ids, counts, p0 = h.wait()
            out.append(
                (
                    ids,
                    counts,
                    p0,
                    h.drafts(),
                    None if h.audit is None else {k: v.data.clone() for k, v in h.audit.items()},
                )
            )
        res.append((out, r.e.mask_steps, h1.audit is not None, r.e._def_data(h1)["n_chain_audit"]))
    for a, b in zip(res[0][0], res[1][0]):
        for x, y in zip(a[:4], b[:4]):
            assert torch.equal(x, y)
    # the split step has its chain audit (made by the begin, checked at the read); the one-call masked step was not chained
    assert res[1][2] and res[1][3] == 1 and res[1][1] == (1 if masked else 0) == res[0][1]


# --------------------------------------------------------------------------- #
# c. begin before finish
# --------------------------------------------------------------------------- #
def test_begin_before_finish_raises(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 3)
    p = _SplitPipe(r, B, [3, 2, 1])
    ss = _ss(B, 2)
    h0 = p.enq(None, sampling=_ss(B, 1))
    h1 = p.begin(h0, sampling=ss)
    n_runs = len(r.log)
    # a chained begin and a plain begin on a pending split step
    _err(lambda: p.begin(h1, sampling=ss), RuntimeError, "no decode_forward_finish")
    _err(lambda: p.begin(None, sampling=ss), RuntimeError, "no decode_forward_finish")
    # the one-call decode_forward (chained or not) and prefill
    _err(lambda: p.enq(h1, sampling=ss), RuntimeError, "split step is pending")
    _err(lambda: p.enq(None, sampling=ss), RuntimeError, "split step is pending")
    _err(lambda: r.e.prefill(0, [1, 2, 3], _pt1()), RuntimeError, "before a prefill")
    # the defensive check in _def_chain_args itself (a caller that got past the other guards)
    cr = torch.ones(B, dtype=torch.bool)
    _err(lambda: r.e._def_chain_args(cr, h1.tip_id, h1), RuntimeError, "no decode_forward_finish")
    assert len(r.log) == n_runs and r.e._split_pending is h1 and not r.e._split_begin
    p.finish(h1, None)
    h2 = p.begin(h1, sampling=_ss(B, 3))  # after the finish the next begin chains on it
    p.finish(h2, None)
    assert h2.cmask is not None and r.ch_runs == [CHS, CHS]
    assert r.e._def_chain_args(cr, h2.tip_id, h2) is not None


def _pt1():
    return _pt(1)


# --------------------------------------------------------------------------- #
# d. the reader thread
# --------------------------------------------------------------------------- #
def test_reader_wait_on_pending_handle_raises(monkeypatch, expect_error):
    B = 3
    r = _rig(monkeypatch, B, 2)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    h1 = p.begin(h0, sampling=_ss(B, 2))
    for fn in (h1.wait, h1.candidates, h1.drafts):  # pending: nothing to wait on, no drafts yet
        with expect_error(RuntimeError, "not enqueued|before decode_forward_finish"):
            fn()
    # the step it chains on can be read while the successor is pending (its own event exists)
    ids0, _, _ = h0.wait()
    assert ids0.dtype == torch.int32 and ("event_sync",) in r.log
    n_sync = r.log.count(("event_sync",))
    with expect_error(RuntimeError, "not enqueued"):
        h1.wait()
    assert r.log.count(("event_sync",)) == n_sync  # the pending handle never reached the device
    p.finish(h1, None)
    with expect_error(RuntimeError, "before the handle was read"):  # finished, not read yet
        h1.drafts()
    h1.wait()
    assert h1.drafts().shape == (B, K)


def test_reader_thread_waits_after_finish(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 2, seed=30)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    h1 = p.begin(h0, sampling=_ss(B, 2))
    p.finish(h1, _mask(B, {(0, 0): [5, 9]}))
    got = {}

    def reader():  # the reader thread: FIFO, step 0 then step 1; no engine state, no device call but the event
        got["v"] = (h0.wait(), h1.wait(), h1.drafts())

    t = threading.Thread(target=reader)
    t.start()
    t.join(10)
    assert not t.is_alive() and r.threads == {t.ident} and threading.get_ident() not in r.threads
    (ids1, counts1, p01) = got["v"][1]
    assert ids1.dtype == counts1.dtype == p01.dtype == torch.int32 and int(ids1[0, 0]) in (5, 9)
    assert got["v"][2].shape == (B, K) and h1.t_done is not None and h0.t_done is not None


# --------------------------------------------------------------------------- #
# e. late counter
# --------------------------------------------------------------------------- #
def test_late_counter_chain_rebase(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 6, seed=40)
    p = _SplitPipe(r, B, [3, 2, 1])
    clock = [100.0]
    monkeypatch.setattr(se_mod.time, "perf_counter", lambda: clock[0])
    assert se_mod.MASK_LATE_MS == 31.0
    mask = _mask(B, {(0, 0): [5, 9]})
    ss = lambda i: _ss(B, i)  # noqa: E731
    prev = p.enq(None, sampling=ss(1))
    seeds = iter(range(2, 20))

    def step(prev, dt_end, dt_fin, read_prev=True):
        """begin at the current clock; the previous step ends dt_end later (its first read stamps t_done); the finish runs dt_fin after."""
        h = p.begin(prev, sampling=ss(next(seeds)))
        clock[0] += dt_end
        if read_prev:
            prev.wait()
        clock[0] += dt_fin
        p.finish(h, mask)
        return h

    n = r.e.mask_late_finish
    # 1. early mask: the previous step ended just before the finish
    h = step(prev, 0.050, 0.010)
    assert prev.t_done == 100.050 and r.e.mask_late_finish == n and r.e.mask_chain_steps == 1
    # 2. a long wait since the begin (80 ms) but the mask is early against the end of the previous step: NOT late (the rebase)
    h2 = step(h, 0.060, 0.020)
    assert r.e.mask_late_finish == n and r.e.mask_chain_steps == 2
    # 3. the mask arrives more than 31 ms after the previous step ended: late
    h3 = step(h2, 0.010, 0.035)
    assert r.e.mask_late_finish == n + 1 and r.e.mask_chain_steps == 3
    # 4. static-only mask: the previous step was not read, its end time is unknown: never late, however long it takes
    h4 = step(h3, 0.0, 5.0, read_prev=False)
    assert h3.t_done is None and r.e.mask_late_finish == n + 1 and r.e.mask_chain_steps == 4
    # 5. a begin later than the end of the previous step counts from the begin (the device idles since then)
    h4.wait()  # also reads h3 (the check of a chained step reads its predecessor)
    clock[0] += 0.5
    h5 = p.begin(h4, sampling=ss(next(seeds)))
    clock[0] += 0.040
    p.finish(h5, mask)
    assert r.e.mask_late_finish == n + 2
    # a split step that is not chained keeps the rule of P3a: finish - begin > MASK_LATE_MS
    r2 = _rig(monkeypatch, B, 2, seed=41)
    p2 = _SplitPipe(r2, B, [3, 2, 1])
    for dt, late in ((0.010, 0), (0.045, 1)):
        h = p2.begin(None, sampling=ss(1))
        clock[0] += dt
        p2.finish(h, mask)
        assert r2.e.mask_late_finish == late and r2.e.mask_chain_steps == 0
        h.wait()


# --------------------------------------------------------------------------- #
# f. drafts() and the adapter readout
# --------------------------------------------------------------------------- #
def test_handle_drafts_equal_chain_tokens(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 3, seed=50)
    p = _SplitPipe(r, B, [3, 3, 3])
    h0 = p.enq(None, sampling=_ss(B, 1))
    _err(lambda: h0.drafts(), RuntimeError, "before the handle was read")
    h1 = p.begin(h0, sampling=_ss(B, 2))
    p.finish(h1, _mask(B, {(0, 0): [5, 9]}))
    h0.wait()
    d0 = h0.drafts()
    assert d0.dtype == torch.int32 and tuple(d0.shape) == (B, K)
    assert [[int(x) for x in row] for row in d0.tolist()] == [
        list(r.e._def_data(h0)["steps"][k][u] for k in range(K)) for u in range(B)
    ]
    # what the chained step verifies as columns 1..K is exactly drafts() (every row, nv = K)
    toks = r.e._def_tokens(h1, r.e._def_data(h0))
    assert torch.equal(toks[:, 1 : 1 + K].to(torch.int32), d0)
    assert h0.drafts() is not None and h1.wait() is not None
    # a pure host view: no device call
    n = len(r.log)
    h0.drafts()
    assert len(r.log) == n
    # runner slot order: the slot permutation of the handle
    h0.perm = (torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]))
    assert torch.equal(h0.drafts(), d0[[2, 0, 1]])
    h0.perm = None
    # a step that ran on the host (a ready handle) made no drafts: zeros
    fz = types.SimpleNamespace(committed=torch.zeros(B, T), counts=torch.ones(B), candidates=None)
    ready = r.e._def_ready_handle(fz, torch.zeros(B), torch.ones(B, dtype=torch.bool))
    assert ready.drafts().shape == (B, K) and not bool(ready.drafts().any())


def test_adapter_readout_has_drafts(monkeypatch):
    from vllm_tt_plugin import spec_decode as sd

    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    B = 3
    r = _rig(monkeypatch, B, 2, seed=60)
    p = _SplitPipe(r, B, [3, 2, 1])
    h = p.enq(None, sampling=_ss(B, 1))
    # an older plugin: FusedReadout has no ``drafts`` field: the readout is built as ever
    if "drafts" not in sd.FusedReadout.__dataclass_fields__:
        ro = qv.Qwen36MTPContractForCausalLM.read_decode_output(object(), h)
        assert isinstance(ro, sd.FusedReadout) and not hasattr(ro, "drafts")

        @dataclasses.dataclass(frozen=True)
        class _RO(sd.FusedReadout):
            drafts: object = None

        monkeypatch.setattr(sd, "FusedReadout", _RO)
    ro = qv.Qwen36MTPContractForCausalLM.read_decode_output(object(), h)
    assert isinstance(ro, sd.FusedReadout) and ro.drafts.dtype == torch.int32 and tuple(ro.drafts.shape) == (B, K)
    assert torch.equal(ro.drafts, h.drafts()) and ro.p0.tolist() == [50, 60, 70]
    # the split step's readout after the finish carries its own drafts
    h1 = p.begin(h, sampling=_ss(B, 2))
    p.finish(h1, None)
    ro1 = qv.Qwen36MTPContractForCausalLM.read_decode_output(object(), h1)
    assert torch.equal(ro1.drafts, h1.drafts())


# --------------------------------------------------------------------------- #
# g. chain breaks after a split step
# --------------------------------------------------------------------------- #
def _drop_bucket_kw(r):
    """The fake model does not take the ``bucket`` kwarg the engine passes with QWEN36_SPEC_BUCKETS."""
    orig = r.m.verify_stage_and_enqueue

    def wrap(*a, **k):
        k.pop("bucket", None)
        return orig(*a, **k)

    r.m.verify_stage_and_enqueue = wrap


def _stream(h):
    ids, counts, p0 = h.wait()
    return ids.clone(), counts.clone()


def _break_case(monkeypatch, why):
    """Two flows (one-call control, split): step 0, then step 1 that breaks the chain for ``why``. Returns (control, split) engines
    and the step-1 outputs."""
    out = []
    for split in (False, True):
        if why == "mixed_fit":
            B = 2
            r = _rig(monkeypatch, B, 2, seed=70)
            cap = NB * r.e.block_size
            p = _SplitPipe(r, B, [3, 3], base=cap - 8, stride=0)
            kw = {}
        elif why == "host_row":
            B = 4
            r = _rig(monkeypatch, B, 2, seed=71)
            r.e._active = {0, 1, 2}
            p = _SplitPipe(r, B, [3, 2, -1, -1])
            kw = dict(host={2: (7, 60)})
        else:  # bucket_change: rows 2 and 3 leave, 4 -> 2
            B = 4
            r = _rig(monkeypatch, B, 2, seed=72)
            monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
            r.e._buckets, r.e._dr_traces = (1, 2, 4), {1: 31, 2: 32}
            _drop_bucket_kw(r)
            p = _SplitPipe(r, B, [3, 2, 1, 0])
            kw = {}
        mask = _mask(B, {(0, 0): [5, 9], (1, 1): range(0, 64, 2)})
        h0 = p.enq(None, sampling=_ss(B, 1, [0.8, 0.0, 1.2, 0.9]))
        if why == "bucket_change":
            p.resolve(h0, release=(2, 3))
        ss1 = _ss(B, 2, [0.8, 0.0, 1.2, 0.9])
        if split:
            h1 = p.begin(h0, sampling=ss1, **kw)
            assert h1.pending
            p.finish(h1, mask)
        else:
            h1 = p.enq(h0, sampling={**ss1, "mask": mask}, **kw)
        assert h1.cmask is None, "the chain broke: the step is host-staged"
        h0.wait()
        out.append((r, _stream(h1), h1.drafts()))
    return out


@pytest.mark.parametrize("why", ["bucket_change", "mixed_fit", "host_row"])
def test_chain_break_materializes_after_split(monkeypatch, why):
    (rc, (ids_c, cnt_c), dr_c), (rs, (ids_s, cnt_s), dr_s) = _break_case(monkeypatch, why)
    # the split step breaks for the reason ("mask" of the one-call control comes after the others only when none applies)
    assert rs.e.__dict__.get("def_breaks") == {why: 1}, rs.e.__dict__.get("def_breaks")
    assert rs.e.mask_chain_steps == 0 and rs.e.chain_steps == 0
    assert torch.equal(ids_c, ids_s) and torch.equal(cnt_c, cnt_s) and torch.equal(dr_c, dr_s)
    assert CHS not in _traces(rs)
    if why == "mixed_fit":  # the step is not fused-eligible: the host walk applies the mask
        assert rs.e.fused_fallbacks and MASKT not in _traces(rs)
    else:
        assert MASKT in _traces(rs)


# --------------------------------------------------------------------------- #
# h. overflow, certificate row, kill switch
# --------------------------------------------------------------------------- #
def test_overflow_in_split_chain_fails_closed(monkeypatch, expect_error):
    B = 3
    r = _rig(monkeypatch, B, 2, seed=80)
    lg = _smooth_logits(B * T, 3)
    lg[0, :] = 1.0  # the chained step: a flat row keeps a tie group in the union
    r.queue[1] = ("logits", lg)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    h1 = p.begin(h0, sampling=_ss(B, 2, [0.8, 1.0, 0.7]))
    p.finish(h1, _mask(B, {(2, 2): [7, 8, 60]}))
    h0.wait()
    with expect_error(RuntimeError, "tie group"):
        h1.wait()


def test_cert_row_split_stays_sync(monkeypatch):
    B = 3
    r = _rig(monkeypatch, B, 3, seed=90)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    ss = _ss(B, 2)
    ss["top_k"][0], ss["top_p"][0] = 0, 0.9  # a certificate row
    # deferred (chained or not): refused, nothing left pending
    _err(lambda: p.begin(h0, sampling=ss), ValueError, "certificate row")
    assert r.e._split_pending is None and not r.e._split_begin
    # the sync begin takes no chain: chain args need defer_read
    cr = torch.ones(B, dtype=torch.bool)
    _err(
        lambda: r.e.decode_forward_begin(
            p.tok, p.pos, num_valid_drafts=p.nvd, accepted_counts=-ONES(B), page_table=p.pt, spec_mode="fused_sample",
            spec_sampling=ss, chain_rows=cr, chain_tip=h0.tip_id,
        ), ValueError, "need defer_read",
    )  # fmt: skip
    assert r.e._split_pending is None
    # a cert row alone, sync begin without a chain: the sync finish returns the result
    r2 = _rig(monkeypatch, B, 1, seed=91)
    p2 = _SplitPipe(r2, B, [3, 2, 1])
    res = r2.e.decode_forward_begin(
        p2.tok,
        p2.pos,
        num_valid_drafts=p2.nvd,
        accepted_counts=ONES(B),
        page_table=p2.pt,
        spec_mode="fused_sample",
        spec_sampling=ss,
    )
    out = r2.e.decode_forward_finish(res.handle, None)
    assert out.accepted_token_ids is not None and out.handle is None and ("sync",) in r2.log


def test_kill_switch_off_chained_begin_raises(monkeypatch):
    B = 3
    r = _rig(
        monkeypatch, B, 3, seed=95, mask=False
    )  # no mask traces (QWEN36_SPEC_DEVICE_MASK=0 leaves the same engine)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_ss(B, 1))
    assert not r.e._mask_ok()
    n = len(r.log)
    _err(lambda: p.begin(h0, sampling=_ss(B, 2)), ValueError, "mask traces")
    assert len(r.log) == n and r.e._split_pending is None and not r.e._split_begin
    # a chain_rows tensor with no True row is no chain: the begin works as ever (and so does a begin without chain args)
    h1 = _plain_begin(r, p, _ss(B, 3), torch.zeros(B, dtype=torch.bool))
    assert h1.pending
    r.e.decode_forward_finish(h1, None)
    monkeypatch.setenv("QWEN36_SPEC_DEVICE_MASK", "0")
    assert not r.e._mask_ok()


def _plain_begin(r, p, ss, chain_rows):
    res = r.e.decode_forward_begin(
        p.tok, p.pos, num_valid_drafts=p.nvd, accepted_counts=ONES(p.B), page_table=p.pt, spec_mode="fused_sample",
        spec_sampling=ss, defer_read=True, chain_rows=chain_rows,
    )  # fmt: skip
    return res.handle


# --------------------------------------------------------------------------- #
# P7. the model-side log line of the mask counters
# --------------------------------------------------------------------------- #
def test_mask_counter_log_line(monkeypatch):
    """The first mask step and every 256th log "[spec_engine] device masks: ..."; the other steps stay silent."""
    from loguru import logger

    B = 3
    r = _rig(monkeypatch, B, 3)
    p = _SplitPipe(r, B, [3, 2, 1])
    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    try:
        h = p.enq(None, sampling=_ss(B, 1))
        h = p.begin(h, sampling=_ss(B, 2))
        p.finish(h, _mask(B, {(0, 0): [5, 9]}))
        mine = [x for x in lines if x.startswith("[spec_engine] device masks:")]
        assert len(mine) == 1 and "1 steps, 1 rows, " in mine[0] and mine[0].endswith("fallback rows"), mine
        r.e.mask_steps = 255  # the next mask step is the 256th
        h = p.begin(h, sampling=_ss(B, 3))
        p.finish(h, _mask(B, {(0, 0): [5, 9]}))
        mine = [x for x in lines if x.startswith("[spec_engine] device masks:")]
        assert len(mine) == 2 and mine[1].startswith("[spec_engine] device masks: 256 steps"), mine
    finally:
        logger.remove(sink)
