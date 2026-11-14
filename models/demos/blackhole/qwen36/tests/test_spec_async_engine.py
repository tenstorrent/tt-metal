# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S4 deferred mode (decode_forward defer_read=True) on the engine, device tests; vLLM async depth 1 driver.
QWEN36_SPEC_FUSED=1 QWEN36_SPEC_CHAIN=1 (QWEN36_SPEC_CHAIN_SAMPLED unset: sampled steps are not deferred), chain audit every step.
Loop: enqueue step n+1 (chained on step n's tip) BEFORE reading step n, then read n (handle.wait()), commit, propose(hidden=n).
"""

import statistics
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.spec_helpers import PAD
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import (
    BIG,
    ChainDriver,
    _env,
    _greedy_refs,
    _marks,
    _report,
    _setup,
    _streams,
)
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all

K = tc.K
T = K + 1


class AsyncDriver(ChainDriver):
    """ChainDriver (its sync ``step`` still available for sampled sync steps) plus the deferred depth-1 pipeline."""

    def __init__(self, eng, seed, **kw):
        super().__init__(eng, seed, recycle=False, **kw)
        self.pend = []  # (handle, rows live at enqueue, request id of each)
        self.tip = None  # the newest handle, whether or not it was read
        self.chain_on_tip = (
            False  # the next step chains on ``tip`` although it was read already (prefill over a pending step)
        )
        self.host_rows = set()  # rows admitted since the last step: they carry host tokens, not placeholders
        self.next0 = {}  # row -> exact column-0 position of the oldest pending (or next) step
        self.it_ms = []

    def enq(self):
        B, eng = tc.B, self.eng
        prev = self.pend[-1][0] if self.pend else (self.tip if self.chain_on_tip else None)
        assert len(self.pend) <= 1
        tok = torch.zeros((B, T), dtype=torch.int32)
        pos = torch.full((B, T), PAD, dtype=torch.int32)
        nvd = torch.zeros(B, dtype=torch.int32)
        acc = torch.ones(B, dtype=torch.int32)
        pt = torch.zeros((B, tc.NB), dtype=torch.int32)
        live = self.live()
        kw = {}
        if prev is None:
            for r in live:
                q = self.reqs[r]
                p0 = q["P"] + len(q["out"]) - 1
                self.next0[r] = p0
                d = list(q["proposal"])
                n = len(d)
                tok[r] = PAD
                tok[r, : 1 + n] = torch.tensor([q["out"][-1]] + d, dtype=torch.int32)
                pos[r, : 1 + n] = torch.arange(p0, p0 + 1 + n, dtype=torch.int32)
                nvd[r], acc[r] = n, q["acc"]
                pt[r] = self._blocks_for(q, p0)
        else:
            cr = torch.zeros(B, dtype=torch.bool)
            for r in live:
                if r in self.host_rows:
                    q = self.reqs[r]
                    p0 = q["P"] + len(q["out"]) - 1
                    self.next0[r] = p0
                    tok[r, 0], pos[r, 0], nvd[r], acc[r] = q["out"][-1], p0, 0, q["acc"]
                    pt[r] = self._blocks_for(q, p0)
                    continue
                up = self.next0[r] + (T if self.pend else 0)  # bound of the real column-0 position of this step
                cr[r] = True
                tok[r] = PAD
                pos[r] = up + torch.arange(T, dtype=torch.int32)
                nvd[r], acc[r] = K, -1
                pt[r] = self._blocks_for(self.reqs[r], up)
            kw.update(chain_rows=cr, chain_tip=prev.tip_id)
        if self.sampled(self.steps):
            kw["spec_sampling"] = self._spec_sampling()
        res = eng.decode_forward(
            tok,
            pos,
            num_valid_drafts=nvd,
            accepted_counts=acc,
            page_table=pt,
            spec_mode="fused_sample",
            defer_read=True,
            **kw,
        )
        assert res.argmax_ids is None and res.handle is not None
        self.pend.append((res.handle, list(live), {r: self.reqs[r]["rid"] for r in live}))
        self.tip, self.chain_on_tip = res.handle, False
        self.host_rows.clear()
        self.steps += 1

    def resolve(self):
        h, rows, rids = self.pend.pop(0)
        ids, counts, p0 = h.wait()
        B = tc.B
        cpos = torch.full_like(ids, -1)
        for u in range(B):
            if int(p0[u]) >= 0:
                cpos[u, : int(counts[u])] = int(p0[u]) + 1 + torch.arange(int(counts[u]), dtype=torch.int32)
        dr = self.eng.propose_draft_tokens(K, ids.clone(), cpos, counts.clone(), hidden=h)
        ms = {}
        for r in rows:
            q = self.reqs[r]
            if (
                q is None or q["rid"] != rids[r]
            ):  # finished one step ago (its row may carry a new request): output discarded
                continue
            cnt = int(counts[r])
            assert 1 <= cnt <= T and int(p0[r]) >= 0
            assert int(ids[r, :cnt].min()) >= 0
            new = ids[r, :cnt].tolist()[: q["max_new"] - len(q["out"])]
            q["out"].extend(new)
            q["acc"] = cnt
            self.next0[r] = int(p0[r]) + cnt
            nv = int(dr.num_valid[r])
            q["proposal"] = dr.draft_token_ids[r, :nv].tolist()
            ms[r] = cnt - 1
            if len(q["out"]) >= q["max_new"]:
                self._finish(r)  # released now; the pending step n+1 still carries it, the row is absent from n+2
        self.stats.append({"live": [r for r in rows if r in ms], "n": {}, "m": ms, "cross": [], "no_draft": []})

    def drain(self):
        while self.pend:
            self.resolve()

    def pipeline(self, n):
        for _ in range(n):
            if not self.live():
                break
            t0 = time.perf_counter()
            self.enq()
            if len(self.pend) == 2:
                self.resolve()
            self.it_ms.append(1e3 * (time.perf_counter() - t0))
        self.drain()


def _sync_streams(eng, prompts, max_news, steps, seed=1):
    d = ChainDriver(eng, seed=seed, sampled=lambda s: False)
    for u, mn in enumerate(max_news):
        d.admit(u, u, prompts[u], max_new=mn)
    while d.live() and d.steps < steps:
        d.step()
    s = _streams(d)
    for r in d.live():
        d._finish(r)
    return s


def _cmp(tag, a, s):
    for u in sorted(s):
        n = min(len(a[u]), len(s[u]))
        assert n > 0 and a[u][:n] == s[u][:n], f"{tag} request {u}: async != sync within {n} tokens"
    logger.info(f"[async:{tag}] async == sync chain on common prefixes, lengths async {[len(a[u]) for u in sorted(a)]}")


def _report_a(eng, drv, tag):
    cb, fb = _report(eng, drv, tag)
    logger.info(
        f"[async:{tag}] enqueued steps={drv.steps} chain_steps={eng.chain_steps} audits={eng.chain_audits} breaks={cb}"
    )
    return cb, fb


def _time(eng, prompts):
    import os

    os.environ["QWEN36_SPEC_CHAIN_AUDIT"] = "0"
    res = {}
    try:
        d = AsyncDriver(eng, seed=41, sampled=lambda s: False)
        _admit_all(d, prompts, 8, 400)
        d.pipeline(52)
        res["async deferred chain"] = statistics.median(d.it_ms[2:52])
        for r in d.live():
            d._finish(r)
        d = ChainDriver(eng, seed=42, sampled=lambda s: False)
        _admit_all(d, prompts, 8, 400)
        for _ in range(52):
            d.step()
        res["sync chain"] = statistics.median(d.wall_ms[2:52])
        for r in d.live():
            d._finish(r)
        saved = (eng._ch_g, eng._ch_s)
        eng._ch_g = eng._ch_s = None
        try:
            d = ChainDriver(eng, seed=43, sampled=lambda s: False)
            _admit_all(d, prompts, 8, 400)
            for _ in range(52):
                d.step()
            res["sync unchained fused"] = statistics.median(d.wall_ms[2:52])
            for r in d.live():
                d._finish(r)
        finally:
            eng._ch_g, eng._ch_s = saved
            eng._chain_prev = eng._chain_in = eng._chain_cand = None
    finally:
        os.environ["QWEN36_SPEC_CHAIN_AUDIT"] = "1"
    logger.info(
        "[async:time] b8 greedy median ms/step over 50 (audit off): "
        + ", ".join(f"{k}={v:.2f}" for k, v in res.items())
    )


@_marks
def test_async_chain_greedy(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        sync = _sync_streams(eng, prompts, [BIG] * 8, 200)
        cs0, ca0 = eng.chain_steps, eng.chain_audits
        drv = AsyncDriver(eng, seed=1, sampled=lambda s: False)
        _admit_all(drv, prompts, 8, BIG)
        drv.pipeline(200)
        cb, fb = _report_a(eng, drv, "greedy")
        logger.info(
            f"[async:greedy] async-only chain_steps={eng.chain_steps - cs0} audits={eng.chain_audits - ca0} (counters include sync pass: {cs0}/{ca0})"
        )
        assert eng.chain_steps - cs0 >= 150
        streams = _streams(drv)
        for r in drv.live():
            drv._finish(r)
        _cmp("greedy", streams, sync)
        _time(eng, prompts)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    model.free_kv_caches()
    model = eng = drv = None
    _greedy_refs(mesh_device, tokenizer, prompts, streams, "async greedy")


@_marks
def test_async_chain_finishes(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        lens = [BIG] * 5 + [40, 80, 120]
        sync = _sync_streams(eng, prompts, lens, 60)
        drv = AsyncDriver(eng, seed=3, sampled=lambda s: False)
        for u in range(8):
            drv.admit(u, u, prompts[u], max_new=lens[u])
        drv.pipeline(60)
        cb, fb = _report_a(eng, drv, "finishes")
        assert {5, 6, 7} <= set(drv.outs), f"finished {sorted(drv.outs)}"
        streams = _streams(drv)
        for u in (5, 6, 7):
            assert len(streams[u]) == lens[u] and streams[u] == sync[u], f"finished request {u} differs from sync"
        _cmp("finishes", streams, sync)
        assert "row_set" not in cb
        for r in drv.live():
            drv._finish(r)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_async_no_sync(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    count = [0]
    orig = ttnn.synchronize_device

    def counted(*a, **k):
        count[0] += 1
        return orig(*a, **k)

    try:
        drv = AsyncDriver(eng, seed=5, sampled=lambda s: False)
        _admit_all(drv, prompts, 8, BIG)
        ttnn.synchronize_device = counted
        try:
            drv.pipeline(40)
        finally:
            ttnn.synchronize_device = orig
        cb, fb = _report_a(eng, drv, "no_sync")
        logger.info(
            f"[async:no_sync] ttnn.synchronize_device calls during 40 deferred steps (incl. reads and proposes): {count[0]}"
        )
        assert sum(cb.values()) <= 1 and eng.chain_steps >= 38, f"breaks {cb}"
        assert count[0] == 0, f"{count[0]} synchronize_device calls"
        for r in drv.live():
            drv._finish(r)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_async_mixed_sampled(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        mode = {"sampled": False}
        drv = AsyncDriver(eng, seed=6, sampled=lambda s: mode["sampled"])
        _admit_all(drv, prompts, 8, BIG)
        for cycle in range(6):
            mode["sampled"] = False
            drv.pipeline(10)  # drains at the end: read + propose of the pending step before a sync step
            assert not drv.pend
            mode["sampled"] = True
            for _ in range(5):
                tc_steps = drv.steps
                drv.step()
                assert drv.steps == tc_steps + 1
        cb, fb = _report_a(eng, drv, "mixed")
        assert eng.chain_steps > 0 and eng.chain_audits > 0
        for u in drv.live():
            assert len(drv.reqs[u]["out"]) > 40
        for r in drv.live():
            drv._finish(r)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_async_chain_sampled(mesh_device, monkeypatch):
    """Deferred depth-1 pipeline with every row sampled (QWEN36_SPEC_CHAIN_SAMPLED=1), audits on every step, 200 steps."""
    import os

    _env(monkeypatch)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    err = None
    try:
        drv = AsyncDriver(eng, seed=7, sampled=lambda s: True)
        _admit_all(drv, prompts, 8, BIG)
        try:
            drv.pipeline(200)
        except RuntimeError as e:
            err = e
            logger.error(f"[async:sampled] RuntimeError (overflow key risk if it mentions overflow/top-k tie): {e}")
        fb = dict(getattr(eng, "fused_fallbacks", {}))
        logger.info(
            f"[async:sampled] enqueued={drv.steps} chain_steps={eng.chain_steps} chain_breaks={dict(eng.chain_breaks)} "
            f"chain_audits={eng.chain_audits} fused_audited_steps={eng.fused_audited_steps} "
            f"fused_boundary_diffs={eng.fused_boundary_diffs} fused_fallbacks={fb} overflow_redo={fb.get('overflow_redo', 0)} "
            f"overflow_runtime_error={'YES: ' + str(err) if err is not None else 'none'}"
        )
        if err is not None:
            raise err
        for u in range(8):
            q = drv.reqs[u]
            o = q["out"] if q is not None else drv.outs[u]
            assert len(o) > 0 and min(o) >= 0 and max(o) < drv.vocab, f"row {u} token out of range"
        for r in drv.live():
            drv._finish(r)
        # timing, audits off
        os.environ["QWEN36_SPEC_CHAIN_AUDIT"] = "0"
        os.environ["QWEN36_SPEC_FUSED_AUDIT"] = "0"
        res = {}
        d = AsyncDriver(eng, seed=51, sampled=lambda s: True)
        _admit_all(d, prompts, 8, 400)
        d.pipeline(52)
        res["async deferred chained sampled"] = statistics.median(d.it_ms[2:52])
        for r in d.live():
            d._finish(r)
        d = ChainDriver(eng, seed=52, sampled=lambda s: True)
        _admit_all(d, prompts, 8, 400)
        for _ in range(52):
            d.step()
        res["sync sampled fused (default chain state)"] = statistics.median(d.wall_ms[2:52])
        for r in d.live():
            d._finish(r)
        saved = (eng._ch_g, eng._ch_s)
        eng._ch_g = eng._ch_s = None
        try:
            d = ChainDriver(eng, seed=53, sampled=lambda s: True)
            _admit_all(d, prompts, 8, 400)
            for _ in range(52):
                d.step()
            res["sync sampled fused unchained"] = statistics.median(d.wall_ms[2:52])
            for r in d.live():
                d._finish(r)
        finally:
            eng._ch_g, eng._ch_s = saved
            eng._chain_prev = eng._chain_in = eng._chain_cand = None
        logger.info(
            "[async:sampled:time] b8 median ms/step over 50 (audits off): "
            + ", ".join(f"{k}={v:.2f}" for k, v in res.items())
        )
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


def _prefill_over_pending_run(eng, prompts, overlap, steps=40):
    """Depth-1 deferred pipeline over 6 requests (row 5 ends after 30 tokens). At step 4 a new request is prefilled into the free row 6,
    and once row 5 has finished, another into that just-finished row, each while one deferred step is pending. ``overlap``: prefill, then
    read and propose the pending step (the plugin's order with QWEN36_SPEC_PREFILL_OVERLAP); else read and propose first, then
    prefill (the drained order). The step after either chains on the (already read) tip and so materializes it."""
    drv = AsyncDriver(eng, seed=11, sampled=lambda s: False)
    for u in range(6):
        drv.admit(u, u, prompts[u], max_new=30 if u == 5 else BIG)
    joined = set()

    def join(row, rid, prompt):
        if overlap:
            drv.admit(row, rid, prompt, max_new=BIG)
            drv.resolve()
        else:
            drv.resolve()
            drv.admit(row, rid, prompt, max_new=BIG)
        drv.host_rows.add(row)
        drv.chain_on_tip = True
        joined.add(rid)

    for i in range(steps):
        drv.enq()
        if len(drv.pend) == 2:
            drv.resolve()
        if len(drv.pend) == 1 and i >= 4 and 20 not in joined:
            join(6, 20, prompts[6])
        elif len(drv.pend) == 1 and drv.reqs[5] is None and 21 not in joined and 20 in joined:
            join(5, 21, prompts[5])
    drv.drain()
    streams = _streams(drv)
    for r in drv.live():
        drv._finish(r)
    assert joined == {20, 21}, f"joins done {sorted(joined)}"
    return streams


@_marks
def test_prefill_over_pending_deferred_step(mesh_device, monkeypatch):
    """The plugin may prefill while one deferred verify step is pending and finish that step afterwards (QWEN36_SPEC_PREFILL_OVERLAP).
    Greedy, all M1 flags (fused, chain, chain audit every step): every token stream of the run that prefills over the pending step
    (into a free row and into a just-finished row) equals the run that finishes the pending step before each prefill, bit for bit.
    """
    _env(monkeypatch)
    monkeypatch.setenv("QWEN36_SPEC_PREFILL_OVERLAP", "1")
    model, eng, tokenizer, prompts = _setup(mesh_device)
    # warm pass with program-cache misses allowed: the eager programs of a join into running rows and of the materialize compile here
    _prefill_over_pending_run(eng, prompts, overlap=False, steps=30)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        cb0 = dict(eng.chain_breaks)
        drained = _prefill_over_pending_run(eng, prompts, overlap=False)
        cb1 = dict(eng.chain_breaks)
        overlapped = _prefill_over_pending_run(eng, prompts, overlap=True)
        cb2 = dict(eng.chain_breaks)
        for k, name in ((cb1, "drained"), (cb2, "overlap")):
            logger.info(f"[async:prefill_overlap] {name} chain_breaks={k}")
        for cb_a, cb_b in ((cb0, cb1), (cb1, cb2)):
            assert cb_b.get("tip_not_chainable", 0) - cb_a.get("tip_not_chainable", 0) >= 2, (cb_a, cb_b)
        assert sorted(drained) == sorted(overlapped) and {20, 21} <= set(overlapped)
        for rid in sorted(drained):
            assert len(drained[rid]) > 1 and drained[rid] == overlapped[rid], f"request {rid}: overlap != drained order"
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
