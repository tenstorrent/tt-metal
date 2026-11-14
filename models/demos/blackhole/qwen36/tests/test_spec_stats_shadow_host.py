# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the top-p certificate failure-rate probe (phase 2 step S8, design note 67 section 5.6): no device.

Part A, pure math and counters (tt/spec_shadow.py): the candidate mass fraction q against a float64 full-vocabulary reference (peaked and
flat rows), the histogram / fail counters, the examined-position mask, the temperature split, the log line and its round trip through
tests/spec_shadow_parse.py.
Part B, engine on the fake device (the rigs of test_spec_logprobs_engine_host.py): QWEN36_SPEC_STATS_SHADOW_EVERY off = no stats trace
and no extra read (the call log of a step is the same as a not-due probe step minus the probe reads); on = every N-th eligible sampled
step reads, the others do not; logprobs and certificate steps are not eligible; sync and deferred paths; an error turns the probe off.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_stats_shadow_host.py
"""

import importlib.util
import math
import os

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _Pipe
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    FS,
    S_M,
    TOPK,
    K,
    T,
    _draft_block,
    _pt,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_logprobs_engine_host import STATS, _DeferRig, _ss, _SyncRig
from models.demos.blackhole.qwen36.tt import spec_shadow as SS
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import CERT_EPS
from models.demos.blackhole.qwen36.tt.spec_shadow import EDGES, P_LIST, StatsShadow, candidate_mass_fraction
from models.demos.blackhole.qwen36.tt.tp_common import spec_stats_shadow_every

ONES = lambda B: torch.ones(B, dtype=torch.int32)  # noqa: E731


def _load_parser():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_shadow_parse.py")
    spec = importlib.util.spec_from_file_location("shadow_parse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Part A: math and counters
# --------------------------------------------------------------------------- #
def _union_of(logits, shards=8, M=64):
    """Union of the per-shard top-M (vals bf16 [R, shards*M]) of full logits [R, V], the stats (L_raw, L_T) in float64 -> fp32."""
    R, V = logits.shape
    per = V // shards
    vals = []
    for s in range(shards):
        vals.append(torch.topk(logits[:, s * per : (s + 1) * per], M, dim=1).values)
    return torch.cat(vals, dim=1)


def _ref_q(logits, temp, M=64):
    """float64 q = (mass of the global top-M) / (mass of the whole vocabulary) at the temperature."""
    x = logits.to(torch.float64) / temp
    p = torch.softmax(x, dim=1)
    return torch.topk(p, M, dim=1).values.sum(dim=1)


def _rows(kind, R=8, V=8 * 1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "peaked":  # a few high logits over a gaussian floor: q near 1 at low T
        x = torch.randn(R, V, generator=g) * 2.0
        x[:, :6] += 9.0
    elif kind == "flat":  # nearly uniform: q about M / V
        x = torch.randn(R, V, generator=g) * 0.3
    else:  # moderate (text-like): q in between
        x = torch.randn(R, V, generator=g) * 3.0
    return x.to(torch.bfloat16).float()


@pytest.mark.parametrize("kind", ["peaked", "flat", "mid"])
@pytest.mark.parametrize("temp", [0.6, 1.0, 1.3])
def test_q_equals_the_float64_full_vocabulary_reference(kind, temp):
    x = _rows(kind)
    union = _union_of(x)
    xm = x.max(dim=1, keepdim=True).values.to(torch.float64)
    l_raw = (x.to(torch.float64) - xm).exp().sum(1)
    l_t = ((x.to(torch.float64) - xm) / temp).exp().sum(1)
    t = torch.full((x.shape[0],), temp)
    qt = candidate_mass_fraction(union, l_t.to(torch.float32), t, 64)
    q1 = candidate_mass_fraction(union, l_raw.to(torch.float32), torch.ones_like(t), 64)
    assert qt.dtype == torch.float32
    assert torch.allclose(qt.double(), _ref_q(x, temp), atol=1e-5, rtol=0), (qt, _ref_q(x, temp))
    assert torch.allclose(q1.double(), _ref_q(x, 1.0), atol=1e-5, rtol=0)
    assert bool(((qt > 0) & (qt <= 1)).all())


def test_q_uses_the_row_temperature_per_row_and_is_finite_for_a_stale_stats_buffer():
    x = _rows("mid", R=4, seed=3)
    union = _union_of(x)
    temps = torch.tensor([0.6, 0.8, 1.0, 1.4])
    xm = x.max(dim=1, keepdim=True).values.to(torch.float64)
    l_t = ((x.to(torch.float64) - xm) / temps[:, None].double()).exp().sum(1)
    q = candidate_mass_fraction(union, l_t.to(torch.float32), temps, 64)
    for i in range(4):
        assert abs(float(q[i]) - float(_ref_q(x[i : i + 1], float(temps[i]))[0])) < 1e-5
    q0 = candidate_mass_fraction(union, torch.zeros(4), temps, 64)  # zero stats: q = 1 (max(L, zc) = zc), finite
    assert bool((q0 == 1.0).all())


def _stats_of(x, temp):
    xm = x.max(dim=1, keepdim=True).values.to(torch.float64)
    st = torch.zeros(64)
    R = x.shape[0]
    st[:R] = (x.to(torch.float64) - xm).exp().sum(1).float()
    st[32 : 32 + R] = ((x.to(torch.float64) - xm) / temp).exp().sum(1).float()
    return st


def test_counters_histogram_fail_counts_and_examined_mask():
    B, Tn = 4, 4
    x = _rows("mid", R=B * Tn, seed=5)
    union = _union_of(x)
    temp = torch.tensor([0.7, 1.0, 1.0, 0.7])
    st = torch.zeros(64)
    xm = x.max(dim=1, keepdim=True).values.to(torch.float64)
    st[: B * Tn] = (x.to(torch.float64) - xm).exp().sum(1).float()
    t_row = temp.repeat_interleave(Tn)[:, None].double()
    st[32 : 32 + B * Tn] = ((x.to(torch.float64) - xm) / t_row).exp().sum(1).float()
    counts = torch.tensor([2, 4, 1, 3])
    live = torch.tensor([True, True, False, True])  # row 2 is not live: not read at all
    sh = StatsShadow(8, pid=7)
    sh.record(union, st, counts, live, temp, B, Tn, 64)
    assert (sh.steps, sh.rows, sh.pos, sh.allpos) == (1, 3, 2 + 4 + 3, 3 * Tn)
    qT = torch.stack([_ref_q(x[u * Tn : (u + 1) * Tn], float(temp[u])) for u in range(B)])
    q1 = torch.stack([_ref_q(x[u * Tn : (u + 1) * Tn], 1.0) for u in range(B)])
    ex = (torch.arange(Tn)[None, :] < counts[:, None]) & live[:, None]
    for name, q, fp, fr, fs_ in (
        ("T", qT, sh.failpos["T"], sh.failrow["T"], sh.failstep["T"]),
        ("1", q1, sh.failpos["1"], sh.failrow["1"], sh.failstep["1"]),
    ):
        for i, p in enumerate(P_LIST):
            margin = (q - (p + CERT_EPS)).abs()[ex].min()
            assert margin > 1e-4, "pick another seed: a q too close to the threshold"
            f = (q < p + CERT_EPS) & ex
            assert fp[i] == int(f.sum()) and fr[i] == int(f.any(dim=1).sum()) and fs_[i] == int(f.any())
        h = torch.bucketize(q[ex].float(), torch.tensor(EDGES), right=True) - 1
        assert sh.hist[name] == torch.bincount(h.clamp(0), minlength=len(EDGES)).tolist()
        assert sum(sh.hist[name]) == int(ex.sum())
    # every live position counts for allfail, examined or not
    for i, p in enumerate(P_LIST):
        assert sh.allfail["T"][i] == int(((qT < p + CERT_EPS) & live[:, None]).sum())
    # the per-temperature split: rows 0 and 3 at 0.7, row 1 at 1.0
    assert set(sh.bytemp) == {"0.7", "1.0"} and sh.bytemp["0.7"][0] == 2 + 3 and sh.bytemp["1.0"][0] == 4
    assert sh.bytemp["0.7"][3] == 2 and sh.bytemp["1.0"][3] == 1


def test_a_step_without_a_live_row_changes_nothing():
    sh = StatsShadow(2)
    x = _rows("mid", R=4, seed=1)
    sh.record(_union_of(x), _stats_of(x, 1.0), torch.tensor([1]), torch.tensor([False]), torch.tensor([1.0]), 1, 4, 64)
    assert sh.steps == 0 and sh.pos == 0


def test_t_buckets_are_capped():
    sh = StatsShadow(1)
    B, Tn = 1, 4
    x = _rows("mid", R=4, seed=2)
    for i in range(SS.MAX_T_BUCKETS + 5):
        sh.record(
            _union_of(x),
            _stats_of(x, 1.0),
            torch.tensor([1]),
            torch.tensor([True]),
            torch.tensor([0.5 + 0.1 * i]),
            B,
            Tn,
            64,
        )
    assert len(sh.bytemp) == SS.MAX_T_BUCKETS + 1 and "other" in sh.bytemp


def test_log_line_round_trips_through_the_parser_and_the_last_line_per_pid_wins():
    sp = _load_parser()
    assert sp is not None, "tests/spec_shadow_parse.py not found"
    B, Tn = 4, 4
    x = _rows("mid", R=B * Tn, seed=9)
    st = _stats_of(x, 0.8)
    a, b = StatsShadow(8, pid=111), StatsShadow(8, pid=222)
    temp = torch.full((B,), 0.8)
    for _ in range(3):
        a.record(_union_of(x), st, torch.tensor([1, 2, 3, 4]), torch.ones(B, dtype=torch.bool), temp, B, Tn, 64)
    b.record(_union_of(x), st, torch.tensor([4, 4, 4, 4]), torch.ones(B, dtype=torch.bool), temp, B, Tn, 64)
    first = a.line()
    a.record(_union_of(x), st, torch.tensor([1, 1, 1, 1]), torch.ones(B, dtype=torch.bool), temp, B, Tn, 64)
    lines = [
        "2026-10-28 12:00:00 | INFO | spec_engine:_x:1 - " + first,
        "noise",
        "x " + b.line(),
        "y " + a.line(),
        "z shadow no",
    ]
    d = sp.parse_line(lines[0])
    assert d["pid"] == 111 and d["steps"] == 3 and d["pos"] == 3 * 10 and d["rows"] == 12 and d["M"] == 64
    assert d["ps"] == list(P_LIST) and d["edges"] == list(EDGES) and d["eps"] == CERT_EPS
    assert d["hist_qT"] == a_first_hist(first) and len(d["failpos_qT"]) == len(P_LIST)
    assert d["T0.8_pos"] == 30 and d["T0.8_failpos_qT"] == [int(v) for v in d["failpos_qT"]]
    assert abs(d["qT_mean"] - (sum_q(x, 0.8, [1, 2, 3, 4], Tn) / 10)) < 1e-5
    last = sp.last_per_pid(lines)
    assert set(last) == {111, 222} and last[111]["steps"] == 4 and last[222]["steps"] == 1
    tot = sp.combine(last)
    assert tot["steps"] == 5 and tot["pos"] == a.pos + b.pos
    assert tot["failpos_qT"] == [x_ + y for x_, y in zip(a.failpos["T"], b.failpos["T"])]
    out = sp.report(tot, 2)
    assert "probe_steps=5" in out and "tok/s" in out and "T=0.8" in out
    assert sp.parse_line("INFO nothing here") is None and sp.parse_line("SPEC_SHADOW x=1") is None
    assert math.isclose(sp.tokps(0.0), 2400.0 / 48.0)


def a_first_hist(line):
    return [int(v) for v in line.split("hist_qT=")[1].split()[0].split("/")]


def sum_q(x, temp, counts, Tn):
    return float(sum(_ref_q(x[u * Tn : u * Tn + c], temp).sum() for u, c in enumerate(counts)))


def test_tp_common_switch_parses_the_env(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_STATS_SHADOW_EVERY", raising=False)
    assert spec_stats_shadow_every() == 0
    for v, want in (("8", 8), ("0", 0), ("-3", 0), ("x", 0), (" 5 ", 5)):
        monkeypatch.setenv("QWEN36_SPEC_STATS_SHADOW_EVERY", v)
        assert spec_stats_shadow_every() == want


# --------------------------------------------------------------------------- #
# Part B: the engine (sync and deferred)
# --------------------------------------------------------------------------- #
def _stats_reads(r):
    return [x for x in r.log if x[0] == "from_device" and x[1] == "stats"]


def _sync_steps(monkeypatch, every, n, temps=(0.8, 0.0, 1.2), ss_fn=None, seed=51):
    B = 3
    r = _SyncRig(monkeypatch, B)
    r.e._shadow_every = every
    logits = _smooth_logits(B * T, seed)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    out = []
    for i in range(n):
        ss = (ss_fn or (lambda i: _sampling(B, nvd, list(temps), seed=i)))(i)
        res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
        out.append((list(r.log), res, ss))
    return r, out


def test_off_runs_no_stats_trace_reads_nothing_and_counts_nothing(monkeypatch):
    r, out = _sync_steps(monkeypatch, 0, 4)
    for log, res, _ in out:
        assert ("trace", STATS) not in log and "stats" not in [x[1] for x in log if x[0] == "from_device"]
        assert ("trace", FS) in log
    assert r.e._shadow is None and r.e._shadow_seen == 0


def test_engine_default_is_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_STATS_SHADOW_EVERY", raising=False)
    r = _SyncRig(monkeypatch, 3)
    assert r.e._shadow_every == 0 and r.e._shadow is None


def test_on_every_nth_sampled_step_reads_the_others_do_not(monkeypatch):
    r, out = _sync_steps(monkeypatch, 3, 7)
    stats_steps = [i + 1 for i, (log, _, _) in enumerate(out) if ("trace", STATS) in log]
    assert stats_steps == [3, 6]
    for i, (log, res, _) in enumerate(out):
        due = (i + 1) % 3 == 0
        assert (("trace", STATS) in log) == due
        reads = [x[1] for x in log if x[0] == "from_device"]
        assert ("stats" in reads) == due and ("uvals" in reads) == due and res.candidates is None
        assert [x[0] for x in log].count("sync") == 1
        if due:  # the stats replay sits where a logprobs step puts it
            tr = [x for x in log if x[0] == "trace"]
            assert tr.index(("trace", TOPK)) < tr.index(("trace", STATS)) < tr.index(("trace", FS))
    assert r.e._shadow.steps == 2 and r.e._shadow_seen == 7


def test_a_not_due_probe_step_has_the_call_log_of_a_step_with_the_probe_off(monkeypatch):
    _, off = _sync_steps(monkeypatch, 0, 2)
    _, on = _sync_steps(monkeypatch, 5, 2)
    assert off[0][0] == on[0][0] and off[1][0] == on[1][0]  # same traces, same reads, same order
    for a, b in zip(off, on):
        assert torch.equal(a[1].accepted_token_ids, b[1].accepted_token_ids)
        assert torch.equal(a[1].accepted_counts, b[1].accepted_counts)


def test_a_probe_step_does_not_change_the_result(monkeypatch):
    _, off = _sync_steps(monkeypatch, 0, 3)
    _, on = _sync_steps(monkeypatch, 1, 3)
    for a, b in zip(off, on):
        assert torch.equal(a[1].accepted_token_ids, b[1].accepted_token_ids)
        assert torch.equal(a[1].accepted_counts, b[1].accepted_counts)


def test_logprobs_steps_are_not_counted_and_do_not_move_the_phase(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    r.e._shadow_every = 2
    logits = _smooth_logits(B * T, 52)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    for i in range(3):  # a logprobs step: the stats run for the payload, not for the probe
        r.step(tok, pos, nvd, ONES(B), _pt(B), _ss(B, nvd, [0.8, 0.0, 1.2], [5, -2, -2], seed=i))
    assert r.e._shadow_seen == 0 and r.e._shadow is None
    r.step(tok, pos, nvd, ONES(B), _pt(B), _sampling(B, nvd, [0.8, 0.0, 1.2]))
    assert r.e._shadow_seen == 1 and ("trace", STATS) not in r.log
    r.step(tok, pos, nvd, ONES(B), _pt(B), _sampling(B, nvd, [0.8, 0.0, 1.2]))
    assert r.e._shadow_seen == 2 and ("trace", STATS) in r.log and r.e._shadow.steps == 1


def test_certificate_steps_and_all_greedy_steps_are_not_eligible(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    r.e._shadow_every = 1
    nvd = torch.full((B,), K)
    cert = _sampling(B, nvd, [0.8, 0.9, 1.0], topp=0.9)
    cert["top_k"] = torch.zeros(B, dtype=torch.int32)  # top_k 0 and top_p < 1: certificate rows
    assert r.e._shadow_due(cert) is False and r.e._shadow_seen == 0
    plain = _sampling(B, nvd, [0.8, 0.9, 1.0])
    assert r.e._shadow_due(plain) is True and r.e._shadow_seen == 1
    r.e.model._vfy_stats_ready = False  # no stats trace: never due, nothing counted
    assert r.e._shadow_due(plain) is False and r.e._shadow_seen == 1
    r.e.model._vfy_stats_ready = True
    r.e._shadow_every = 0
    assert r.e._shadow_due(plain) is False and r.e._shadow_seen == 1


def test_counters_of_a_sync_probe_step_match_the_reference_on_the_examined_positions(monkeypatch):
    B = 3
    r, out = _sync_steps(monkeypatch, 1, 1, temps=(0.8, 0.0, 1.2), seed=53)
    res = out[0][1]
    counts = res.accepted_counts.to(torch.int64)
    sh = r.e._shadow
    want = int(counts[0] + counts[2])  # row 1 is greedy: not read
    assert (sh.steps, sh.rows, sh.pos, sh.allpos) == (1, 2, want, 2 * T) and sum(sh.hist["T"]) == want
    x = r.logits.float()
    # every fail count equals the float64 reference (the margins are checked: else the seed must change)
    qs = {u: _ref_q(x[u * T : (u + 1) * T], tmp, S_M) for u, tmp in ((0, 0.8), (2, 1.2))}
    for i, p in enumerate(P_LIST):
        for u in (0, 2):
            assert (qs[u] - (p + CERT_EPS)).abs().min() > 1e-4
        f = sum(int((qs[u][: int(counts[u])] < p + CERT_EPS).sum()) for u in (0, 2))
        assert sh.failpos["T"][i] == f, (i, f, sh.failpos["T"])
    assert set(sh.bytemp) == {"0.8", "1.2"}


def test_a_probe_error_turns_the_probe_off_and_keeps_the_step(monkeypatch):
    B = 3
    r = _SyncRig(monkeypatch, B)
    r.e._shadow_every = 1
    logits = _smooth_logits(B * T, 54)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2])
    ref = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)

    class Boom(StatsShadow):
        def record(self, *a, **k):
            raise RuntimeError("boom")

    r.e._shadow = Boom(1)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e._shadow_every == 0 and torch.equal(res.accepted_token_ids, ref.accepted_token_ids)
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert ("trace", STATS) not in r.log


def test_shutdown_logs_the_final_counters(monkeypatch):
    r, _ = _sync_steps(monkeypatch, 1, 2)
    seen = []
    import models.demos.blackhole.qwen36.tt.spec_engine as SE

    monkeypatch.setattr(SE.logger, "info", lambda m, *a, **k: seen.append(str(m)))
    try:
        r.e.shutdown()
    except Exception:  # noqa: BLE001  the fake model cannot release traces; the line is logged first
        pass
    assert any(m.startswith("SPEC_SHADOW pid=") and " steps=2 " in m for m in seen)


def _defer_probe(monkeypatch, every, n, temps=(0.8, 0.0, 1.2), nlp=None):
    B = 3
    r = _DeferRig(monkeypatch, B)
    r.e._shadow_every = every
    L = [_smooth_logits(B * T, 60 + i) for i in range(n)]
    r.queue = [("logits", x) for x in L]
    p = _Pipe(r, B, [3, 2, 1])
    hs, logs = [], []
    for i in range(n):
        ss = _sampling(B, torch.full((B,), K), list(temps), seed=i)
        if nlp is not None:
            ss["num_logprobs"] = torch.tensor(nlp, dtype=torch.int32)
        hs.append(p.enq(hs[-1] if hs else None, sampling=ss))
        logs.append(list(r.log))
        del r.log[:]
    return r, p, hs, logs


def test_deferred_probe_step_enqueues_the_reads_without_a_sync_and_the_reader_counts(monkeypatch):
    r, p, hs, logs = _defer_probe(monkeypatch, 2, 4)
    for i, log in enumerate(logs):
        due = (i + 1) % 2 == 0
        kinds = [x[0] for x in log]
        assert "sync" not in kinds and "to_torch" not in kinds, "no host conversion on the engine thread"
        assert (("trace", STATS) in log) == due
        reads = [x[1] for x in log if x[0] == "from_device"]
        assert ("stats" in reads) == due and reads.count(
            "uvals"
        ) == 1  # the S2 audit reads the union on every step (read once)
        if due:
            assert max(j for j, x in enumerate(log) if x[0] == "from_device") < kinds.index("record_event")
        assert (hs[i].shadow_t is not None) == due and hs[i].cand_t is None
    assert r.e._shadow is None, "nothing is counted before a reader waits"
    for h in hs:
        p.resolve(h)
        assert h.candidates() is None
    assert r.e._shadow.steps == 2 and r.e._shadow_seen == 4
    n_pos = sum(int(hs[i].wait()[1][u]) for i in (1, 3) for u in (0, 2))
    assert r.e._shadow.pos == n_pos and r.e._shadow.rows == 4


def test_deferred_probe_waits_once_per_handle(monkeypatch):
    r, p, hs, _ = _defer_probe(monkeypatch, 1, 1)
    hs[0].wait()
    hs[0].wait()
    assert r.e._shadow.steps == 1


def test_deferred_off_is_the_old_enqueue(monkeypatch):
    r, p, hs, logs = _defer_probe(monkeypatch, 0, 3)
    for log in logs:
        assert ("trace", STATS) not in log and "stats" not in [x[1] for x in log if x[0] == "from_device"]
    assert all(h.shadow_t is None for h in hs)
    for h in hs:
        p.resolve(h)
    assert r.e._shadow is None and r.e._shadow_seen == 0


def test_deferred_logprobs_steps_are_not_counted(monkeypatch):
    r, p, hs, logs = _defer_probe(monkeypatch, 1, 2, nlp=[5, -2, -2])
    assert r.e._shadow_seen == 0 and all(h.shadow_t is None and h.cand_t is not None for h in hs)
    for h in hs:
        p.resolve(h)
    assert r.e._shadow is None
