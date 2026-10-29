# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the top-p certificate (phase 2 step S3, work/research_1026/67 sections 5 and 8; tests H4, H5, H6 and the engine part of
H12): no device.

Part A, pure math (spec_fused_sampled): the certificate rule against a brute-force float64 full-vocabulary nucleus (H4), the
reference with ``full_norm`` against host_redo_walk (H5), the per-user subset redo (H6), the fast row distribution with and without
its path gate.
Part B, engine on the fake device (the rigs of test_spec_fused_engine_host.py and test_spec_logprobs_engine_host.py): selection of the
plain / certificate accept trace, the sync rule, the failure -> subset redo, the overflow word, ``_sampled_ineligible``, the audit, the
capture / prepare calls, the adapter gate.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_topp_cert_host.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    DRAFT,
    S_M,
    S_V,
    TOPK,
    K,
    T,
    _draft_block,
    _plateau_logits,
    _pt,
    _SampledRig,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_logprobs_engine_host import STATS, _StatsMixin
from models.demos.blackhole.qwen36.tt import spec_engine as SE
from models.demos.blackhole.qwen36.tt import spec_fused_sampled as SF
from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    CERT_EPS,
    OVF_CERT,
    OVF_TIE,
    STATS_LT_ROW,
    _row_distribution,
    _row_distribution_full,
    certificate_cut,
    fused_sampled_reference,
    host_redo_walk,
    merge_redo,
    predraw_uniforms,
)
from models.demos.blackhole.qwen36.tt.spec_logprobs import stats_from_logits

# --------------------------------------------------------------------------- #
# Part A: the math
# --------------------------------------------------------------------------- #
SH, PER, M = 4, 1024, 64  # shards, per-shard width, candidates
V = SH * PER
BA, KA = 2, 3  # users, drafts of the math tests
TA = KA + 1
SCRATCH, NB, CAP = 5000, 6, 4096


def _bf(x):
    return x.to(torch.bfloat16).float()


def _rows(g, n, kind):
    """n bf16 logit rows [n, V] of one kind: peaked (a few strong tokens), flat (gaussian), zipf (heavy tail), ties (plateaus of equal
    values, one crossing the 63 / 64 border)."""
    if kind == "peaked":
        x = torch.randn(n, V, generator=g) * 1.2
        for r in range(n):
            x[r, torch.randperm(V, generator=g)[:4]] += torch.tensor([10.0, 9.0, 8.0, 7.0])
    elif kind == "flat":
        x = torch.randn(n, V, generator=g) * (0.5 + 1.5 * torch.rand(n, 1, generator=g))
    elif kind == "zipf":
        c = 0.9 + 1.2 * torch.rand(n, 1, generator=g)
        x = torch.stack([-torch.log1p(torch.randperm(V, generator=g).float()) for _ in range(n)]) * c
    else:  # ties
        x = torch.randn(n, V, generator=g).round() * 0.5  # a handful of distinct values: huge tie groups
        x[:, :3] += 6.0
    return _bf(x)


def _union(x):
    """The topk trace of rows x [R, V]: per-shard top-M, (value desc, local id asc). vals bf16 [R, SH*M], local ids int64."""
    vs, ls = [], []
    for s in range(SH):
        sh = x[:, s * PER : (s + 1) * PER]
        o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :M]
        ls.append(o)
        vs.append(sh.gather(1, o))
    return torch.cat(vs, 1).to(torch.bfloat16), torch.cat(ls, 1)


def _reference(x, temp, top_k, top_p, greedy, nv, uni, tok, with_norm=True):
    R = x.shape[0]
    B = R // TA
    uv, ul = _union(x)
    aux = fused_accept_aux_host(
        nv,
        torch.full((B,), 7),
        torch.ones(B, dtype=torch.bool),
        torch.zeros(B, dtype=torch.bool),
        CAP,
        KA,
        SCRATCH,
        0,
        4096,
    )
    pt = torch.arange(1, 1 + B * NB, dtype=torch.int32).reshape(B, NB)
    trow = torch.where(greedy | (temp <= 0), torch.ones(B), temp).repeat_interleave(TA)
    l_t = stats_from_logits(x, trow)[2].to(torch.float32) if with_norm else None
    order = torch.sort(x, dim=1, descending=True, stable=True).indices
    return fused_sampled_reference(
        uv, ul, tok.reshape(-1), order[:, 0], aux, pt, B=B, K=KA, per_shard=PER, M=M, greedy=greedy,
        temperature=temp, top_k=top_k, top_p=top_p, uniforms=uni, num_valid=nv, full_norm=l_t,
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["peaked", "flat", "zipf", "ties"])
def test_h4_certificate_rule_equals_the_float64_full_vocabulary_nucleus(kind):
    """H4: for every row of every case: pass => the kept set and the probabilities of the candidates equal the float64 full-vocabulary
    nucleus (_row_distribution_full, top_k off) within 1e-6 and nothing outside the candidates is kept; fail <=> q - p < CERT_EPS;
    the certificate never passes when the float64 q is below p + CERT_EPS (rows within 1e-4 of the margin are skipped).
    """
    g = torch.Generator().manual_seed({"peaked": 1, "flat": 2, "zipf": 3, "ties": 4}[kind])
    n_pass = n_fail = n_skip = 0
    for case in range(40):
        x = _rows(g, BA * TA, kind)
        temp = torch.rand(BA, generator=g) * 1.2 + 0.3  # 0.3 .. 1.5
        top_p = torch.tensor([0.5, 0.8, 0.9, 0.95, 0.99])[torch.randint(0, 5, (BA,), generator=g)]
        nv = torch.full((BA,), KA)
        tok = torch.zeros(BA, TA, dtype=torch.int64)
        torch.manual_seed(case)
        ref = _reference(
            x, temp, torch.zeros(BA, dtype=torch.int64), top_p, torch.zeros(BA, dtype=torch.bool), nv,
            predraw_uniforms({}, BA, KA, nv), tok,
        )  # fmt: skip
        assert ref.cert_q is not None
        for r in range(BA * TA):
            u = r // TA
            t, p = float(temp[u]), float(top_p[u])
            ids64, prob64 = _row_distribution_full(x[r], t, 0, p)
            e = torch.exp((x[r].double() - x[r].double().max()) / t)
            q64 = float(torch.topk(e, M).values.sum() / e.sum())  # ties: the sum of the top M values is order free
            q32 = float(ref.cert_q[r])
            fail = (q32 - p) < CERT_EPS
            if abs(q64 - p - CERT_EPS) < 1e-4:
                n_skip += 1  # the eps zone: float32 vs float64 may differ
                continue
            assert fail == (q64 < p + CERT_EPS), (kind, case, r, q32, q64, p)
            if fail:
                n_fail += 1
                continue
            n_pass += 1
            cand = ref.cand_ids[r]
            kept = torch.zeros(V, dtype=torch.float64)
            kept[cand] = ref.prob[r].double()
            full = torch.zeros(V, dtype=torch.float64)
            full[ids64] = prob64
            assert float((full - kept).abs().max()) < 1e-6, (kind, case, r, float((full - kept).abs().max()))
            assert bool(((full > 0) == (kept > 0)).all() or float(full[(full > 0) != (kept > 0)].max()) < 1e-6)
    print(f"\n[H4 {kind}] pass {n_pass} fail {n_fail} eps-zone skipped {n_skip}")
    assert n_pass + n_fail > 150
    if kind == "peaked":
        assert n_pass > 0.5 * (n_pass + n_fail)
    if kind == "flat":
        assert n_fail > 0.9 * (n_pass + n_fail)


def test_h4_certificate_cut_formula_and_never_a_false_pass():
    """certificate_cut: q = zc / max(L_T, zc) in (0, 1]; a stale (zero) or too small L_T gives q = 1, never above; fail iff q - p < eps."""
    zc = torch.tensor([3.0, 3.0, 3.0, 3.0])
    full = torch.tensor([3.0, 6.0, 0.0, 1.0])  # q = 1, 0.5, 1 (stale zero), 1 (L_T < zc)
    q, omp, fail = certificate_cut(zc, full, torch.tensor([0.99, 0.5, 0.99, 0.5]))
    assert torch.allclose(q, torch.tensor([1.0, 0.5, 1.0, 1.0])) and bool((q <= 1.0).all())
    assert torch.allclose(omp, 1.0 - torch.tensor([0.99, 0.5, 0.99, 0.5]) / q)
    assert fail.tolist() == [False, True, False, False]  # q - p = 0.5 - 0.5 = 0 < eps: fail
    q, _, fail = certificate_cut(torch.tensor([1.0]), torch.tensor([1.0 / 0.95]), torch.tensor([0.95]))
    assert bool(fail[0]) or float(q[0]) - 0.95 >= CERT_EPS  # q = 0.95 exactly: inside the margin, so fail
    assert (
        bool(certificate_cut(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))[2][0]) is True
    )  # p = 1 never passes


def test_h5_reference_with_full_norm_equals_the_host_redo():
    """H5: fused_sampled_reference(full_norm=) on certificate rows plus top-k rows plus greedy rows commits what host_redo_walk commits
    (float64, whole vocabulary) with the same uniforms, except the float32 boundary classes (audit_boundary_users); overflow bit 2 is
    set only for a failing certificate row, bit 1 never for it."""
    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import audit_boundary_users

    g = torch.Generator().manual_seed(77)
    n_users = n_pass = n_fail = n_diff = 0
    for case in range(80):
        B = 4
        kind = ["peaked", "zipf", "flat", "peaked"][case % 4]
        x = _rows(g, B * TA, kind)
        greedy = torch.zeros(B, dtype=torch.bool)
        greedy[case % B] = case % 3 == 0
        temp = torch.rand(B, generator=g) * 1.2 + 0.3
        top_k = torch.tensor([0, 0, 0, int(torch.randint(1, 63, (), generator=g))])
        top_p = torch.tensor([0.5, 0.8, 0.9, 0.95, 0.99])[torch.randint(0, 5, (B,), generator=g)]
        nv = torch.randint(0, KA + 1, (B,), generator=g)
        order = torch.sort(x, dim=1, descending=True, stable=True).indices
        drafts = order.reshape(B, TA, V)[:, :KA, :6].gather(2, torch.randint(0, 6, (B, KA, 1), generator=g)).squeeze(2)
        tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), drafts], dim=1)
        torch.manual_seed(case)
        uni = predraw_uniforms({}, B, KA, nv)
        ref = _reference(x, temp, top_k, top_p, greedy, nv, uni, tok)
        rc, rn = host_redo_walk(
            x, tok, {"temperature": temp, "top_k": top_k, "top_p": top_p, "greedy": greedy}, uni, B=B, K=KA, num_valid=nv
        )  # fmt: skip
        cert = ~greedy & (top_k == 0)
        assert not bool(((ref.overflow & OVF_TIE) != 0)[cert].any())
        assert bool(((ref.overflow & OVF_CERT) == 0)[~cert].all())
        bound = audit_boundary_users(ref, uni, rc, rn, tok[:, 1:], nv, B=B, K=KA, M=M, top_p=top_p)
        for u in range(B):
            n_users += 1
            if int(ref.overflow[u]) != 0:
                n_fail += 1
                continue
            n_pass += 1
            same = int(ref.counts[u]) == int(rn[u]) and torch.equal(ref.committed[u, : int(rn[u])], rc[u, : int(rn[u])])
            if not same:
                n_diff += 1
                assert u in bound, (case, u)
    print(f"\n[H5] users {n_users} pass {n_pass} flagged {n_fail} boundary-explained differences {n_diff}")
    assert n_pass > 0.4 * n_users and n_fail > 0 and n_diff <= 0.02 * n_pass


def _redo_inputs(seed, B=3):
    g = torch.Generator().manual_seed(seed)
    x = _rows(g, B * TA, "peaked")
    order = torch.sort(x, dim=1, descending=True, stable=True).indices
    tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), order.reshape(B, TA, V)[:, :KA, 0]], dim=1)
    params = {
        "temperature": torch.tensor([0.7, 1.0, 0.5][:B]),
        "top_k": torch.zeros(B, dtype=torch.int64),
        "top_p": torch.tensor([0.9, 0.95, 0.8][:B]),
        "greedy": torch.zeros(B, dtype=torch.bool),
    }
    nv = torch.full((B,), KA)
    torch.manual_seed(seed)
    return x, tok, params, predraw_uniforms({}, B, KA, nv), nv


def test_h6_subset_redo_merges_exactly():
    """H6: host_redo_walk(users=) equals the full redo for those users, a user's result does not depend on the other users (each user
    alone gives the same row), and merge_redo keeps the device result of every other user."""
    B = 3
    for seed in range(4):
        x, tok, params, uni, nv = _redo_inputs(seed, B)
        fc, fn = host_redo_walk(x, tok, params, uni, B=B, K=KA, num_valid=nv)
        sc, sn = host_redo_walk(x, tok, params, uni, B=B, K=KA, num_valid=nv, users=[1])
        assert torch.equal(sc[1], fc[1]) and int(sn[1]) == int(fn[1])
        assert bool((sc[[0, 2]] == -1).all()) and sn[[0, 2]].tolist() == [1, 1]  # the others are not walked
        for u in range(B):  # independence: user u alone (B = 1 slice of the same rows, uniforms and parameters)
            sl = slice(u * TA, (u + 1) * TA)
            ac, an = host_redo_walk(
                x[sl], tok[u : u + 1], {k: v[u : u + 1] for k, v in params.items()}, uni[u : u + 1], B=1, K=KA, num_valid=nv[u : u + 1]
            )  # fmt: skip
            assert torch.equal(ac[0], fc[u]) and int(an[0]) == int(fn[u])
        dev_c = torch.randint(0, 100, (B, TA), dtype=torch.int32)
        dev_n = torch.tensor([2, 3, 1], dtype=torch.int32)
        mc, mn = merge_redo(dev_c, dev_n, sc, sn, [1])
        assert torch.equal(mc[0], dev_c[0]) and torch.equal(mc[2], dev_c[2]) and mn[[0, 2]].tolist() == [2, 1]
        assert torch.equal(mc[1], fc[1]) and int(mn[1]) == int(fn[1])
        mc, mn = merge_redo(dev_c, dev_n, fc, fn, [])  # nothing flagged: the device result stays
        assert torch.equal(mc, dev_c) and torch.equal(mn, dev_n)


@pytest.mark.parametrize("flat", [False, True])
def test_fast_row_distribution_equals_the_full_sort_with_and_without_the_gate(monkeypatch, flat):
    """The fast row distribution (prefix top-k + full normaliser) returns the first n entries of the full sort's result, with the tail
    path gate forced both ways (it only chooses the path, never the result)."""
    g = torch.Generator().manual_seed(5 + int(flat))
    for gate in (True, False):
        monkeypatch.setattr(SF, "_tail_beyond_prefix_estimate", lambda x, t, pr, gate=gate: 9.0 if gate else 0.0)
        for case in range(12):
            x = _rows(g, 1, "flat" if flat else "peaked")[0]
            t, p = 0.3 + 1.2 * float(torch.rand((), generator=g)), [0.5, 0.9, 0.95, 0.99][case % 4]
            a = _row_distribution_full(x, t, 0, p)
            b = _row_distribution(x, t, 0, p, prefix=64)
            n = b[0].numel()
            assert torch.equal(a[0][:n], b[0]) and torch.allclose(a[1][:n], b[1], atol=1e-12, rtol=0)
            assert float(a[1][n:].sum()) == 0.0


# --------------------------------------------------------------------------- #
# Part B: the engine on the fake device
# --------------------------------------------------------------------------- #
FSC = 17  # the id of the fake certificate accept trace


class _CertMixin:
    """A rig with the certificate trace: ``e._fs_trace_c`` = FSC, ``fs.stats_in`` = the fake stats buffer; the fake FSC trace runs the
    reference with full_norm = rows 32.. of the stats buffer (what the stats trace 'computed')."""

    def _add_cert(self):
        self.e._fs_trace_c = FSC
        self.fs.stats_in = self.m._vfy_stats_out

    def _exec(self, dev, tid, cq_id=0, blocking=False):
        if tid != FSC:
            return super()._exec(dev, tid, cq_id, blocking)
        super()._exec(
            dev, tid, cq_id, blocking
        )  # the log entry only (a trace id that is not FS runs nothing in the base rig)
        fs, B = self.fs, self.B
        sp = fs.samp_in.data
        norm = self.m._vfy_stats_out.data.reshape(64)[STATS_LT_ROW : STATS_LT_ROW + self.R]
        ref = fused_sampled_reference(
            fs.union_vals_in.data.reshape(32, -1)[: self.R], fs.union_ids_in.data.reshape(32, -1)[: self.R],
            self.m._vfy_token_buf.data, self.m._vfy_ids_out.data.reshape(-1), fs.aux_in.data, fs.pt_rows_in.data[::T],
            B=B, K=K, per_shard=SF_PER, M=S_M, greedy=sp[3, :B] > 0.5, temperature=sp[0, :B], top_k=sp[1, :B].to(torch.int64),
            top_p=sp[2, :B], uniforms=sp[4 : 4 + B, : K + 2], full_norm=norm,
        )  # fmt: skip
        res = torch.zeros(3, 32, dtype=torch.int32)
        res[0, : self.R] = ref.committed.reshape(-1)
        res[1, :B], res[2, :B] = ref.counts, ref.overflow
        fs.result_out.data = res


SF_PER = S_V // 2  # the base rig's TP is 2


class _CertRig(_CertMixin, _StatsMixin, _SampledRig):
    def __init__(self, monkeypatch, B, cert=True):
        super().__init__(monkeypatch, B)
        self._add_stats()
        if cert:
            self._add_cert()


def _peaked_logits(R, seed):
    """Rows with three strong tokens and a weak plateau of distinct small values: the top-S_M candidates hold almost all the mass."""
    g = torch.Generator().manual_seed(seed)
    out = torch.stack([torch.linspace(-3.0, -3.5, S_V)[torch.randperm(S_V, generator=g)] for _ in range(R)])
    for r in range(R):
        a, b, c = torch.randperm(S_V, generator=g)[:3].tolist()
        out[r, a], out[r, b], out[r, c] = 9.0, 8.0, 7.0
    return out


def _mixed_logits(B, seed):
    """User 0 peaked (a certificate row that passes), user 1 flat (a certificate row that fails), the rest smooth (top-k rows)."""
    x = _smooth_logits(B * T, seed)
    x[0:T] = _peaked_logits(T, seed + 1)
    return x


def _cert_sampling(B, nvd, temps, cert_users, topp=0.9, seed=0):
    ss = _sampling(B, nvd, temps, topk=3, seed=seed)
    for u in cert_users:
        ss["top_k"][u] = 0
        ss["top_p"][u] = topp
    return ss


ONES = lambda B: torch.ones(B, dtype=torch.int32)  # noqa: E731


def _redo_of(logits, tok, ss, nvd, B):
    return host_redo_walk(
        logits.to(torch.bfloat16).float(), tok, ss, ss["uniforms"], B=B, K=K, num_valid=torch.as_tensor(nvd)
    )


def test_a_certificate_step_replays_the_certificate_trace_sync_and_exact(monkeypatch):
    """Selection + sync + result: a step with a certificate row replays topk, stats, FSC (not FS), the drafter, with ONE sync; a passing
    row commits what the float64 full-vocabulary host walk commits (audit on: reads the stats, no difference)."""
    B = 2
    r = _CertRig(monkeypatch, B)
    logits = _peaked_logits(B * T, 11)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    ss = _cert_sampling(B, nvd, [0.8, 0.9], [0, 1])
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    tr = [x[1] for x in r.log if x[0] == "trace"]
    assert tr.index(TOPK) < tr.index(STATS) < tr.index(FSC) < tr.index(DRAFT)
    assert 13 not in tr, "the plain accept trace must not run on a certificate step"
    assert [x[0] for x in r.log].count("sync") == 1
    assert r.e.cert_steps == 1 and r.e.cert_failed_steps == 0 and not r.e.__dict__.get("fused_fallbacks")
    assert (
        r.e.fused_boundary_diffs == 0 and ("to_torch", "stats") in r.log
    ), "the audit compares against the reference with L_T"
    rc, rn = _redo_of(logits, tok, ss, nvd, B)
    assert torch.equal(res.accepted_counts, rn)
    for u in range(B):
        n = int(rn[u])
        assert torch.equal(res.accepted_token_ids[u, :n], rc[u, :n])


def test_b_normal_steps_keep_the_plain_trace_even_with_the_certificate_trace_present(monkeypatch):
    B = 3
    r = _CertRig(monkeypatch, B)
    logits = _smooth_logits(B * T, 12)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2, 1])
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3)
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    tr = [x[1] for x in r.log if x[0] == "trace"]
    assert 13 in tr and FSC not in tr and STATS not in tr and r.e.cert_steps == 0
    # a top_k = 0 row with top_p = 1 is not a certificate row: top_k_range (the host walk), the certificate trace stays out
    ss = _sampling(B, nvd, [0.8, 0.0, 1.2], topk=3)
    ss["top_k"][0] = 0
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks == {"top_k_range": 1} and FSC not in [x[1] for x in r.log if x[0] == "trace"]


def test_c_failure_redoes_only_the_flagged_user_and_the_others_keep_the_device_result(monkeypatch):
    B = 3
    r = _CertRig(monkeypatch, B)
    logits = _mixed_logits(B, 13)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 3, 3])
    ss = _cert_sampling(B, nvd, [0.8, 1.0, 1.1], [0, 1], topp=0.9)  # user 0 passes, user 1 fails, user 2 a top-k 3 row
    calls = []
    real = SE.host_redo_walk
    monkeypatch.setattr(SE, "host_redo_walk", lambda *a, **k: calls.append(k.get("users")) or real(*a, **k))
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert calls == [[1]], "only the user whose certificate failed is redone on the host"
    assert r.e.fused_fallbacks == {"overflow_redo": 1} and r.e.cert_steps == 1 and r.e.cert_failed_steps == 1
    assert [x[0] for x in r.log].count("sync") == 2 and ("trace", 15) in r.log  # the logits trace of the redo
    assert r.e._fused_cache is None, "a redo step caches no drafts"
    # the merged result equals a full float64 redo of every user (users 0 and 2 from the device)
    rc, rn = _redo_of(logits, tok, ss, nvd, B)
    assert torch.equal(res.accepted_counts, rn)
    for u in range(B):
        n = int(rn[u])
        assert torch.equal(res.accepted_token_ids[u, :n], rc[u, :n]), u


def test_d_overflow_word_decode_tie_and_certificate_bits(monkeypatch):
    """bit 1 (top-k tie) and bit 2 (certificate) both redo the user; cert_failed_steps counts bit 2 only."""
    B = 3
    r = _CertRig(monkeypatch, B)
    logits = _plateau_logits(
        B * T, 3
    )  # a tie plateau: top_k = 3 users overflow (bit 1); plateau rows are flat: a cert user fails
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 3, 3])
    ss = _cert_sampling(B, nvd, [0.8, 1.0, 1.1], [1], topp=0.99)
    seen = []
    real = SE.host_redo_walk
    monkeypatch.setattr(SE, "host_redo_walk", lambda *a, **k: seen.append(sorted(k["users"])) or real(*a, **k))
    r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert seen == [[0, 1, 2]] and r.e.cert_failed_steps == 1
    # only tie overflows (no certificate row): bit 2 never set, cert_failed_steps stays
    r2 = _CertRig(monkeypatch, B)
    r2.set_logits(logits)
    r2.step(tok, pos, nvd, ONES(B), _pt(B), _sampling(B, nvd, [0.8, 1.0, 1.1], topk=3))
    assert r2.e.fused_fallbacks == {"overflow_redo": 1} and r2.e.cert_failed_steps == 0 and r2.e.cert_steps == 0
    assert OVF_TIE == 1 and OVF_CERT == 2 and (OVF_TIE | OVF_CERT) == 3


def test_e_deferred_steps_never_carry_a_certificate_row(monkeypatch, expect_error):
    from models.demos.blackhole.qwen36.tests.test_spec_logprobs_engine_host import _DeferRig

    B = 2
    r = _DeferRig(monkeypatch, B)
    logits = _peaked_logits(B * T, 14)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    ss = _cert_sampling(B, nvd, [0.8, 0.9], [0])
    with expect_error(ValueError, "certificate row"):
        r.e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=ONES(B), page_table=_pt(B), spec_mode="fused_sample",
                           spec_sampling=ss, defer_read=True)  # fmt: skip
    assert not r.log, "nothing was staged or enqueued"
    with expect_error(ValueError, "certificate row"):  # the engine-level guard too
        r.e._def_enqueue("s", tok, pos, _pt(B), None, nvd, (), {}, r.e._engine_sampling(ss), None)
    # a sampled step without a certificate row still defers as before
    ss = _sampling(B, nvd, [0.8, 0.9], topk=3)
    assert r.e._cert_rows(ss).tolist() == [False, False]


def test_f_selection_rules(monkeypatch):
    B = 3
    r = _CertRig(monkeypatch, B)
    e = r.e
    nvd = torch.full((B,), K)
    cert = _cert_sampling(B, nvd, [0.8, 0.9, 1.0], [0])
    plain = _sampling(B, nvd, [0.8, 0.9, 1.0], topk=3)
    assert e._cert_rows(cert).tolist() == [True, False, False] and e._cert_rows(plain).tolist() == [False] * 3
    assert e._accept_trace(cert) == FSC and e._accept_trace(plain) == 13
    assert e._need_stats(cert) and not e._need_stats(plain) and not e._need_cands(cert)
    both = {**cert, "num_logprobs": torch.tensor([-2, 3, -2], dtype=torch.int32)}
    assert e._need_stats(both) and e._need_cands(both)
    p1 = _cert_sampling(B, nvd, [0.8, 0.9, 1.0], [0], topp=1.0)
    assert not e._cert_rows(p1).any() and e._sampled_ineligible(p1) == "top_k_range"  # top_p = 1: the old path
    assert e._sampled_ineligible(cert) is None
    greedy0 = {**cert, "greedy": torch.tensor([True, False, False])}
    assert not e._cert_rows(greedy0).any()
    e._fs_trace_c = (
        None  # no certificate trace (kill switch off, no stats trace): a top_k = 0 row is out of range as before
    )
    assert e._sampled_ineligible(cert) == "top_k_range" and e._sampled_ineligible(plain) is None
    e._fs_trace_c = FSC
    r.m._vfy_stats_ready = False  # the stats are gone
    assert e._sampled_ineligible(cert) == "top_k_range"


def test_g_without_the_certificate_trace_a_top_k0_row_takes_the_exact_host_walk(monkeypatch):
    B = 2
    r = _CertRig(monkeypatch, B, cert=False)
    logits = _peaked_logits(B * T, 15)
    r.set_logits(logits)
    tok, pos, nvd = _draft_block(logits, B, [3, 2])
    ss = _cert_sampling(B, nvd, [0.8, 0.9], [0, 1])
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), ss)
    assert r.e.fused_fallbacks == {"top_k_range": 1}
    rc, rn = _redo_of(logits, tok, ss, nvd, B)
    assert torch.equal(res.accepted_counts, rn) and torch.equal(res.accepted_token_ids[:, 0], rc[:, 0])


class _FakeFs:
    def __init__(self, log):
        self.log, self.stats_in = log, None
        self.bind_stats = lambda t: (log.append(("bind_stats", t)), setattr(self, "stats_in", t))
        self.bind_ids = lambda t: log.append(("bind_ids",))
        self.bind_union = lambda a, b: log.append(("bind_union",))

    def body(self, cert=False):
        self.log.append(("body", cert))


def test_h_prepare_and_capture_of_the_certificate_variant(monkeypatch):
    B = 2
    r = _CertRig(monkeypatch, B, cert=False)
    e, m, log = r.e, r.m, []
    fs = _FakeFs(log)
    e._fs, e._fs_trace, e._fs_trace_c = fs, None, None
    tids = iter(range(100, 110))
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda *a, **k: log.append(("begin",)) or next(tids))
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda *a, **k: log.append(("end",)))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *a, **k: None)
    e._log_trace_memory = lambda *a: None
    # prepare: needs QWEN36_SPEC_DEVICE_TOPP and the stats trace
    monkeypatch.setenv("QWEN36_SPEC_DEVICE_TOPP", "0")
    e._prepare_cert_body()
    assert ("body", True) not in log
    monkeypatch.setenv("QWEN36_SPEC_DEVICE_TOPP", "1")
    m._vfy_stats_ready = False
    e._prepare_cert_body()
    assert ("body", True) not in log
    m._vfy_stats_ready = True
    e._prepare_cert_body()
    assert log == [("bind_stats", m._vfy_stats_out), ("body", True)]
    # capture: the plain trace first, then the variant (same buffers); a build without stats_in captures the plain one only
    del log[:]
    e._capture_fused_sampled()
    assert [x for x in log if x[0] in ("begin", "end", "body")] == [
        ("begin",), ("body", False), ("end",), ("begin",), ("body", True), ("end",),
    ]  # fmt: skip
    assert e._fs_trace == 100 and e._fs_trace_c == 101
    del log[:]
    fs.stats_in, e._fs_trace, e._fs_trace_c = None, None, None
    e._capture_fused_sampled()
    assert ("body", True) not in log and e._fs_trace == 102 and e._fs_trace_c is None


def test_i_the_adapter_gates_a_top_k0_row_on_the_capability(expect_error):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    cls, B = qv.Qwen36MTPContractForCausalLM, 3
    caps = {}
    fake = types.SimpleNamespace(_SPEC_K=K, model_capabilities=caps)
    chk = lambda ss: cls._check_spec_sampling(fake, ss, "fused_sample", B)  # noqa: E731
    nvd = torch.full((B,), K)
    plain = _sampling(B, nvd, [0.8, 0.9, 1.0], topk=3)
    assert chk(plain) is plain
    cert = _cert_sampling(B, nvd, [0.8, 0.9, 1.0], [1])
    with expect_error(ValueError, "supports_spec_device_top_p"):
        chk(cert)
    with expect_error(ValueError, "supports_spec_device_top_p"):  # the capability for logprobs is another one
        caps["supports_spec_device_logprobs"] = True
        chk(cert)
    caps["supports_spec_device_top_p"] = True  # what the capability line would set
    assert chk(cert) is cert
    g0 = {**cert, "greedy": torch.tensor([False, True, False])}  # a greedy row ignores top_k / top_p
    caps.clear()
    assert chk(g0) is g0
    p1 = _cert_sampling(
        B, nvd, [0.8, 0.9, 1.0], [1], topp=1.0
    )  # top_p = 1: not a certificate row (the engine sends it to the host walk)
    assert chk(p1) is p1


def test_j_the_two_capabilities_are_declared_from_the_kill_switches(monkeypatch):
    """Human-approved capabilities (2026-10-28): both are plain bools in the class table, decided at import by the fused
    switches and their own kill switch. The table itself is built at import, so the test pins the switch functions the
    declaration is made of under each env setting, and that both keys are present."""
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv
    from models.demos.blackhole.qwen36.tt import tp_common as tc

    caps = qv.Qwen36MTPContractForCausalLM.model_capabilities
    assert isinstance(caps["supports_spec_device_logprobs"], bool) and isinstance(
        caps["supports_spec_device_top_p"], bool
    )

    def expected():
        base = tc.spec_fused_enabled() and tc.spec_fused_sampled_enabled()
        return base and tc.spec_device_logprobs_enabled(), base and tc.spec_device_topp_enabled()

    for env, want in (
        ({"QWEN36_SPEC_FUSED": "1"}, (True, True)),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_DEVICE_LOGPROBS": "0"}, (False, True)),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_DEVICE_TOPP": "0"}, (True, False)),
        ({"QWEN36_SPEC_FUSED": "1", "QWEN36_SPEC_FUSED_SAMPLED": "0"}, (False, False)),
        ({"QWEN36_SPEC_FUSED": "0"}, (False, False)),
    ):
        for key in (
            "QWEN36_SPEC_FUSED",
            "QWEN36_SPEC_FUSED_SAMPLED",
            "QWEN36_SPEC_DEVICE_LOGPROBS",
            "QWEN36_SPEC_DEVICE_TOPP",
        ):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert expected() == want, env
