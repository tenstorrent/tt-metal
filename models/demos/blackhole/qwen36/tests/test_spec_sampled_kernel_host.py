# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED, no device. tt/spec_sampled_kernel_math.py is the bit-exact host model of
kernels/spec_accept_sampled_fused.cpp (the device test pins the C++ to it word for word); here the model is checked
against everything the kernel must agree with:
  * its integer exp and fixed-point rounding against 80-digit Decimal values (the exp is the correctly rounded float32);
  * its merge against union_topk_stable (the reference's stable descending sort of the union);
  * fused_sampled_reference (the S2 algorithm the op graph and the engine's audit implement) on thousands of random
    blocks at K = 3 and at the adaptive-K wide family's K = 7: every user equal except float32 boundary users (an
    examined accept within 1e-6 of its uniform, an inverse-CDF edge, a top-p tail edge), overflow flags equal, pack /
    reseed table equal to the S1 drafter tail of the model's own commit;
  * losslessness: the joint (count, first committed token) and (count, last committed token) of the model's walk over
    30000 draws match the exact target distribution (chi-square at p = 0.999, plus the power check that a 3% mass shift
    is detected), at K = 3 and K = 7, with and without top-p / ties;
  * the decomposed walk (row states + uniforms, used above) equals the whole model;
  * the adaptive-K guard (state width Bs != B refused by every kernel), the kernel plan / L1 budget, the flag default and
    the audit's top-p class (only with the kernel attached).
Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_sampled_kernel_host.py -s"""

import math
from decimal import Decimal, getcontext
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_sampled_kernel_math as km
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.spec_fused import fused_greedy_reference
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    _ONE_MINUS,
    FusedSampledOut,
    audit_boundary_users,
    fused_drafter_reference,
    fused_sampled_reference,
    pack_samp_host,
    union_topk_stable,
)

fsh = pytest.importorskip("models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_host")
EPS = 1e-6
W = 32


# ------------------------------------------------------------------------------------------------ helpers
def _bf16_words(t):
    return torch.as_tensor(t).to(torch.bfloat16).view(torch.int16).numpy().astype(np.int64) & 0xFFFF


def _f32_words(x):
    return [int(v) for v in np.asarray(x, dtype=np.float32).reshape(-1).view(np.uint32)]


def _dec_to_f32_bits(d):
    """Nearest float32 (ties to even) of a positive Decimal, normal range."""
    getcontext().prec = 80
    e = int(math.floor(float(d.log10() / Decimal(2).log10())))
    while Decimal(2) ** e > d:
        e -= 1
    while Decimal(2) ** (e + 1) <= d:
        e += 1
    q = d / (Decimal(2) ** (e - 23))  # in [2^23, 2^24)
    fl = int(q)
    rem = q - fl
    if rem > Decimal("0.5") or (rem == Decimal("0.5") and fl & 1):
        fl += 1
    if fl == 1 << 24:
        fl, e = 1 << 23, e + 1
    return ((e + 127) << 23) | (fl & 0x7FFFFF)


def _exact_exp_bits(xb):
    getcontext().prec = 80
    x = Decimal(float(km.f32(xb)))
    y = x.exp()
    if y < Decimal(2) ** -126:
        return 0  # the model's flush rule
    return _dec_to_f32_bits(y)


def _twin_case(c, uni, K, M, per_shard, TP):
    """The model on one fsh._case (staging exactly what FusedSampledDevice.stage_sampled stages)."""
    B = c["B"]
    T = K + 1
    samp = pack_samp_host(c["temp"], c["top_k"], c["top_p"], c["greedy"], uni, W).numpy()
    ids = c["mi"].reshape(B, T, M)[:, :, 0].reshape(-1)
    aux = torch.zeros(7, W, dtype=torch.int32)
    aux[:, : c["aux"].shape[1]] = c["aux"]
    return km.sampled_accept_twin(
        _bf16_words(c["uv"]),
        c["ul"].numpy(),
        c["tok_buf"].numpy(),
        ids.numpy(),
        aux.numpy(),
        samp,
        c["pt"].repeat_interleave(T, dim=0).numpy(),
        B=B,
        K=K,
        M=M,
        TP=TP,
        per_shard=per_shard,
        W=W,
        nb=int(c["pt"].shape[1]),
    )


def _boundary(c, ref, uni, u, m_cols, K, M):
    """True if a decision of user u is within EPS of its threshold on a row examined under either walk."""
    T = K + 1
    nv = int(c["nv"][u])
    prob, cum = ref.prob.double().reshape(-1, T, M)[u], ref.cum.double().reshape(-1, T, M)[u]
    gid = ref.cand_ids.reshape(-1, T, M)[u]
    mx = max(m_cols)
    for j in range(min(mx, nv - 1) + 1) if nv else []:
        if abs(float(ref.p_d[u, j]) - float(uni[u, 1 + j])) < EPS:
            return True
    for m in set(m_cols):
        corr = m < nv
        w = prob[m].clone()
        if corr:
            w[gid[m] == c["drafts"][u, m]] = 0
        cdf = w.cumsum(0) / w.sum()
        uu = float(uni[u, K + 1] if corr else uni[u, 0])
        if float((cdf - uu).abs().min()) < EPS:
            return True
    if float(c["top_p"][u]) < 1.0:
        for m in range(mx + 1):
            x = cum[m][cum[m] > 0]
            if x.numel() and float((x - (1 - float(c["top_p"][u]))).abs().min()) < EPS:
                return True
    return False


def _use_k(monkeypatch, k):
    monkeypatch.setattr(fsh, "K", int(k))
    monkeypatch.setattr(fsh, "T", int(k) + 1)


# ------------------------------------------------------------------------------------------------ primitives
def test_constants_match_high_precision():
    getcontext().prec = 80
    two, ln2 = Decimal(2), Decimal(2).ln()
    rnd = lambda x: int(x.to_integral_value(rounding="ROUND_HALF_EVEN"))
    assert km.L2E == rnd(two**62 / ln2)
    assert km.LN2_64 == rnd(ln2 * two**64)
    assert km.POLY == tuple(rnd(two**62 / Decimal(math.factorial(k))) for k in range(7))
    assert km.EXP2_TAB == tuple(rnd(two ** (Decimal(i) / 64) * two**62) for i in range(64))
    # the kernel source carries the same literals
    import pathlib

    src = (pathlib.Path(km.__file__).parent / "kernels" / "spec_accept_sampled_fused.cpp").read_text()
    for v in (km.L2E, km.LN2_64, *km.POLY, *km.EXP2_TAB):
        assert f"0x{v:016X}ull" in src, hex(v)


def test_exp_is_correctly_rounded():
    g = np.random.default_rng(5)
    xs = list(-g.random(20000).astype(np.float32) * np.float32(30))  # the softmax range of tempered logits
    xs += list(-g.random(4000).astype(np.float32) * np.float32(104))  # down through the flush threshold
    xs += list(-(np.float32(2) ** -g.integers(1, 60, 2000).astype(np.float32)) * g.random(2000).astype(np.float32))
    xs += [0.0, -0.0, -1e-30, -2.0**-25, -2.0**-24, -87.3365447, -87.3365, -87.34, -103.9, -126.0]
    # the table / exponent seams: x * log2(e) just above and below every integer -1 .. -126 (2^f near 1 and near 2)
    for n in range(1, 127):
        c = np.float32(-n * math.log(2.0))
        xs += [float(np.nextafter(np.nextafter(c, np.float32(0)), np.float32(0))), float(c)]
        xs += [float(np.nextafter(c, np.float32(-1e9))), float(np.nextafter(c, np.float32(0)))]
    bad, n, torch_diff = [], 0, 0
    for x in xs:
        xb = km.bits(np.float32(x))
        got = km.exp_f32(xb)
        want = _exact_exp_bits(xb) if xb & 0x7FFFFFFF else km.F_ONE
        n += 1
        if got != want:
            bad.append((x, hex(got), hex(want)))
        torch_diff += int(km.bits(torch.exp(torch.tensor([x], dtype=torch.float32)).numpy()[0]) != got)
    for xb, want in ((0xFF800000, 0), (0x7F800000, km.F_INF), (0x7FC00000, km.F_NAN), (0x80000001, km.F_ONE)):
        assert km.exp_f32(xb) == want
    print(f"\n[exp] {n} inputs: {len(bad)} not correctly rounded; differs from torch.exp on {torch_diff}")
    assert not bad, bad[:5]


def test_fixed_point_rounding():
    g = np.random.default_rng(9)
    for frac in (km.E_FRAC, km.P_FRAC):
        for _ in range(20000):
            v = int(g.integers(1, 1 << 53, dtype=np.int64)) >> int(g.integers(0, 50))
            if v == 0:
                continue
            want = km.bits(np.float32(np.float64(v) / np.float64(2.0**frac)))  # v < 2^53: the double is exact
            assert km.fix2f(v, frac) == want, (v, frac)
        for _ in range(5000):
            x = np.float32(g.random() * 1.5)
            b = km.bits(x)
            assert km.f2fix(b, frac) == math.floor(Fraction(float(x)) * 2**frac), (x, frac)
    # ties to even at the 24-bit boundary
    assert km.fix2f((1 << 24) + 1, 0) == km.bits(np.float32(16777216.0))
    assert km.fix2f((1 << 24) + 3, 0) == km.bits(np.float32(16777220.0))
    assert km.f2int_trunc(km.bits(np.float32(20.0))) == 20 and km.f2int_trunc(km.bits(np.float32(-3.0))) == 0


def test_reciprocal_division_is_ieee():
    """fdiv_r (the kernel's reciprocal division) == IEEE float32 division on every input class the kernel feeds it (bf16
    logits / temperature, e / s with denormal quotients) and on random words of any sign / exponent."""
    g = np.random.default_rng(13)
    n = 120000
    cases = []
    # e / s: e in [2^-126, 1] (incl. tiny ones whose quotient is denormal), s in [1, 64]
    e = (g.random(n).astype(np.float32) * np.float32(2) ** -g.integers(0, 126, n).astype(np.float32)).astype(np.float32)
    s = (np.float32(1) + g.random(n).astype(np.float32) * np.float32(63)).astype(np.float32)
    cases.append((e, s))
    # bf16 logits / temperature in [1e-3, 100]
    lv = (g.standard_normal(n).astype(np.float32) * np.float32(12)).astype(np.float32)
    lv = (lv.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    tv = (np.float32(10) ** (g.random(n).astype(np.float32) * np.float32(5) - np.float32(3))).astype(np.float32)
    cases.append((lv, tv))
    # random words: any numerator, any positive divisor word (the fallbacks included)
    a = g.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    b = (g.integers(0, 1 << 31, n, dtype=np.uint64).astype(np.uint32)).view(np.float32)
    cases.append((a, b))
    # exact ties: (2k+1) * 2^-25 style midpoints from small integer ratios
    q = np.arange(1, 4097, dtype=np.float32)
    cases.append((q * np.float32(3), np.full_like(q, 2.0)))
    bad = 0
    for av, bv in cases:
        for x, y in zip(av.view(np.uint32).tolist(), bv.view(np.uint32).tolist()):
            if km.fdiv_r(x, y, km.rcp62(y)) != km.fdiv(x, y):
                bad += 1
    for x, y in ((0, km.F_ONE), (0x80000000, km.F_ONE), (1, km.F_ONE), (km.F_INF, km.F_ONE), (km.F_NAN, km.F_ONE)):
        assert km.fdiv_r(x, y, km.rcp62(y)) == km.fdiv(x, y)
    assert km.rcp62(0xBF800000) == 0 and km.rcp62(0) == 0 and km.rcp62(km.F_INF) == 0
    assert bad == 0, bad


def test_merge_equals_union_topk_stable():
    g = torch.Generator().manual_seed(3)
    TP, M, PS = fsh.TP, fsh.M, fsh.PER_SHARD
    for _ in range(60):
        kk = torch.randint(1, 64, (32,), generator=g)
        uv, ul = fsh._union(g, 32, kk)
        v_ref, i_ref = union_topk_stable(uv, ul, PS, M)
        words = _bf16_words(uv)
        for r in range(32):
            mg = km._Merge(words[r], ul[r].numpy(), TP, M, PS)
            got = [mg.next() for _ in range(M)]
            want_h = _bf16_words(v_ref[r])
            assert [h for h, _ in got] == [int(x) for x in want_h], r
            assert [gid for _, gid in got] == i_ref[r].tolist(), r


# ------------------------------------------------------------------------------------------------ model vs reference
def _vs_reference(n_cases, K, seed):
    g = torch.Generator().manual_seed(seed)
    TP, M, PS = fsh.TP, fsh.M, fsh.PER_SHARD
    T = K + 1
    st = dict(users=0, sampled=0, ovf=0, compared=0, exact=0, boundary=0, boundary_equal=0)
    for case in range(n_cases):
        B = (1, 4, 8)[case % 3] if K == 3 else (1, 2, 4)[case % 3]
        c = fsh._case(g, B)
        uni = torch.rand(B, K + 2, generator=g).clamp(max=_ONE_MINUS)
        ids = c["mi"].reshape(B, T, M)[:, :, 0].reshape(-1)
        ref = fused_sampled_reference(
            c["uv"], c["ul"], c["tok_buf"], ids, c["aux"], c["pt"], B=B, K=K, per_shard=PS, M=M,
            greedy=c["greedy"], temperature=c["temp"], top_k=c["top_k"], top_p=c["top_p"], uniforms=uni,
        )  # fmt: skip
        pk, rs, rr = _twin_case(c, uni, K, M, PS, TP)
        R = B * T
        committed = torch.as_tensor(((rr[0, :R] + 2**31) % 2**32 - 2**31).reshape(B, T))
        counts = torch.as_tensor(rr[1, :B])
        ovf = torch.as_tensor(rr[2, :B])
        assert not rr[0, R:].any() and not rr[1, B:].any() and not rr[2, B:].any(), "result padding"
        for u in range(B):
            st["users"] += 1
            st["sampled"] += int(not bool(c["greedy"][u]))
            same = torch.equal(committed[u], ref.committed[u].to(torch.int64)) and int(counts[u]) == int(ref.counts[u])
            bnd = None
            if int(ovf[u]) != int(ref.overflow[u]):
                bnd = _boundary(c, ref, uni, u, [int(ref.counts[u]) - 1, int(counts[u]) - 1], K, M)
                assert bnd and int(counts[u]) != int(ref.counts[u]), (case, u, "overflow flag")
                continue
            if int(ref.overflow[u]):
                st["ovf"] += 1
                continue
            st["compared"] += 1
            if same:
                st["exact"] += 1
                continue
            bnd = _boundary(c, ref, uni, u, [int(ref.counts[u]) - 1, int(counts[u]) - 1], K, M)
            assert bnd, (case, u, committed[u].tolist(), ref.committed[u].tolist(), int(counts[u]), int(ref.counts[u]))
            st["boundary"] += 1
        # the drafter tail is the S1 tail of the model's own commit, word for word
        pack, rs_pt = fused_drafter_reference(committed, counts, c["aux"], c["pt"], B=B, K=K)[:2]
        want_pk = np.zeros((2 * K + 5, W), dtype=np.int64)
        want_pk[:, : pack.shape[1]] = pack.to(torch.int64).numpy() & 0xFFFFFFFF
        assert np.array_equal(pk, want_pk), (case, "pack")
        assert np.array_equal(rs, rs_pt.to(torch.int64).numpy() & 0xFFFFFFFF), (case, "rs_pt")
    return st


def test_model_matches_reference_k3():
    st = _vs_reference(2400, 3, 20261031)
    print(f"\n[model vs reference K=3] {st}")
    assert st["compared"] > 0.6 * st["users"] and st["boundary"] <= max(3, st["compared"] // 500)


def test_model_matches_reference_k7(monkeypatch):
    _use_k(monkeypatch, 7)
    st = _vs_reference(600, 7, 20261107)
    print(f"\n[model vs reference K=7] {st}")
    assert st["compared"] > 0.5 * st["users"] and st["boundary"] <= max(3, st["compared"] // 500)


def test_greedy_users_equal_s1():
    """An all-greedy (or all-inactive) step commits exactly what the S1 reference commits."""
    g = torch.Generator().manual_seed(17)
    K, M, PS, TP = 3, fsh.M, fsh.PER_SHARD, fsh.TP
    T = K + 1
    for case in range(200):
        B = (1, 4, 8)[case % 3]
        c = fsh._case(g, B)
        c["greedy"] = torch.ones(B, dtype=torch.bool)
        uni = torch.rand(B, K + 2, generator=g).clamp(max=_ONE_MINUS)
        ids = c["mi"].reshape(B, T, M)[:, :, 0].reshape(-1)
        ids = torch.where(torch.rand(ids.shape, generator=g) < 0.6, c["tok_buf"].roll(-1), ids)  # accept some drafts
        c["mi"] = c["mi"].clone()
        c["mi"].reshape(B, T, M)[:, :, 0] = ids.reshape(B, T)
        pk, rs, rr = _twin_case(c, uni, K, M, PS, TP)
        s1 = fused_greedy_reference(ids, c["tok_buf"], c["aux"], c["pt"], B=B, K=K)
        R = B * T
        assert np.array_equal(((rr[0, :R] + 2**31) % 2**32 - 2**31), s1.committed.reshape(-1).numpy())
        assert np.array_equal(rr[1, :B], s1.counts.numpy()) and not rr[2].any()
        assert np.array_equal(rs, s1.rs_pt.to(torch.int64).numpy() & 0xFFFFFFFF)


# ------------------------------------------------------------------------------------------------ losslessness
def _expected_count_last(tgt, drafts, nv):
    """P(count, last committed token) of the exact point-mass walk: stop at row m (accepted 0..m-1) and draw from row m
    without the draft (m < nv) or with it (bonus, m == nv)."""
    out = {}
    run = 1.0
    for m in range(nv + 1):
        for t, q in tgt[m].items():
            if m < nv and t == drafts[m]:
                continue
            out[(m + 1, t)] = out.get((m + 1, t), 0.0) + run * q
        if m < nv:
            run *= tgt[m].get(drafts[m], 0.0)
    return out


def _chi(obs, exp, N):
    keys = set(exp) | set(obs)
    big = [kk for kk in keys if N * exp.get(kk, 0.0) >= 5]
    s = sum((obs.get(kk, 0) - N * exp[kk]) ** 2 / (N * exp[kk]) for kk in big)
    rest_e = N * (1 - sum(exp[kk] for kk in big))
    rest_o = N - sum(obs.get(kk, 0) for kk in big)
    df = len(big) - 1
    if rest_e >= 5:
        s += (rest_o - rest_e) ** 2 / rest_e
        df += 1
    else:
        assert rest_o <= 10 + 5 * rest_e, (rest_o, rest_e)
    return s, df, big


@pytest.mark.parametrize(
    "K,temp,k,p,nv,ranks",
    [
        (3, 1.0, 20, 1.0, 3, [0, 0, 1]),
        (3, 0.6, 20, 0.95, 3, [0, 1, 0]),  # production sampling (temperature 0.6, top_k 20, top_p 0.95)
        (3, 0.7, 40, 0.8, 3, [0, 3, 0]),
        (3, 1.2, 10, 0.5, 2, [0, -1, 0]),
        (3, 0.9, 30, 0.95, 0, [0, 0, 0]),
        (7, 0.6, 20, 0.95, 7, [0, 0, 1, 0, 0, 2, 0]),
        (7, 0.9, 30, 0.95, 5, [0, 0, 0, 1, 0, -1, 0]),
    ],
)
def test_walk_is_lossless(monkeypatch, K, temp, k, p, nv, ranks):
    """The model's walk over 30000 uniform tables: the joint (count, first token) and (count, last token) match the exact
    float64 target distribution of every row (chi-square at p = 0.999 + a 3% mass-shift power check)."""
    _use_k(monkeypatch, K)
    T, M, PS, TP = K + 1, fsh.M, fsh.PER_SHARD, fsh.TP
    g = torch.Generator().manual_seed(1000 + 7 * k + K)
    while True:  # a block with no overflow
        uv, ul = fsh._union(g, T, torch.full((T,), k))
        mv, mi = fsh.merge_shard_topk(uv, ul, PS, M)
        if not bool((mv[:, M - 1] >= mv[:, k - 1]).any()):
            break
    drafts = [int(mi[j, r]) if r >= 0 else 12345 for j, r in enumerate(ranks)]
    tgt = [fsh._target(mv[j], mi[j], temp, k, p) for j in range(T)]
    exp_first = fsh._expected_joint(tgt, drafts, nv)
    exp_last = _expected_count_last(tgt, drafts, nv)
    temp_b = km.bits(np.float32(temp))
    omp_b = km.fsub(km.F_ONE, km.bits(np.float32(p)))
    states = km.user_row_states(_bf16_words(uv), ul.numpy(), drafts, nv, K=K, TP=TP, M=M, per_shard=PS, temp_b=temp_b, k=k, omp_b=omp_b)
    assert not any(s.ovf for s in states)
    N = 30000
    uni = torch.rand((N, K + 2), generator=torch.Generator().manual_seed(99)).clamp(max=_ONE_MINUS)
    words = np.asarray(uni.numpy(), dtype=np.float32).view(np.uint32)
    of, ol = {}, {}
    for n in range(N):
        cnt, comm, ovf = km.user_walk(states, [int(x) for x in words[n]], nv, K, M, drafts)
        of[(cnt, comm[0])] = of.get((cnt, comm[0]), 0) + 1
        ol[(cnt, comm[-1])] = ol.get((cnt, comm[-1]), 0) + 1
    for name, obs, exp in (("first", of, exp_first), ("last", ol, exp_last)):
        assert all(kk in exp for kk in obs if obs[kk] > 0), f"{name}: committed (count, token) outside the target support"
        s, df, big = _chi(obs, exp, N)
        crit = fsh._chi_crit(df)
        print(f"\n[chi2 {name}] K={K} T={temp} k={k} p={p} nv={nv}: stat={s:.1f} df={df} crit={crit:.1f}")
        assert s < crit
        biased = dict(obs)
        donor = max(obs, key=obs.get)
        recv = min((kk for kk in big if kk != donor), key=exp.get)
        moved = int(0.03 * N)
        biased[donor] -= moved
        biased[recv] = biased.get(recv, 0) + moved
        assert _chi(biased, exp, N)[0] > crit


def test_walk_equals_model():
    """user_row_states + user_walk (the chi-square path) == sampled_accept_twin (the kernel's model) on random blocks."""
    g = torch.Generator().manual_seed(23)
    K, M, PS, TP = 3, fsh.M, fsh.PER_SHARD, fsh.TP
    T = K + 1
    n = 0
    for case in range(150):
        B = 4
        c = fsh._case(g, B)
        uni = torch.rand(B, K + 2, generator=g).clamp(max=_ONE_MINUS)
        _, _, rr = _twin_case(c, uni, K, M, PS, TP)
        words = np.asarray(uni.numpy(), dtype=np.float32).view(np.uint32)
        vals = _bf16_words(c["uv"])
        for u in range(B):
            if bool(c["greedy"][u]):
                continue
            nv = int(c["nv"][u])
            temp_b = km.bits(np.float32(float(c["temp"][u])))
            omp_b = km.fsub(km.F_ONE, km.bits(np.float32(float(c["top_p"][u]))))
            dr = c["tok_buf"].reshape(B, T)[u, 1:].tolist()
            states = km.user_row_states(
                vals[u * T : (u + 1) * T], c["ul"][u * T : (u + 1) * T].numpy(), dr, nv,
                K=K, TP=TP, M=M, per_shard=PS, temp_b=temp_b, k=int(c["top_k"][u]), omp_b=omp_b,
            )  # fmt: skip
            cnt, comm, ovf = km.user_walk(states, [int(x) for x in words[u]], nv, K, M, dr)
            assert cnt == int(rr[1, u]) and ovf == int(rr[2, u])
            assert comm == [int(x) for x in rr[0, u * T : u * T + cnt]]
            n += 1
    assert n > 300


# ------------------------------------------------------------------------------------------------ guards, plan, flag, audit
def test_state_width_guard():
    from models.demos.blackhole.qwen36.tt.spec_fused_kernels import ChainKernel, SampledAcceptKernel, state_width_reason

    assert state_width_reason(SimpleNamespace(B=8)) is None
    assert state_width_reason(SimpleNamespace(B=8, Bs=8)) is None
    why = state_width_reason(SimpleNamespace(B=4, Bs=8))
    assert why and "Bs=8" in why
    # supports() refuses before it looks at a single buffer (no tensors on these stand-ins)
    wide_chain = SimpleNamespace(B=4, Bs=8, K=7, Nv=6, R=32)
    assert ChainKernel.supports(wide_chain) == why
    wide_sampled = SimpleNamespace(B=4, Bs=8, K=7, R=32, W=32, nb=4097, TP=8, M=64)
    assert SampledAcceptKernel.supports(wide_sampled) == why


@pytest.mark.parametrize("B,K", [(8, 3), (4, 7), (1, 3), (16, 1)])
def test_sampled_plan_and_l1(B, K):
    from models.demos.blackhole.qwen36.tt.spec_fused_kernels import sampled_l1_bytes, sampled_plan

    R = B * (K + 1)
    for nb in list(range(1, 80)) + [537, 2048, 4096, 4097, 4112, 4113, 8193]:
        nx, ny, rs, scr = sampled_plan(R, nb, 11)
        assert nx == min(R, 11) and ny == -(-R // nx)
        assert rs % 64 == 0 and scr % 64 == 0 and rs >= 64 and scr >= 64
        row = 4 * nb
        pieces = -(-row // rs)
        assert pieces * rs >= row and (pieces - 1) * rs < row
        l1 = sampled_l1_bytes(B, K, 32, 64, 8, rs, scr) + 64
        assert l1 <= 48 * 1024, (B, K, nb, l1)
    assert sampled_plan(32, 4097, 11)[2] == 16448  # the serving width: one read per row


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED", raising=False)
    assert not tp_common.spec_accept_kernel_sampled_enabled()
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED", "1")
    assert tp_common.spec_accept_kernel_sampled_enabled()


def test_audit_topp_class_only_with_kernel():
    """A user whose difference comes from a top-p tail edge is explained only when top_p is passed (kernel attached);
    the flag-off call (top_p=None) keeps today's two classes."""
    B, K, M = 1, 3, 4
    T = K + 1
    prob = torch.zeros(B * T, M)
    prob[:, :3] = torch.tensor([0.6, 0.35, 0.05])
    cum = torch.zeros(B * T, M)
    cum[:, :3] = torch.tensor([1.0, 0.4, 0.05 + 3e-7])  # within 1e-6 of 1 - top_p = 0.05
    gids = torch.arange(M).repeat(B * T, 1)
    ref = FusedSampledOut(
        committed=torch.tensor([[7, 8, -1, -1]], dtype=torch.int32),
        counts=torch.tensor([2], dtype=torch.int32),
        overflow=torch.zeros(1, dtype=torch.int32),
        pack=None, rs_pt=None, mtp_pt=None, declined=None, num_valid_out=None, drafter_runs=True,
        prob=prob, cand_ids=gids, p_d=torch.tensor([[0.9, 0.2, 0.3]]), cum=cum,
    )  # fmt: skip
    uni = torch.tensor([[0.5, 0.1, 0.7, 0.1, 0.5]])  # far from every p_d: not an accept boundary
    dev_committed = torch.tensor([[7, 9, 2, -1]])
    dev_counts = torch.tensor([3])
    drafts = torch.tensor([[7, 9, 5]])
    nv = torch.tensor([3])
    off = audit_boundary_users(ref, uni, dev_committed, dev_counts, drafts, nv, B=B, K=K, M=M)
    on = audit_boundary_users(ref, uni, dev_committed, dev_counts, drafts, nv, B=B, K=K, M=M, top_p=torch.tensor([0.95]))
    assert off == {} and on == {0: "topp"}
    far = cum.clone()
    far[:, 2] = 0.07
    ref2 = ref._replace(cum=far)
    assert audit_boundary_users(ref2, uni, dev_committed, dev_counts, drafts, nv, B=B, K=K, M=M, top_p=torch.tensor([0.95])) == {}
