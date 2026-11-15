# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Hy4 MTP speculative-decode accounting. Run: python -m pytest --noconftest -c /dev/null --rootdir=/tmp <this dir> -q"""
import random

import pytest

from models.demos.blackhole.hy4_preview.spec import accept, policy, speedup
from models.demos.blackhole.hy4_preview.spec.rollback import PAGE, SimStore, UserSpecState, pages_for

VOCAB = 11


# ---------------------------------------------------------------- accept
def test_accept_count_and_finish():
    assert accept.accept_count([1, 2, 3], [1, 2, 3, 9]) == 3
    assert accept.accept_count([1, 2, 3], [1, 5, 3, 9]) == 1
    assert accept.accept_count([1, 2, 3], [7, 2, 3, 9]) == 0
    r = accept.finish_round(100, [1, 2, 3], [1, 2, 3, 9])
    assert (r.m, r.emitted, r.next_pending, r.new_len) == (3, (1, 2, 3, 9), 9, 104)
    r = accept.finish_round(100, [1, 2, 3], [1, 5, 3, 9])
    assert (r.m, r.emitted, r.next_pending, r.new_len) == (1, (1, 5), 5, 102)
    r = accept.finish_round(100, [1, 2, 3], [7, 2, 3, 9])
    assert (r.m, r.emitted, r.new_len) == (0, (7,), 101)


def test_cap_and_keff():
    assert accept.finish_round(0, [1, 2, 3], [1, 2, 3, 9], cap=1).emitted == (1, 2)
    assert accept.finish_round(0, [1, 2, 3], [1, 2, 3, 9], cap=0).emitted == (1,)
    assert accept.finish_round(0, [1, 2, 3], [1, 2, 3, 9], k_eff=2).emitted == (1, 2, 3)


def test_truncate_at_stop():
    assert accept.truncate_at_stop((5, 6, 7, 8), [7], 10) == ((5, 6, 7), True)
    assert accept.truncate_at_stop((5, 6, 7, 8), [], 2) == ((5, 6), True)
    assert accept.truncate_at_stop((5, 6), [], 5) == ((5, 6), False)


def _rand_dist(rng, n=5, peak=0.0):
    w = [rng.random() + 0.05 for _ in range(n)]
    w[rng.randrange(n)] += peak
    return accept._norm(w)


def test_sample_then_compare_equals_rejection_distribution():
    """Greedy (one-hot) draft: sample-then-compare and the textbook rejection rule emit the same joint distribution of (first emitted token, length)."""
    rng_p = random.Random(1)
    k, n = 2, 4
    rows = [_rand_dist(rng_p, n, 1.0) for _ in range(k + 1)]
    drafts = [
        max(range(n), key=lambda t: rows[i][t]) if i == 0 else 1 for i in range(k)
    ]  # first draft = argmax of row 0, second arbitrary
    N = 60000
    ra, rb = random.Random(7), random.Random(8)
    ca, cb = {}, {}
    for _ in range(N):
        a = tuple(accept.rejection_accept(drafts, rows, ra))
        b = tuple(accept.sample_then_compare(drafts, rows, rb))
        ca[a] = ca.get(a, 0) + 1
        cb[b] = cb.get(b, 0) + 1
    for key in set(ca) | set(cb):
        pa, pb = ca.get(key, 0) / N, cb.get(key, 0) / N
        assert abs(pa - pb) < 0.012, (key, pa, pb)


def test_rejection_preserves_target_marginal_first_token():
    """The first emitted token is distributed as the target row-0 distribution, whatever the draft was (speculative sampling exactness)."""
    rng = random.Random(3)
    n = 5
    rows = [_rand_dist(rng, n) for _ in range(3)]
    for d0 in (0, 3):
        r, N = random.Random(11), 50000
        cnt = [0] * n
        for _ in range(N):
            cnt[accept.rejection_accept([d0, 1], rows, r)[0]] += 1
        for t in range(n):
            assert abs(cnt[t] / N - rows[0][t]) < 0.012


# ---------------------------------------------------------------- rollback
def test_pages_and_round_lifecycle():
    s = UserSpecState(length=500, pages=pages_for(500))
    assert s.pages == 1
    extra = s.begin_round(3)  # rows 500..503 fit in page 0 .. page boundary at 512: 500+4 = 504 <= 512
    assert extra == 0
    s.commit(3)
    assert s.length == 504
    s.begin_round(3)  # rows 504..507
    s.commit(0)
    assert s.length == 505
    s.length = 510
    extra = s.begin_round(3)  # rows 510..513 cross the page boundary
    assert extra == 1 and s.pages == 2
    assert [s.row_nvalid(j) for j in range(4)] == [511, 512, 513, 514]
    s.commit(0)
    assert s.length == 511 and s.pages == 2  # pages never shrink within a request
    s.length = 20
    s.begin_round(3)
    assert list(s.stale_rows(1)) == [22, 23]


def test_clamp_k():
    s = UserSpecState(length=100, max_len=110)
    assert s.clamp_k(3, 100) == 3
    assert s.clamp_k(3, 2) == 1  # 2 tokens left: a round emits up to k+1 = 2 -> k = 1
    assert s.clamp_k(3, 1) == 0
    s.length = 108
    assert s.clamp_k(3, 100) == 1  # rows 108, 109 only


def test_seed_after_prefill():
    s = UserSpecState()
    s.seed_after_prefill(1000, 999)
    assert (s.length, s.mtp_len, s.pages) == (1000, 999, 2)
    s.seed_done()
    assert s.mtp_len == 1000
    s.begin_round(3)
    s.commit(2)
    assert s.length == s.mtp_len == 1003


def _raises_assert(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def test_double_begin_and_bad_commit():
    s = UserSpecState(length=10)
    s.begin_round(1)
    assert _raises_assert(lambda: s.begin_round(1))
    assert _raises_assert(lambda: s.commit(2))


# toy target model: the token at position p+1 is a function of (all tokens <= p) read from a KV store; the "draft" is the same function with a fraction of
# deliberate errors, computed from a separately maintained MTP store. Spec loop with stale writes and truncation must equal plain greedy decoding.
def target_next(prefix):
    h = 17
    for t in prefix:
        h = (h * 31 + t + 7) % 1000003
    return h % VOCAB


def plain_decode(prompt, n):
    toks = list(prompt)
    while len(toks) < len(prompt) + n:
        toks.append(target_next(toks))
    return toks[len(prompt) :]


def spec_decode(prompt, n, k, err_rate, seed):
    rng = random.Random(seed)
    kv = SimStore()  # main KV: row p = token at position p (the toy 'latent')
    mtp = SimStore()  # MTP KV: row p = token at p+1 (the pair), only needed for the reads
    L = len(prompt)
    for p, t in enumerate(prompt):
        kv.write(p, t)
    first = target_next(prompt)
    st = UserSpecState(length=L, pages=pages_for(L), max_len=L + n + 8)
    out = [first]
    pending = first  # x_L
    drafts_cache = None
    rounds = acc = 0
    while len(out) < n:
        kk = st.clamp_k(k, n - len(out))
        # draft k tokens by recursion from the committed state (toy: exact next-token function with errors, reading the committed prefix + pending)
        ctx = kv.read_prefix(st.length) + [pending]
        drafts = []
        c = list(ctx)
        for _ in range(kk):
            d = target_next(c)
            if rng.random() < err_rate:
                d = (d + 1) % VOCAB
            drafts.append(d)
            c.append(d)
        st.begin_round(kk)
        P = st.length
        rows_in = [pending] + drafts
        for j, t in enumerate(
            rows_in
        ):  # verify step writes ALL k+1 rows (stale ones too), each row reads only its causal prefix
            kv.write(P + j, t)
        tgt = []
        for j in range(kk + 1):
            prefix = kv.read_prefix(P + j + 1)  # row j sees positions <= P+j
            assert len(prefix) == st.row_nvalid(j)
            tgt.append(target_next(prefix))
        r = accept.finish_round(P, drafts, tgt, k_eff=kk)
        st.commit(r.m)
        out.extend(r.emitted)
        pending = r.next_pending
        rounds += 1
        acc += r.m
        # stale rows beyond the committed length still hold garbage: poison them to prove they are never read
        for pos in range(st.length, P + kk + 1):
            kv.write(pos, -1)
    return out[:n], rounds, acc


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("err", [0.0, 0.3, 0.9])
def test_spec_equals_plain_greedy(k, err):
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    for seed in range(5):
        ref = plain_decode(prompt, 60)
        got, rounds, acc = spec_decode(prompt, 60, k, err, seed)
        assert got == ref, (k, err, seed)
    if err == 0.0:
        assert rounds <= 60 // (k + 1) + 2  # perfect drafts: k+1 tokens per round


def test_page_boundary_rounds_equal_plain():
    prompt = [t % VOCAB for t in range(PAGE - 3)]
    assert spec_decode(prompt, 40, 3, 0.3, 1)[0] == plain_decode(prompt, 40)


# ---------------------------------------------------------------- policy
def test_expected_tokens():
    assert policy.expected_tokens([0.5, 0.5, 0.5], 0) == 1.0
    assert policy.expected_tokens([0.5, 0.5, 0.5], 3) == pytest.approx(1 + 0.5 + 0.25 + 0.125)
    assert policy.expected_tokens([1, 1, 1], 3) == 4.0


def test_observe_estimates_conditional_rates():
    a = policy.AdaptiveK(ks=(0, 1, 3), ema=0.0)  # no memory: p = last round's batch frequency
    a.observe(
        3, [3, 1, 0, 0]
    )  # position 1 tried by 4 users, ok for 2; position 2 tried by 2 (m>=1), ok for 1; position 3 tried by 1 (m>=2), ok for 1
    assert a.p == pytest.approx([2 / 4, 1 / 2, 1.0])


def test_choose_prefers_plain_when_acceptance_low_and_spec_when_high():
    t = {0: 43.8, 1: 50.9, 3: 66.2}
    lo = policy.AdaptiveK(ks=(0, 1, 3), p=[0.1, 0.1, 0.1], round_ms=dict(t), probe_every=0)
    hi = policy.AdaptiveK(ks=(0, 1, 3), p=[0.9, 0.9, 0.9], round_ms=dict(t), probe_every=0)
    assert lo.choose() == 0 and hi.choose() == 3
    mid = policy.AdaptiveK(ks=(0, 1, 3), p=[0.6, 0.3, 0.3], round_ms=dict(t), probe_every=0)
    assert mid.choose() == 1


def test_probe_refreshes_deep_positions():
    a = policy.AdaptiveK(ks=(0, 3), p=[0.1] * 3, round_ms={0: 43.0, 3: 66.0}, probe_every=4)
    picks = [a.choose() for _ in range(9)]
    assert picks.count(3) >= 2 and picks.count(0) >= 5


# ---------------------------------------------------------------- speedup model
def test_speedup_model_sanity():
    assert speedup.plain_ms() == pytest.approx(43.0, abs=0.3)
    for k in (1, 2, 3):
        assert speedup.verify_ms(k) > speedup.plain_ms()  # a bigger verify step always costs more than a plain step
        assert speedup.speedup(k, 0.95) > speedup.speedup(k, 0.5)
    assert speedup.speedup(3, 0.9) > 1.8  # DSV4 shipped 1.81x at B=4
    assert 0.1 < speedup.breakeven_alpha(3) < 0.5
    assert speedup.moe_us(4) == 363 and speedup.moe_us(16) == 584 and speedup.moe_us(12) == pytest.approx(507)
