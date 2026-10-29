# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/spec_fused_sampled.py (no device). The float32 device-algorithm mirror equals the plugin's
accept_sampled_support on the same candidates and the same uniforms.

How the SAME uniforms reach both walks: both start from one torch.manual_seed. The plugin draws them itself from the
global stream (and from per-request generators for seeded users). The reference gets predraw_uniforms() run from the same
seed, which consumes the global and the generator streams in the plugin's order and returns them rounded to float32 (the
plugin uses the float64 values; the two differ by < 6e-8, so decisions differ only inside the 1e-6 boundary classes, which
are counted and reported). The test also asserts that the plugin and predraw leave the global and generator RNG states equal.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_fused_sampled_host.py -s"""

import math

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_engine import exact_topk, merge_shard_topk
from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host, fused_greedy_reference
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    fused_sampled_reference,
    predraw_uniforms,
    union_topk_stable,
)

try:
    import vllm_tt_plugin.spec_accept as SA
    from vllm_tt_plugin.spec_accept import SpecSamplingInputs, accept_sampled_support
except ImportError:  # pragma: no cover
    pytest.skip("vllm_tt_plugin not importable", allow_module_level=True)

K, T, TP, M, PER_SHARD, VOCAB = 3, 4, 8, 64, 31040, 248320
SCRATCH, NB, CAP = 5000, 6, 4096
EPS = 1e-6
N_CASES = 10000
STATS = {}


def _order_shards(vals, ids):
    """Per shard block: (value desc, local id asc). vals/ids [R, TP, M]."""
    by_id = ids.argsort(dim=2)
    ids, vals = ids.gather(2, by_id), vals.gather(2, by_id)
    by_val = vals.float().sort(dim=2, descending=True, stable=True).indices
    return vals.gather(2, by_val), ids.gather(2, by_val)


def _merged_positions(vals, ids):
    """Union positions of the merged order (value desc, gid asc) for ONE row; vals/ids [TP, M]. Independent of the code
    under test: plain sorts on the flattened union."""
    v, i = vals.reshape(-1).float(), ids.reshape(-1).to(torch.int64) + (torch.arange(TP * M) // M) * PER_SHARD
    o = i.argsort()
    o = o[v[o].sort(descending=True, stable=True).indices]
    return o[:M]


def _union(g, R, kk):
    """Random bf16 per-shard top-M blocks with natural bf16 ties plus injected ties (at the k-th value, through the 64th
    candidate, in the tail). kk [R] the row's top_k. Returns vals bf16 [R, TP*M], local ids int64 [R, TP*M]."""
    scale = torch.rand(R, 1, 1, generator=g) * 3.8 + 0.2
    base = torch.randn(R, TP, 1, generator=g) * 0.5 + 8
    raw = torch.randn(R, TP, 256, generator=g) * scale + base
    vals = raw.topk(M, dim=2).values
    coarse = torch.rand(R, 1, 1, generator=g) < 0.4
    step = torch.where(torch.rand(R, 1, 1, generator=g) < 0.5, 0.25, 0.5)
    vals = torch.where(coarse, torch.round(vals / step) * step, vals).to(torch.bfloat16)
    ids = torch.arange(M)[None, None, :] * 485 + torch.randint(0, 485, (R, TP, M), generator=g)
    ids = ids.gather(2, torch.rand(R, TP, M, generator=g).argsort(dim=2))
    vals, ids = _order_shards(vals, ids)
    for r in range(R):
        if float(torch.rand((), generator=g)) >= 0.45:
            continue
        mode = int(torch.randint(0, 3, (), generator=g))
        k = int(kk[r])
        pos = _merged_positions(vals[r], ids[r])
        if mode == 0:  # tie at the k-th value, short of the 64th candidate
            lo, hi = k, min(k + int(torch.randint(0, 4, (), generator=g)), M - 2)
        elif mode == 1:  # tie at the k-th reaching the 64th candidate: overflow
            lo, hi = k - 1, M - 1
        else:  # tie inside the tail
            lo = int(torch.randint(max(k, 30), M, (), generator=g))
            hi = M - 1
        if hi < lo:
            continue
        flat = vals[r].reshape(-1)
        flat[pos[lo : hi + 1]] = flat[pos[max(k - 1, 0) if mode != 2 else lo]].clone()
        vals[r] = flat.reshape(TP, M)
        vals[r], ids[r] = (x[0] for x in _order_shards(vals[r : r + 1], ids[r : r + 1]))
    return vals.reshape(R, TP * M), ids.reshape(R, TP * M)


def _drafts(g, cand_ids, B, nv):
    """tok_buf [B*T] and drafts [B, K]: drafts that are candidates (top ranks favoured), not candidates, or padding."""
    drafts = torch.randint(0, VOCAB, (B, K), generator=g)
    ids = cand_ids.reshape(B, T, M)
    for u in range(B):
        for j in range(K):
            c = float(torch.rand((), generator=g))
            if c < 0.5:
                drafts[u, j] = ids[u, j, 0]
            elif c < 0.85:
                drafts[u, j] = ids[u, j, int(torch.randint(0, M, (), generator=g))]
    tok = torch.cat([torch.randint(0, VOCAB, (B, 1), generator=g), drafts], dim=1)
    return tok.reshape(-1), drafts


def _case(g, B):
    R = B * T
    greedy = torch.rand(B, generator=g) < 0.2
    inactive = torch.rand(B, generator=g) < 0.1
    nv = torch.randint(0, K + 1, (B,), generator=g)
    nv = torch.where(torch.rand(B, generator=g) < 0.5, torch.full_like(nv, K), nv)
    nv[inactive] = 0
    temp = torch.rand(B, generator=g) * 1.2 + 0.3
    top_k = torch.where(
        torch.rand(B, generator=g) < 0.5,
        torch.randint(1, 16, (B,), generator=g),
        torch.randint(1, 64, (B,), generator=g),
    )
    top_p = torch.tensor([1.0, 0.95, 0.8, 0.5])[torch.randint(0, 4, (B,), generator=g)]
    uv, ul = _union(g, R, top_k.repeat_interleave(T))
    mv, mi = merge_shard_topk(uv, ul, PER_SHARD, M)  # the plugin candidates, as the host merge builds them
    tok_buf, drafts = _drafts(g, mi, B, nv)
    skip = (nv == 0) & ~inactive & (torch.rand(B, generator=g) < 0.5)
    start0 = torch.randint(1, 60, (B,), generator=g)
    aux = fused_accept_aux_host(nv, start0, ~inactive, skip, CAP, K, SCRATCH, 0, 4096)
    pt = torch.randint(1, 4000, (B, NB), generator=g, dtype=torch.int32)
    seeded = (torch.rand(B, generator=g) < 0.25) & ~greedy & ~inactive & (nv > 0)
    return dict(
        B=B,
        uv=uv,
        ul=ul,
        mv=mv,
        mi=mi,
        tok_buf=tok_buf,
        drafts=drafts,
        nv=nv,
        aux=aux,
        pt=pt,
        greedy=greedy | inactive,
        temp=temp,
        top_k=top_k,
        top_p=top_p,
        seeded=seeded,
    )


def _plugin(c, seed, generators):
    """accept_sampled_support on the case. _support_cut is wrapped: the overflow it finds is recorded and cleared (the
    plugin would need full logits for it; the comparison skips those users)."""
    B = c["B"]
    rec = []
    real_cut = SA._support_cut

    def cut(vals, idx, temp, k, p, can_overflow):
        v, i, ovf = real_cut(vals, idx, temp, k, p, can_overflow)
        rec.append(ovf.clone())
        return v, i, torch.zeros_like(ovf)

    SA._support_cut = cut
    try:
        torch.manual_seed(seed)
        sampling = SpecSamplingInputs(
            vocab_size=VOCAB,
            temperature=torch.where(c["greedy"], torch.zeros(B), c["temp"]),
            top_k=c["top_k"].to(torch.int64),
            top_p=c["top_p"],
            generators=generators,
        )
        res = accept_sampled_support(
            c["mi"].reshape(B, T, M)[:, :, 0].to(torch.int32),
            c["mv"].reshape(B, T, M),
            c["mi"].reshape(B, T, M),
            c["drafts"].to(torch.int32),
            c["nv"].to(torch.int32),
            sampling,
        )
    finally:
        SA._support_cut = real_cut
    return res, (rec[0].reshape(B, T) if rec else torch.zeros(B, T, dtype=torch.bool))


def _gens(c, seed):
    out = {}
    for u in c["seeded"].nonzero().reshape(-1).tolist():
        out[u] = torch.Generator().manual_seed(seed * 100 + u)
    return out


def _boundary(c, ref, uni, m_cols):
    """Per user: is a decision within EPS of its threshold on a row either walk examined? (accept compare, inverse-CDF
    bucket edge, top-p tail edge)."""
    B = c["B"]
    out = torch.zeros(B, dtype=torch.bool)
    prob = ref.prob.double().reshape(B, T, M)
    cum = ref.cum.double().reshape(B, T, M)
    gid = ref.cand_ids.reshape(B, T, M)
    for u in range(B):
        if bool(c["greedy"][u]):
            continue
        nv = int(c["nv"][u])
        for j in range(min(max(m_cols[u]), nv - 1) + 1) if nv else []:
            if abs(float(ref.p_d[u, j]) - float(uni[u, 1 + j])) < EPS:
                out[u] = True
        for m in set(m_cols[u]):
            corr = m < nv
            w = prob[u, m].clone()
            if corr:
                w[gid[u, m] == c["drafts"][u, m]] = 0
            cdf = w.cumsum(0) / w.sum()
            uu = float(uni[u, K + 1] if corr else uni[u, 0])
            if float((cdf - uu).abs().min()) < EPS:
                out[u] = True
        p = float(c["top_p"][u])
        if p < 1.0:
            for m in range(max(m_cols[u]) + 1):
                x = cum[u, m][cum[u, m] > 0]
                if x.numel() and float((x - (1 - p)).abs().min()) < EPS:
                    out[u] = True
    return out


def _run_case(c, seed):
    B = c["B"]
    gens_a = _gens(c, seed)
    res, plug_ovf = _plugin(c, seed, gens_a)
    state_a = torch.get_rng_state()
    torch.manual_seed(seed)
    gens_b = _gens(c, seed)
    any_sampled = bool((~c["greedy"]).any())
    uni = predraw_uniforms(gens_b, B, K, c["nv"], c["seeded"], draw_global=any_sampled)
    gens_ok = all(torch.equal(gens_a[r].get_state(), gens_b[r].get_state()) for r in gens_a)
    return res, plug_ovf, uni, state_a, gens_ok


def test_union_stable_topk_equals_merge_shard_topk_ordering():
    g = torch.Generator().manual_seed(1)
    bad = 0
    for _ in range(40):
        kk = torch.randint(1, 64, (32,), generator=g)
        uv, ul = _union(g, 32, kk)
        v1, i1 = union_topk_stable(uv, ul, PER_SHARD, M)
        v2, i2 = merge_shard_topk(uv, ul, PER_SHARD, M)
        bad += int((v1 != v2).any()) + int((i1 != i2.to(torch.int64)).any())
        # and the generic exact_topk over a dense-ish list agrees too
        v3, i3 = exact_topk(uv, M, ul + (torch.arange(TP * M) // M) * PER_SHARD)
        bad += int((v1 != v3).any()) + int((i1 != i3.to(torch.int64)).any())
    assert bad == 0


def test_predraw_error_for_seeded_row_without_drafts(expect_error):
    with expect_error(ValueError, "carry no drafts"):
        predraw_uniforms({1: torch.Generator().manual_seed(1)}, 4, K, torch.tensor([3, 0, 3, 3]))


def test_random_blocks_match_plugin():
    g = torch.Generator().manual_seed(20260420)
    n = n_cmp = n_exact = n_bound = n_bound_equal = n_ovf = n_users = n_state_bad = n_ovf_flag_bad = 0
    n_sampled_users = n_counts_hist = 0
    hist = torch.zeros(T + 1, dtype=torch.long)
    for case in range(N_CASES):
        B = (1, 4, 8)[case % 3]
        c = _case(g, B)
        seed = 1000 + case
        res, plug_ovf, uni, state_a, gens_ok = _run_case(c, seed)
        # RNG streams: plugin and predraw consume the same numbers
        if not torch.equal(state_a, torch.get_rng_state()) or not gens_ok:
            n_state_bad += 1
        ref = fused_sampled_reference(
            c["uv"],
            c["ul"],
            c["tok_buf"],
            c["mi"].reshape(B, T, M)[:, :, 0].reshape(-1),
            c["aux"],
            c["pt"],
            B=B,
            K=K,
            per_shard=PER_SHARD,
            M=M,
            greedy=c["greedy"],
            temperature=c["temp"],
            top_k=c["top_k"],
            top_p=c["top_p"],
            uniforms=uni,
        )
        n += 1
        pc, pcnt = res.committed_token_ids, res.accepted_counts
        cols = torch.arange(T)[None, :]
        ref_m = (ref.counts - 1).tolist()
        # overflow flag vs the plugin's own overflow at the rows up to the reference's m
        want_ovf = ((plug_ovf & (cols <= (ref.counts - 1)[:, None])).any(dim=1)) & ~c["greedy"]
        n_ovf_flag_bad += int((want_ovf.to(torch.int32) != ref.overflow).sum())
        bound = _boundary(c, ref, uni, [[a, int(b) - 1] for a, b in zip(ref_m, pcnt.tolist())])
        for u in range(B):
            n_users += 1
            n_sampled_users += int(not bool(c["greedy"][u]))
            if ref.overflow[u]:
                n_ovf += 1
                continue
            n_cmp += 1
            same = int(ref.counts[u]) == int(pcnt[u]) and torch.equal(ref.committed[u], pc[u])
            if bound[u]:
                n_bound += 1
                n_bound_equal += int(same)
            else:
                assert same, (case, u, ref.committed[u], pc[u], ref.counts[u], pcnt[u])
                n_exact += 1
            hist[int(ref.counts[u])] += 1
        # drafter part == S1 reference on the same committed prefix
        committed, counts = ref.committed.to(torch.int64), ref.counts.to(torch.int64)
        ids_s = torch.where(cols < counts[:, None], committed, torch.zeros_like(committed))
        toks_s = torch.zeros(B, T, dtype=torch.int64)
        for u in range(B):
            cu, nvu = int(counts[u]), int(c["nv"][u])
            toks_s[u, 1:cu] = committed[u, : cu - 1]
            if cu - 1 < nvu:
                toks_s[u, cu] = committed[u, cu - 1] + 1
        s1 = fused_greedy_reference(ids_s.reshape(-1), toks_s.reshape(-1), c["aux"], c["pt"], B=B, K=K)
        assert (
            torch.equal(s1.pack, ref.pack) and torch.equal(s1.rs_pt, ref.rs_pt) and torch.equal(s1.mtp_pt, ref.mtp_pt)
        )
        assert torch.equal(s1.counts, ref.counts) and torch.equal(s1.committed, ref.committed)
        assert torch.equal(s1.num_valid_out, ref.num_valid_out) and s1.drafter_runs == ref.drafter_runs
    print(
        f"\n[random] cases={n} users={n_users} sampled_users={n_sampled_users} overflow_users={n_ovf} "
        f"({100 * n_ovf / n_users:.2f}% of users) compared={n_cmp} exact_nonboundary={n_exact} "
        f"boundary={n_bound} (equal anyway {n_bound_equal}) rng_state_mismatch_cases={n_state_bad} "
        f"overflow_flag_mismatch_users={n_ovf_flag_bad} count_hist(1..4)={hist[1:].tolist()}"
    )
    assert n_cmp > 0.6 * n_users and n_state_bad == 0 and n_ovf_flag_bad == 0
    assert n_exact + n_bound == n_cmp


# ------------------------------------------------------------------------------------------------ chi-square
def _target(vals, gids, temp, k, p):
    """Independent float64 target distribution over the candidates of one row: dict token -> prob."""
    v = vals.double() / temp
    keep = v >= v[k - 1]
    e = torch.where(keep, torch.exp(v - v[0]), torch.zeros_like(v))
    pr = e / e.sum()
    if p < 1.0:
        tail = pr.flip(0).cumsum(0).flip(0)
        drop = tail <= 1 - p
        drop[0] = False
        pr = torch.where(drop, torch.zeros_like(pr), pr)
        pr = pr / pr.sum()
    return {int(t): float(x) for t, x in zip(gids, pr) if x > 0}


def _expected_joint(tgt, drafts, nv):
    """P(count, first committed token) for a point-mass drafter (exact lossless walk)."""
    out = {}
    if nv == 0:
        return {(1, t): q for t, q in tgt[0].items()}
    a = [tgt[j].get(drafts[j], 0.0) for j in range(nv)]
    for t, q in tgt[0].items():
        if t != drafts[0]:
            out[(1, t)] = q
    run = a[0]
    for c in range(2, nv + 2):  # count c: accepted columns 0..c-2, then reject at c-1 (or bonus if c-1 == nv)
        j = c - 1
        pc = run * ((1 - a[j]) if j < nv else 1.0)
        out[(c, drafts[0])] = out.get((c, drafts[0]), 0.0) + pc
        if j < nv:
            run *= a[j]
    return out


def _chi_crit(df, z=3.09):  # p = 0.999, Wilson-Hilferty
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


@pytest.mark.parametrize(
    "temp,k,p,nv,ranks",
    [
        (1.0, 20, 1.0, 3, [0, 0, 1]),
        (0.7, 40, 0.8, 3, [0, 3, 0]),
        (1.2, 10, 0.5, 2, [0, -1, 0]),
        (0.9, 30, 0.95, 0, [0, 0, 0]),
    ],
)
def test_joint_count_first_token_matches_target(temp, k, p, nv, ranks):
    g = torch.Generator().manual_seed(7 + k)
    while True:  # a block with no overflow
        uv, ul = _union(g, T, torch.full((T,), k))
        mv, mi = merge_shard_topk(uv, ul, PER_SHARD, M)
        if not bool((mv[:, M - 1] >= mv[:, k - 1]).any()):
            break
    drafts = [int(mi[j, r]) if r >= 0 else 12345 for j, r in enumerate(ranks)]
    tgt = [_target(mv[j], mi[j], temp, k, p) for j in range(T)]
    exp = _expected_joint(tgt, drafts, nv)
    N = 30000
    aux = fused_accept_aux_host(
        torch.full((N,), nv),
        torch.full((N,), 5),
        torch.ones(N, dtype=torch.bool),
        torch.zeros(N, dtype=torch.bool),
        CAP,
        K,
        SCRATCH,
        0,
        4096,
    )
    tok = torch.cat([torch.zeros(1, dtype=torch.int64), torch.tensor(drafts)])
    gen = torch.Generator().manual_seed(99)
    ref = fused_sampled_reference(
        uv.repeat(N, 1),
        ul.repeat(N, 1),
        tok.repeat(N),
        mi[:, 0].repeat(N),
        aux,
        torch.ones(N, NB, dtype=torch.int32),
        B=N,
        K=K,
        per_shard=PER_SHARD,
        M=M,
        greedy=torch.zeros(N, dtype=torch.bool),
        temperature=torch.full((N,), temp),
        top_k=torch.full((N,), k),
        top_p=torch.full((N,), p),
        uniforms=torch.rand((N, K + 2), generator=gen),
    )
    assert int(ref.overflow.sum()) == 0
    obs = {}
    for cnt, t in zip(ref.counts.tolist(), ref.committed[:, 0].tolist()):
        obs[(cnt, t)] = obs.get((cnt, t), 0) + 1

    def stat(observed):
        keys = set(exp) | set(observed)
        big = [kk for kk in keys if N * exp.get(kk, 0.0) >= 5]
        s = sum((observed.get(kk, 0) - N * exp[kk]) ** 2 / (N * exp[kk]) for kk in big)
        rest_e = N * (1 - sum(exp[kk] for kk in big))
        rest_o = N - sum(observed.get(kk, 0) for kk in big)
        df = len(big) - 1
        if rest_e >= 5:
            s += (rest_o - rest_e) ** 2 / rest_e
            df += 1
        else:
            assert rest_o <= 10 + 5 * rest_e, (rest_o, rest_e)  # a token the target never produces must not appear
        return s, df

    s, df = stat(obs)
    print(f"\n[chi2] T={temp} k={k} p={p} nv={nv}: stat={s:.1f} df={df} crit={_chi_crit(df):.1f} cells={len(exp)}")
    assert all(kk in exp for kk in obs if obs[kk] > 0), "committed (count, token) outside the target support"
    assert s < _chi_crit(df)
    # power: move 3% of the mass from the most frequent cell to the rarest cell above the pooling threshold
    biased = dict(obs)
    moved = int(0.03 * N)
    donor = max(obs, key=obs.get)
    recv = min((kk for kk in exp if N * exp[kk] >= 5 and kk != donor), key=exp.get)
    biased[donor] -= moved
    biased[recv] = biased.get(recv, 0) + moved
    s2, _ = stat(biased)
    assert s2 > _chi_crit(df)


# ------------------------------------------------------------------------------------------------ stage S2 additions
def test_predraw_equals_the_plugin_copy():
    """The plugin carries its own copy of predraw_uniforms (it must not import tt-metal): same table, same RNG states after."""
    from vllm_tt_plugin.spec_accept import predraw_uniforms as plugin_predraw

    g = torch.Generator().manual_seed(11)
    for case in range(200):
        B = (1, 4, 8)[case % 3]
        nv = torch.randint(1, K + 1, (B,), generator=g)
        seeded = [u for u in range(B) if int(torch.randint(0, 2, (1,), generator=g))]
        a = {u: torch.Generator().manual_seed(500 + case * 10 + u) for u in seeded}
        b = {u: torch.Generator().manual_seed(500 + case * 10 + u) for u in seeded}
        torch.manual_seed(case)
        x = predraw_uniforms(a, B, K, nv)
        sx = torch.get_rng_state()
        torch.manual_seed(case)
        y = plugin_predraw(b, B, K, nv)
        assert torch.equal(x, y) and torch.equal(sx, torch.get_rng_state())
        assert all(torch.equal(a[u].get_state(), b[u].get_state()) for u in seeded)


def test_host_redo_walk_equals_the_plugin_full_logits_walk():
    """host_redo_walk (float64, the overflow redo) commits what the plugin's full-logits walk commits with the same uniforms."""
    from vllm_tt_plugin.spec_accept import accept_sampled_drafts

    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import host_redo_walk

    V, Bs = 512, 4
    g = torch.Generator().manual_seed(77)
    bad = n_cmp = 0
    for case in range(300):
        B = Bs
        logits = (torch.randn(B, T, V, generator=g) * 2.5).to(torch.bfloat16).float()
        for u in range(B):  # tie groups at the top-k boundary
            for j in range(T):
                top = torch.topk(logits[u, j], 12).indices
                logits[u, j, top[3:6]] = logits[u, j, top[3]].clone()
        temp = torch.tensor([0.0, 0.7, 1.0, 1.4])[torch.randperm(4, generator=g)]
        top_k = torch.randint(1, 20, (B,), generator=g).to(torch.int32)
        top_p = torch.tensor([1.0, 0.9, 0.5, 0.97])[torch.randperm(4, generator=g)]
        nv = torch.randint(0, K + 1, (B,), generator=g).to(torch.int32)
        drafts = (
            torch.topk(logits[:, :K], 3, dim=-1)
            .indices.gather(2, torch.randint(0, 3, (B, K, 1), generator=g))
            .squeeze(2)
            .to(torch.int32)
        )
        seeded = [u for u in range(B) if int(nv[u]) > 0 and case % 2 == 0 and u % 2 == 0]
        a = {u: torch.Generator().manual_seed(9000 + case * 10 + u) for u in seeded}
        b = {u: torch.Generator().manual_seed(9000 + case * 10 + u) for u in seeded}
        sampling = SpecSamplingInputs(
            vocab_size=V, temperature=temp, top_k=top_k.to(torch.int64), top_p=top_p, generators=a
        )
        torch.manual_seed(case)
        plug = accept_sampled_drafts(logits, drafts, nv, sampling)
        torch.manual_seed(case)
        uni = predraw_uniforms(b, B, K, nv)
        tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), drafts.to(torch.int64)], dim=1)
        committed, counts = host_redo_walk(
            logits.reshape(B * T, V),
            tok,
            {"temperature": temp, "top_k": top_k, "top_p": top_p, "greedy": temp <= 0},
            uni,
            B=B,
            K=K,
            num_valid=nv,
        )
        n_cmp += 1
        same = torch.equal(plug.accepted_counts, counts) and all(
            torch.equal(plug.committed_token_ids[u, : int(counts[u])], committed[u, : int(counts[u])]) for u in range(B)
        )
        bad += int(not same)
    print(f"\n[redo] cases={n_cmp} mismatching={bad}")
    assert bad <= 2, "only float32 uniform rounding boundaries (|p - u| < 1e-7) may differ"


def test_certificate_reference_equals_the_host_redo_for_passing_rows(expect_error):
    """H5 (phase 2): fused_sampled_reference(full_norm=) with certificate rows (top_k = 0, top_p < 1) commits what
    host_redo_walk (float64, whole vocabulary) commits with the same uniforms, except float32 boundary classes
    (audit_boundary_users). Overflow bit 2 is set exactly when an examined row failed q >= p + eps, bit 1 never for a
    certificate row; a reference without full_norm still rejects top_k = 0."""
    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
        CERT_EPS,
        OVF_CERT,
        OVF_TIE,
        audit_boundary_users,
        host_redo_walk,
    )
    from models.demos.blackhole.qwen36.tt.spec_logprobs import stats_from_logits

    S, per, Bc = 8, 512, 4
    V = S * per
    R = Bc * T
    g = torch.Generator().manual_seed(5150)
    n_users = n_pass = n_fail = n_diff = n_bound = 0
    for case in range(300):
        B = Bc
        scale = torch.rand(R, 1, generator=g) * 4 + 1.5
        x = (torch.randn(R, V, generator=g) * scale).to(torch.bfloat16).float()
        for r in range(R):  # a few strong tokens so that many rows pass
            if float(torch.rand((), generator=g)) < 0.6:
                x[r, torch.randint(0, V, (4,), generator=g)] += torch.tensor([9.0, 8.0, 7.0, 6.0])
        x = x.to(torch.bfloat16).float()
        greedy = torch.tensor([False, False, False, False])
        greedy[int(torch.randint(0, 4, (), generator=g))] = bool(case % 3 == 0)
        temp = torch.rand(B, generator=g) * 1.2 + 0.3
        top_k = torch.tensor([0, 0, 0, 0])
        top_k[3] = int(torch.randint(1, 64, (), generator=g)) if case % 2 == 0 else 0  # one ordinary top-k row
        top_p = torch.tensor([0.5, 0.8, 0.9, 0.95, 0.99])[torch.randint(0, 5, (B,), generator=g)]
        nv = torch.randint(0, K + 1, (B,), generator=g)
        # union exactly as the topk trace
        vs, ls = [], []
        for s in range(S):
            sh = x[:, s * per : (s + 1) * per]
            o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :M]
            ls.append(o)
            vs.append(sh.gather(1, o))
        uv, ul = torch.cat(vs, 1).to(torch.bfloat16), torch.cat(ls, 1)
        order = torch.sort(x, dim=1, descending=True, stable=True).indices
        drafts = order.reshape(B, T, V)[:, :K, :6].gather(2, torch.randint(0, 6, (B, K, 1), generator=g)).squeeze(2)
        tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), drafts], dim=1)
        aux = fused_accept_aux_host(
            nv,
            torch.randint(1, 60, (B,), generator=g),
            torch.ones(B, dtype=torch.bool),
            torch.zeros(B, dtype=torch.bool),
            CAP,
            K,
            SCRATCH,
            0,
            4096,
        )
        pt = torch.randint(1, 4000, (B, NB), generator=g, dtype=torch.int32)
        torch.manual_seed(case)
        uni = predraw_uniforms({}, B, K, nv)
        trow = torch.where(greedy | (temp <= 0), torch.ones(B), temp).repeat_interleave(T)
        _, _, l_t = stats_from_logits(x, trow)
        kw = dict(B=B, K=K, per_shard=per, M=M, greedy=greedy, temperature=temp, top_k=top_k, top_p=top_p, uniforms=uni)
        with expect_error(ValueError, "top_k outside"):
            if bool((~greedy & (top_k == 0)).any()):
                fused_sampled_reference(uv, ul, tok.reshape(-1), order[:, 0], aux, pt, **kw)
            else:
                raise ValueError("top_k outside (no certificate row in this case)")
        ref = fused_sampled_reference(
            uv, ul, tok.reshape(-1), order[:, 0], aux, pt, full_norm=l_t.to(torch.float32), **kw
        )
        redo_c, redo_n = host_redo_walk(
            x, tok, {"temperature": temp, "top_k": top_k, "top_p": top_p, "greedy": greedy}, uni, B=B, K=K, num_valid=nv
        )
        cert = ~greedy & (top_k == 0)
        examined = torch.arange(T)[None, :] <= (ref.counts - 1)[:, None]
        q = ref.cert_q.reshape(B, T)
        fail_rows = (q - top_p[:, None] < CERT_EPS) & cert[:, None] & (q < 1.0 + 1e-9)
        want_bit = ((fail_rows & examined).any(1) & cert).to(torch.int32) * OVF_CERT
        assert torch.equal(ref.overflow & OVF_CERT, want_bit), (case, ref.overflow, want_bit)
        assert not bool(((ref.overflow & OVF_TIE) != 0)[cert].any())
        bound = audit_boundary_users(ref, uni, redo_c, redo_n, tok[:, 1:], nv, B=B, K=K, M=M, top_p=top_p)
        for u in range(B):
            n_users += 1
            if int(ref.overflow[u]) != 0:
                n_fail += int(bool(cert[u]))
                continue
            n_pass += 1
            same = int(ref.counts[u]) == int(redo_n[u]) and torch.equal(
                ref.committed[u, : int(redo_n[u])], redo_c[u, : int(redo_n[u])]
            )
            if not same:
                n_diff += 1
                assert u in bound, (case, u, ref.committed[u], redo_c[u], ref.counts[u], redo_n[u])
                n_bound += 1
    print(
        f"\n[cert ref] users={n_users} pass={n_pass} cert_fail_users={n_fail} differing={n_diff} explained_boundary={n_bound}"
    )
    assert n_pass > 0.3 * n_users and n_fail > 0 and n_diff <= 0.02 * n_pass
