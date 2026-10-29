# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of tt/spec_logprobs.py and of the phase 2 additions to tt/spec_fused_sampled.py
(work/research_1026/67 plan, tests H1-H4, H6 and the S1.3 uniform rule; H5 is in test_spec_fused_sampled_host.py).

Run: pytest -o addopts="" -q models/demos/blackhole/qwen36/tests/test_spec_logprobs_host.py"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    CERT_EPS,
    _row_distribution,
    _row_distribution_full,
    certificate_cut,
    host_redo_walk,
    merge_redo,
    predraw_uniforms,
)
from models.demos.blackhole.qwen36.tt.spec_logprobs import (
    candidates_from_device,
    candidates_from_logits,
    candidates_from_union,
    combine_shard_sums,
    gather_logprobs_from_candidates,
    lse_from_stats,
    make_candidates,
    permute_candidates,
    stats_from_logits,
)

M = 64


def _logits(g, R, V, scale=3.0, coarse=0.0, peaked=False):
    x = torch.randn(R, V, generator=g) * scale
    if coarse:  # bf16-like ties: round to a coarse grid
        x = torch.round(x / coarse) * coarse
    if peaked:
        for r in range(R):
            idx = torch.randint(0, V, (5,), generator=g)
            x[r, idx] += torch.tensor([12.0, 10.0, 9.0, 8.0, 7.0])
    return x.to(torch.bfloat16).float()


def _shard_union(x, S):
    """Device topk-trace layout from full logits [R, V]: per shard stable top-64 (value desc, local id asc)."""
    R, V = x.shape
    per = V // S
    vals, ids = [], []
    for s in range(S):
        sh = x[:, s * per : (s + 1) * per]
        o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :M]
        ids.append(o)
        vals.append(sh.gather(1, o))
    return torch.cat(vals, 1).to(torch.bfloat16), torch.cat(ids, 1), per


# ------------------------------------------------------------------ H1
@pytest.mark.parametrize("S", [1, 2, 3, 5, 8])
def test_h1_candidates_from_union_equal_the_global_topk_with_ties(S):
    g = torch.Generator().manual_seed(100 + S)
    per = 200
    for coarse in (0.0, 0.25, 1.0):
        x = _logits(g, 12, S * per, scale=2.0, coarse=coarse)
        uv, ul, per_shard = _shard_union(x, S)
        v, i = candidates_from_union(uv, ul, per_shard, M)
        want_o = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :M]
        assert torch.equal(i, want_o), (S, coarse)  # ids and order, ties by lowest id
        assert torch.equal(v, x.gather(1, want_o))


def test_h1_fallback_payload_from_logits_equals_the_union_payload():
    g = torch.Generator().manual_seed(7)
    x = _logits(g, 8, 8 * 200, coarse=0.5)
    uv, ul, per = _shard_union(x, 8)
    stats_m, l_raw, _ = stats_from_logits(x, torch.ones(8))
    a = candidates_from_device(uv, ul, stats_m, l_raw, per, 2, 4)
    b = candidates_from_logits(x, 2, 4)
    assert torch.equal(a.ids, b.ids) and torch.equal(a.values, b.values)
    assert torch.allclose(a.lse, b.lse, atol=1e-6)
    assert a.ids.dtype == torch.int32 and a.values.dtype == a.lse.dtype == torch.float32


# ------------------------------------------------------------------ H2
@pytest.mark.parametrize("T", [0.3, 0.6, 1.0, 1.5])
def test_h2_lse_and_scaled_sum_combine_in_fp32(T):
    g = torch.Generator().manual_seed(int(T * 100))
    S, per = 8, 31040
    for peaked in (False, True):
        x = _logits(g, 4, S * per, scale=3.0, peaked=peaked)
        m64, lr64, lt64 = stats_from_logits(x, torch.full((4,), T))
        want_lse = torch.logsumexp(x.double(), dim=-1)
        d = x - x.max(dim=1, keepdim=True).values  # relative to the GLOBAL maximum, fp32 per shard
        parts_raw = torch.stack([d[:, s * per : (s + 1) * per].exp().sum(1) for s in range(S)])
        parts_t = torch.stack([(d[:, s * per : (s + 1) * per] / T).exp().sum(1) for s in range(S)])
        lse = lse_from_stats(x.max(dim=1).values, combine_shard_sums(parts_raw))
        assert (lse - want_lse).abs().max() <= 1e-5
        assert ((combine_shard_sums(parts_t).double() - lt64).abs() / lt64).max() <= 1e-5
        assert torch.equal(m64, x.max(dim=1).values.double())


# ------------------------------------------------------------------ H3
@pytest.mark.parametrize("N", [0, 1, 5, 20])
def test_h3_logprobs_rank_and_topn_equal_vllm_sampler(N):
    pytest.importorskip("vllm")
    from vllm.v1.sample.sampler import Sampler

    g = torch.Generator().manual_seed(300 + N)
    V, R = 1500, 24
    for coarse in (0.0, 0.5, 2.0):
        x = _logits(g, R, V, scale=2.5, coarse=coarse)
        cand = candidates_from_logits(x, R, 1)
        order = torch.sort(x, dim=1, descending=True, stable=True).indices
        chosen = order.gather(1, torch.randint(0, 30, (R, 1), generator=g)).reshape(-1)
        lp = Sampler.compute_logprobs(x)
        ref = Sampler.gather_logprobs(lp, N, chosen)
        tok, got_lp, rank, lower, missing = gather_logprobs_from_candidates(
            cand.ids.reshape(R, M), cand.values.reshape(R, M), cand.lse.reshape(R), chosen, N
        )
        assert not bool(missing.any())
        assert torch.allclose(got_lp, ref.logprobs, atol=2e-5, rtol=0), float((got_lp - ref.logprobs).abs().max())
        exact = ~lower
        assert torch.equal(rank[exact], ref.selected_token_ranks.to(torch.int64)[exact])
        assert bool((rank[lower] <= ref.selected_token_ranks.to(torch.int64)[lower]).all())
        assert torch.equal(tok[:, 0], chosen.to(torch.int64))
        # top-N: values equal always, ids equal except inside a tie group (we pick the lowest id)
        if N:
            assert torch.allclose(got_lp[:, 1:], ref.logprobs[:, 1:], atol=2e-5, rtol=0)
            want_ids = order[:, :N]
            assert torch.equal(tok[:, 1:], want_ids.to(torch.int64))
            same = tok[:, 1:] == ref.logprob_token_ids[:, 1:].to(torch.int64)
            tie = torch.stack([(x[r].unsqueeze(0) == x[r, tok[r, 1:]].unsqueeze(1)).sum(1) > 1 for r in range(R)])
            assert bool((same | tie).all())


def test_h3_missing_and_lower_bound_counters(expect_error):
    ids = torch.arange(64).reshape(1, 64)
    vals = torch.full((1, 64), 5.0)  # a 64-way tie: every chosen is a lower bound
    tok, lp, rank, lower, missing = gather_logprobs_from_candidates(
        ids, vals, torch.tensor([6.0]), torch.tensor([10]), 2
    )
    assert bool(lower[0]) and not bool(missing[0]) and int(rank[0]) == 64
    tok, lp, rank, lower, missing = gather_logprobs_from_candidates(
        ids, vals, torch.tensor([6.0]), torch.tensor([999]), 0
    )
    assert bool(missing[0]) and int(rank[0]) == 65 and float(lp[0, 0]) == -1.0
    with expect_error(ValueError, ".*"):
        gather_logprobs_from_candidates(ids, vals, torch.tensor([6.0]), torch.tensor([1]), 65)


def test_candidate_payload_shape_checks_and_permutation(expect_error):
    R, B, T = 12, 3, 4
    vals, ids, lse = torch.randn(R, M), torch.arange(R * M).reshape(R, M), torch.randn(R)
    c = make_candidates(vals, ids, lse, B, T)
    assert c.ids.shape == (B, T, M) and c.lse.shape == (B, T)
    p = permute_candidates(c, torch.tensor([2, 0, 1]))
    assert torch.equal(p.ids[0], c.ids[2]) and torch.equal(p.lse[2], c.lse[1])
    with expect_error(ValueError, ".*"):
        make_candidates(vals[:5], ids[:5], lse[:5], B, T)
    with expect_error(ValueError, ".*"):
        make_candidates(vals, ids[:, :10], lse, B, T)


# ------------------------------------------------------------------ H4
def _cand_distribution(x_row, T, p, M=M):
    """The device-side nucleus on the first M candidates with the certificate cut (float32 as the reference). Returns
    (ids [M], prob [M] fp32, q, fail)."""
    v = x_row.float() / T
    order = torch.sort(v, descending=True, stable=True).indices[:M]
    vs = v[order]
    e = torch.exp(vs - vs[0])
    full = torch.exp(v - vs[0]).sum()
    q, omp, fail = certificate_cut(e.sum(), full, torch.tensor(p))
    cum = (e / e.sum()).flip(0).cumsum(0).flip(0)
    drop = cum <= omp
    drop[0] = False
    e = torch.where(drop, torch.zeros_like(e), e)
    return order, e / e.sum(), float(q), bool(fail), cum, float(omp)


@pytest.mark.parametrize("kind", ["random", "peaked", "flat", "tied"])
def test_h4_certificate_rule_matches_the_full_vocabulary_nucleus(kind):
    g = torch.Generator().manual_seed({"random": 1, "peaked": 2, "flat": 3, "tied": 4}[kind])
    V = 3000
    n_pass = n_fail = n_skip = 0
    for case in range(120):
        x = _logits(
            g,
            1,
            V,
            scale={"random": 2.5, "peaked": 1.5, "flat": 0.4, "tied": 2.0}[kind],
            coarse=0.5 if kind == "tied" else 0.0,
            peaked=kind == "peaked",
        )[0]
        for p in (0.5, 0.8, 0.9, 0.95, 0.99):
            for T in (0.3, 0.6, 1.0, 1.5):
                order, prob, q, fail, cum, omp = _cand_distribution(x, T, p)
                want_ids, want = _row_distribution_full(x, T, 0, p)  # float64, top_k off, whole vocabulary
                if abs(q - (p + CERT_EPS)) < 5e-5 or bool(((cum.double() - omp).abs() < 5e-6).any()):
                    n_skip += 1  # eps zone: float32 vs float64 may differ legitimately
                    continue
                assert fail == (q < p + CERT_EPS), (kind, case, p, T, q)
                if fail:
                    n_fail += 1
                    continue
                n_pass += 1
                assert torch.equal(order, want_ids[:M])
                assert torch.allclose(prob.double(), want[:M], atol=2e-6, rtol=1e-5), float(
                    (prob.double() - want[:M]).abs().max()
                )
                assert float(want[M:].sum()) == 0.0  # the tail beyond the candidates is fully dropped
    assert n_pass + n_fail > 0
    print(f"[H4 {kind}] pass={n_pass} fail={n_fail} eps_zone_skipped={n_skip}")


def test_h4_p_at_or_above_one_never_passes():
    q, omp, fail = certificate_cut(torch.tensor([3.0, 7.0]), torch.tensor([3.0, 7.0]), torch.tensor([1.0, 0.99]))
    assert bool(fail[0]) and not bool(fail[1]) and float(q[0]) == 1.0


@pytest.mark.parametrize("prefix", [64, 200])
def test_h4_fast_row_distribution_equals_the_full_sort(prefix):
    g = torch.Generator().manual_seed(40 + prefix)
    V = 3000
    n_fast = n_full = 0
    for case in range(60):
        kind = case % 4
        x = _logits(g, 1, V, scale=(2.5, 1.0, 0.3, 2.0)[kind], coarse=0.5 if kind == 3 else 0.0, peaked=kind == 0)[0]
        for k in (0, 1, 5, 40, 150, V):
            for p in (1.0, 0.99, 0.9, 0.5):
                for T in (0.6, 1.0):
                    want_ids, want = _row_distribution_full(x, T, k, p)
                    ids, prob = _row_distribution(x, T, k, p, prefix=prefix)
                    n = ids.numel()
                    assert n <= V
                    if n < V:
                        n_fast += 1
                    else:
                        n_full += 1
                    assert torch.equal(ids, want_ids[:n]), (case, k, p, T)
                    assert torch.allclose(prob, want[:n], atol=1e-12, rtol=1e-9), (case, k, p, T)
                    assert float(want[n:].sum()) == 0.0
    assert n_fast > 0 and n_full > 0  # both paths are exercised
    print(f"[H4 fast prefix={prefix}] fast={n_fast} fallback={n_full}")


def test_h4_row_distribution_top_k_zero_means_off():
    x = torch.randn(500)
    a, pa = _row_distribution_full(x, 1.0, 0, 1.0)
    b, pb = _row_distribution_full(x, 1.0, 500, 1.0)
    assert torch.equal(a, b) and torch.equal(pa, pb) and int((pa > 0).sum()) == 500


# ------------------------------------------------------------------ H6
def _redo_case(g, B=4, K=3, V=512):
    T = K + 1
    logits = (torch.randn(B, T, V, generator=g) * 2.5).to(torch.bfloat16).float()
    temp = torch.tensor([0.0, 0.7, 1.0, 1.4])[torch.randperm(4, generator=g)]
    top_k = torch.randint(0, 20, (B,), generator=g).to(torch.int32)  # 0 = off
    top_p = torch.tensor([1.0, 0.9, 0.5, 0.97])[torch.randperm(4, generator=g)]
    nv = torch.randint(0, K + 1, (B,), generator=g).to(torch.int32)
    drafts = (
        torch.topk(logits[:, :K], 3, dim=-1).indices.gather(2, torch.randint(0, 3, (B, K, 1), generator=g)).squeeze(2)
    )
    tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), drafts.to(torch.int64)], dim=1)
    params = {"temperature": temp, "top_k": top_k, "top_p": top_p, "greedy": temp <= 0}
    uni = predraw_uniforms({}, B, K, nv)
    return logits.reshape(B * T, V), tok, params, uni, nv


def test_h6_redo_subset_equals_the_full_redo_and_merge_keeps_the_device_result():
    g = torch.Generator().manual_seed(61)
    B, K, T = 4, 3, 4
    for case in range(40):
        logits, tok, params, uni, nv = _redo_case(g)
        full_c, full_n = host_redo_walk(logits, tok, params, uni, B=B, K=K, num_valid=nv)
        users = [u for u in range(B) if (case + u) % 2 == 0] or [0]
        sub_c, sub_n = host_redo_walk(logits, tok, params, uni, B=B, K=K, num_valid=nv, users=users)
        for u in users:
            assert torch.equal(sub_c[u], full_c[u]) and int(sub_n[u]) == int(full_n[u])
        for u in set(range(B)) - set(users):  # not evaluated
            assert int(sub_n[u]) == 1 and bool((sub_c[u] == -1).all())
        dev_c = torch.randint(0, 100, (B, T), dtype=torch.int32)
        dev_n = torch.randint(1, T + 1, (B,), dtype=torch.int32)
        mc, mn = merge_redo(dev_c, dev_n, sub_c, sub_n, users)
        for u in range(B):
            if u in users:
                assert torch.equal(mc[u], full_c[u]) and int(mn[u]) == int(full_n[u])
            else:
                assert torch.equal(mc[u], dev_c[u]) and int(mn[u]) == int(dev_n[u])
        assert mc.dtype == torch.int32 and mn.dtype == torch.int32
    # empty subset: nothing changes
    c0, n0 = merge_redo(dev_c, dev_n, sub_c, sub_n, [])
    assert torch.equal(c0, dev_c) and torch.equal(n0, dev_n)


# ------------------------------------------------------------------ S1.3
def test_s13_all_greedy_logprobs_step_draws_no_global_uniforms():
    torch.manual_seed(5)
    state = torch.get_rng_state()
    nv = torch.tensor([3, 2, 0, 3])
    u = predraw_uniforms({}, 4, 3, nv, draw_global=False)
    assert torch.equal(state, torch.get_rng_state()), "draw_global=False must not touch the global stream"
    assert u.shape == (4, 5) and float(u.abs().sum()) == 0.0
    u2 = predraw_uniforms({}, 4, 3, nv)  # default: the sampled step still draws (2 + K values per row)
    assert not torch.equal(state, torch.get_rng_state()) and float(u2.abs().sum()) > 0


def test_candidates_from_device_takes_the_max_from_the_union_and_clamps_stale_sums():
    """S4: max_val=None uses the union row maximum (the value the stats trace used); an L_raw below 1 (a zero / stale buffer row of a
    parked row) is clamped to 1, so the lse stays finite and equals max."""
    g = torch.Generator().manual_seed(5)
    R, per, TP = 8, 128, 4
    x = (torch.randn(R, per * TP, generator=g) * 2).to(torch.bfloat16)
    uv, ul = [], []
    for d in range(TP):
        v, i = torch.sort(x[:, d * per : (d + 1) * per].float(), dim=1, descending=True, stable=True)
        uv.append(v[:, :16])
        ul.append(i[:, :16])
    uv, ul = torch.cat(uv, 1).to(torch.bfloat16), torch.cat(ul, 1)
    mx, l_raw, _ = stats_from_logits(x.float(), torch.ones(R))
    a = candidates_from_device(uv, ul, None, l_raw, per, 2, 4, M=16)
    b = candidates_from_device(uv, ul, mx, l_raw, per, 2, 4, M=16)
    assert torch.equal(a.lse, b.lse)
    assert torch.allclose(a.lse.reshape(-1).double(), torch.logsumexp(x.double(), dim=1), atol=1e-5)
    z = candidates_from_device(uv, ul, None, torch.zeros(R), per, 2, 4, M=16)
    assert bool(torch.isfinite(z.lse).all()) and torch.equal(z.lse.reshape(-1), z.values[..., 0].reshape(-1))
