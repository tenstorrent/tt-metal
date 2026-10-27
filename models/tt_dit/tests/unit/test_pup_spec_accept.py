# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only (gate G1 of the pup spec-decode design): the certified top-N acceptance equals ``presence_greedy`` on
the full row, bit for bit, including exact ties, ties across the penalty boundary, all-penalized candidates, p = 0
and NaN rows; the fallback triggers exactly when the candidates cannot certify (``best <= thresh``)."""

import os

import pytest
import torch

from models.tt_dit.pipelines.flux3_klein.pup_spec_accept import (
    accept_block,
    certified_argmax,
    penalized_argmax,
    reference_topn,
)
from models.tt_dit.pipelines.flux3_klein.upsampler import presence_greedy

N_PROPERTY_ROWS = int(os.environ.get("PUP_SPEC_TEST_ROWS", "100000"))
V, N_DEV = 256, 8  # 8 shards of 32, as 248320 / 8 = 31040 on the device


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _random_row(g: torch.Generator, kind: int) -> torch.Tensor:
    if kind == 0:  # continuous, few ties
        return _bf16(torch.randn(V, generator=g) * 3.0)
    if kind == 1:  # half-integers: heavy ties, and x - 1.5 lands exactly on other values (ties across the penalty)
        return torch.randint(-6, 7, (V,), generator=g).float() * 0.5
    if kind == 2:  # very few distinct values: more than top_n ties per shard at the max
        return torch.randint(0, 3, (V,), generator=g).float() * 1.5
    if kind == 3:  # large magnitudes: x - 1.5 rounds back to x in fp32 for |x| >= 2^25
        return _bf16(torch.randn(V, generator=g) * 1e8)
    if kind == 4:  # a constant row (every id ties)
        return torch.full((V,), float(torch.randn(1, generator=g)))
    if kind == 5:  # -inf entries
        row = _bf16(torch.randn(V, generator=g))
        row[torch.rand(V, generator=g) < 0.5] = float("-inf")
        return row
    # kind 6: bf16 values in a narrow range, near ties at the ulp scale
    return _bf16(10.0 + torch.randn(V, generator=g) * 0.05)


def _random_set(g: torch.Generator, row: torch.Tensor, cands: torch.Tensor, mode: int) -> set:
    if mode == 0:
        return set()
    if mode == 1:  # random ids
        n = int(torch.randint(1, 40, (1,), generator=g))
        return set(torch.randint(0, V, (n,), generator=g).tolist())
    if mode == 2:  # every candidate penalized (forces the bound to decide)
        return set(cands.tolist())
    # the row's top ids (the realistic case: the model wants to repeat)
    k = int(torch.randint(1, 10, (1,), generator=g))
    return set(torch.topk(row, k).indices.tolist())


def test_penalized_argmax_is_presence_greedy():
    g = torch.Generator().manual_seed(1)
    for i in range(3000):
        row = _random_row(g, i % 7)
        gen = torch.randint(0, V, (int(torch.randint(0, 30, (1,), generator=g)),), generator=g).tolist()
        for p in (0.0, 1.5, 0.25, -1.0):
            assert penalized_argmax(row, set(gen), p) == presence_greedy(row, gen, p)
        # bf16 input row (the device dtype) goes through the same fp32 cast
        assert penalized_argmax(row.to(torch.bfloat16), set(gen), 1.5) == presence_greedy(
            row.to(torch.bfloat16), gen, 1.5
        )


def test_certified_argmax_property():
    """Certification is never wrong; it is refused exactly when best <= thresh; both outcomes occur often."""
    g = torch.Generator().manual_seed(2)
    certified = fallback = 0
    for i in range(N_PROPERTY_ROWS):
        kind = i % 7
        top_n = (1, 2, 4, 8, 32)[(i // 7) % 5]
        p = (1.5, 1.5, 0.0, 0.5)[(i // 35) % 4]
        row = _random_row(g, kind)
        vals, ids, thresh = reference_topn(row, N_DEV, top_n, generator=g)
        vals, ids, thresh = vals[0], ids[0], thresh[0]
        S = _random_set(g, row, ids, (i // 140) % 4)
        expected = presence_greedy(row, sorted(S), p)
        got = certified_argmax(vals, ids, thresh, S, p)
        # the independent statement of the certificate: best penalised candidate > the bound
        y = vals.clone()
        if p > 0 and S:
            y[torch.isin(ids, torch.tensor(sorted(S)))] -= p
        should_certify = bool(y.max() > thresh)
        assert (got is not None) == should_certify, (i, kind, top_n, p)
        if got is None:
            fallback += 1
        else:
            certified += 1
            assert got == expected, (i, kind, top_n, p, got, expected)
    assert certified > N_PROPERTY_ROWS // 4 and fallback > N_PROPERTY_ROWS // 50, (certified, fallback)


def test_tie_beyond_topn_falls_back_and_the_full_row_gives_the_lowest_id():
    row = torch.zeros(V)
    row[40:60] = 5.0  # 20 tied maxima on shard 1 (ids 32..63) and nothing else above 0
    vals, ids, thresh = reference_topn(row, N_DEV, 4, generator=torch.Generator().manual_seed(0))
    assert certified_argmax(vals[0], ids[0], thresh[0], set(), 0.0) is None  # 5.0 == thresh: not certifiable
    assert penalized_argmax(row, set(), 0.0) == 40 == presence_greedy(row, [], 0.0)
    # with top_n covering every tie, the lowest id is certified
    vals, ids, thresh = reference_topn(row, N_DEV, 32)
    assert certified_argmax(vals[0], ids[0], thresh[0], set(), 0.0) == 40


def test_tie_across_the_penalty_boundary():
    row = torch.zeros(V)
    row[7] = 3.0  # penalised -> 1.5
    row[200] = 1.5  # ties with the penalised 7; the lowest id (7) wins, as presence_greedy
    vals, ids, thresh = reference_topn(row, N_DEV, 2)
    got = certified_argmax(vals[0], ids[0], thresh[0], {7}, 1.5)
    assert got == presence_greedy(row, [7], 1.5) == 7
    row[3] = 1.5  # a lower id joins the tie
    # top-2 on shard 0 is {7: 3.0, 3: 1.5}: its smallest returned value equals the best (1.5), so a hidden id on that
    # shard could tie it with a lower index: not certifiable, the full row decides
    vals, ids, thresh = reference_topn(row, N_DEV, 2)
    assert certified_argmax(vals[0], ids[0], thresh[0], {7}, 1.5) is None
    assert penalized_argmax(row, {7}, 1.5) == presence_greedy(row, [7], 1.5) == 3
    # top-4: the bound drops to 0.0 and the three-way tie {3, 7, 200} at 1.5 is certified to its lowest id
    vals, ids, thresh = reference_topn(row, N_DEV, 4)
    assert certified_argmax(vals[0], ids[0], thresh[0], {7}, 1.5) == 3


def test_all_candidates_penalised_falls_back_when_a_hidden_token_could_win():
    row = torch.zeros(V)
    row[0:4] = 2.0  # shard 0's top-4, all penalised to 0.5
    row[4] = 1.0  # not returned (top_n 4 on shard 0), so thresh >= 1.0 > 0.5: must fall back
    vals, ids, thresh = reference_topn(row, N_DEV, 4)
    S = set(range(4))
    assert certified_argmax(vals[0], ids[0], thresh[0], S, 1.5) is None
    assert penalized_argmax(row, S, 1.5) == presence_greedy(row, sorted(S), 1.5) == 4


def test_nan_raises(expect_error):
    row = torch.randn(V)
    row[17] = float("nan")
    vals, ids, thresh = reference_topn(row, N_DEV, 4)
    with expect_error(RuntimeError, "NaN"):
        certified_argmax(vals[0], ids[0], thresh[0], set(), 1.5)
    with expect_error(RuntimeError, "NaN"):
        penalized_argmax(row, set(), 1.5)


def _block_reference(rows, drafts, num_valid, gen_set, p, stops):
    S = set(gen_set)
    for j in range(num_valid + 1):
        if j:
            S.add(drafts[j - 1])
        w = presence_greedy(rows[j], sorted(S), p)
        if j == num_valid or drafts[j] != w or w in stops:
            return j, w


@pytest.mark.parametrize("top_n", [1, 4, 32])
@pytest.mark.parametrize("p", [1.5, 0.0])
def test_accept_block_matches_sequential_presence_greedy(top_n, p):
    g = torch.Generator().manual_seed(3 + top_n)
    stops = {V - 1, V - 2}
    total_fallback = 0
    for case in range(1500):
        K = (1, 3, 7)[case % 3]
        rows = torch.stack([_random_row(g, (case + r) % 3 + (1 if case % 5 == 0 else 0)) for r in range(K + 1)])
        gen = set(torch.randint(0, V, (int(torch.randint(0, 20, (1,), generator=g)),), generator=g).tolist())
        num_valid = int(torch.randint(0, K + 1, (1,), generator=g))
        # drafts: the true targets with random corruption, so every m in 0..num_valid occurs
        drafts, S = [], set(gen)
        for j in range(num_valid):
            w = presence_greedy(rows[j], sorted(S), p)
            d = w if float(torch.rand(1, generator=g)) < 0.7 else int(torch.randint(0, V, (1,), generator=g))
            drafts.append(d)
            S.add(d)
        reads = []

        def read_rows(js, rows=rows, reads=reads):
            reads.extend(js)
            return rows[list(js)]

        topn = reference_topn(rows, N_DEV, top_n, generator=g)
        res = accept_block(topn, drafts, num_valid, gen, p, read_rows, stops)
        m, bonus = _block_reference(rows, drafts, num_valid, gen, p, stops)
        assert (res.m, res.bonus) == (m, bonus), case
        assert res.rows_evaluated == m + 1
        assert res.fallback_rows == len(reads) and all(j <= m for j in reads)
        total_fallback += res.fallback_rows
    if top_n == 1:
        assert total_fallback > 0


def test_accept_block_stops_at_an_accepted_stop_draft_and_ignores_rows_after_m():
    rows = torch.zeros(4, V)
    rows[0, 10] = 5.0
    rows[1, V - 1] = 5.0  # the target after draft 10 is the stop id V-1
    rows[2:] = float("nan")  # garbage rows past the decision are never looked at
    stops = {V - 1}

    def read_rows(js):
        raise AssertionError("no fallback expected")

    topn = reference_topn(rows[:2], N_DEV, 4)
    topn = tuple(torch.cat([t, t[:1].repeat(2, *([1] * (t.dim() - 1)))]) for t in topn)
    res = accept_block(topn, [10, V - 1, 3], 3, {1}, 1.5, read_rows, stops)
    assert (res.m, res.bonus, res.rows_evaluated) == (1, V - 1, 2)
