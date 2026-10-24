# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device speculative acceptance (kernels/sample_verify.cpp) on the host: its integer mirror
(reference/sample_verify.py) against the plugin's accept-walk semantics in float64 (no device, no ttnn).

The reference here is vllm_tt_plugin.spec_accept's walk for point-mass drafts, written out over the whole vocabulary:
a slot's distribution is softmax(logits / T) over the vocabulary, or over the top_k largest (logit descending, id
ascending), the nucleus keeping a token while the mass ranked above it is below top_p; draft j is accepted when
p_j(d_j) > 0 and p_j(d_j) >= u_accept[j]; the first rejection commits a draw from p_j without d_j (u_corr), a fully
accepted row the bonus from p_n (u_bonus); draws by inverse CDF, in (logit descending, id ascending) order for a top_k
row and in id order for a nucleus row without top_k. Greedy rows commit the argmax chain (the lowest id among tied
maxima). The mirror must agree wherever its candidates hold the nucleus (it says so: not ``truncated``) and no
uniform lies within rounding of a decision.
"""

import math
import random

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference import sample_verify as sv
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    GROUP_SLOTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    plan_verify_passes,
)

TP = 8
LOCAL = 64
VOCAB = TP * LOCAL
K = 8  # candidates per chip: 64 in all, so a wide nucleus runs past them
EDGE = 1e-6


# ---------------------------------------------------------------------- the float64 reference
def ref_dist(logits, temperature, top_k, top_p):
    """(probabilities {id: p} of the kept tokens, their draw order, whether the nucleus cut lies within rounding)."""
    vocab = len(logits)
    order = sorted(range(vocab), key=lambda i: (-logits[i], i))
    has_k = 1 <= top_k < vocab
    cand = order[:top_k] if has_k else order
    top = logits[order[0]]
    w = {i: math.exp((logits[i] - top) / temperature) for i in cand}
    total = sum(w.values())
    kept, above, near = [], 0.0, False
    for i in cand:
        near |= bool(kept) and abs(above - top_p * total) < EDGE * total
        if kept and above >= top_p * total:
            break
        kept.append(i)
        above += w[i]
    kept_total = sum(w[i] for i in kept)
    probs = {i: w[i] / kept_total for i in kept}
    return probs, (kept if has_k else sorted(kept)), near


def ref_draw(probs, order, u, exclude=None):
    """(token, whether u lies within EDGE of a bucket edge)."""
    weights = [(i, probs[i]) for i in order if i != exclude and probs[i] > 0]
    rest = sum(p for _, p in weights)
    threshold = u * rest
    run, near = 0.0, False
    pick = weights[-1][0]
    for i, p in weights:
        run += p
        near |= abs(run - threshold) < EDGE
        if run > threshold:
            pick = i
            break
    return pick, near


def ref_walk(slot_logits, params, drafts, greedy, u_bonus, u_accept, u_corr):
    """(count, tokens, near) of one request over its slots' logits [n + 1][vocab]."""
    n = len(drafts)
    if greedy:
        tokens = []
        for j in range(n + 1):
            row = slot_logits[j]
            top = max(row)
            tokens.append(min(i for i, x in enumerate(row) if x == top))
            if j < n and tokens[-1] != drafts[j]:
                return j + 1, tokens, False
        return n + 1, tokens, False
    temperature, top_k, top_p = params
    tokens, near = [], False
    for j in range(n):
        probs, order, cut = ref_dist(slot_logits[j], temperature, top_k, top_p)
        p = probs.get(drafts[j], 0.0)
        near |= cut or (p > 0 and abs(p - u_accept[j]) < EDGE)
        if p > 0 and p >= u_accept[j]:
            tokens.append(drafts[j])
            continue
        token, edge = ref_draw(probs, order, u_corr, exclude=drafts[j])
        return j + 1, tokens + [token], near or edge
    probs, order, cut = ref_dist(slot_logits[n], temperature, top_k, top_p)
    token, edge = ref_draw(probs, order, u_bonus)
    return n + 1, tokens + [token], near or edge or cut


# ---------------------------------------------------------------------- the mirror over one request
def q24(u):
    return sv.uniform_q24(u) / float(1 << 24)


def f32(x):
    return float(torch.tensor(x, dtype=torch.float32))


def argmax_of(row):
    top = max(row)
    return min(i for i, x in enumerate(row) if x == top)


def mirror_walk(slot_logits, params, drafts, greedy, u_bonus, u_accept, u_corr, first=0):
    """The mirror's (count, tokens, truncated) for one request whose slots start at ``first``."""
    slots = len(slot_logits)
    temperature, top_k, top_p = params
    lists = [None] * first + [sv.chip_lists(row, temperature, TP, K) for row in slot_logits]
    words = [sv.param_words(temperature, top_k, top_p, VOCAB)] * (first + slots)
    argmax = [0] * first + [argmax_of(row) for row in slot_logits]
    ctrl = sv.group_words(first, drafts, greedy, u_bonus, u_accept, u_corr)
    return sv.verify_group(lists, words, argmax, ctrl, LOCAL)


def make_logits(rng, scale, vocab=VOCAB):
    return [f32(rng.gauss(0.0, 1.0) * scale) for _ in range(vocab)]


def compare(slot_logits, params, drafts, greedy, u_bonus, u_accept, u_corr):
    """Mirror against the reference with the uniforms the kernel sees (Q24). Returns (mirror result, near)."""
    got = mirror_walk(slot_logits, params, drafts, greedy, u_bonus, u_accept, u_corr)
    count, tokens, near = ref_walk(
        slot_logits, params, drafts, greedy, q24(u_bonus), [q24(u) for u in u_accept], q24(u_corr)
    )
    if not got[2] and not near:
        assert (got[0], got[1]) == (count, tokens), f"mirror {got[:2]} != reference {(count, tokens)}"
    return got, near


# ---------------------------------------------------------------------- fixed point
def test_relative_probabilities_track_the_exponential():
    for scale_t in (0.5, 1.0, 1.7):
        inv = 1.0 / scale_t
        rel = sv.Rel(sv.f32_bits(inv * sv.LOG2E), sv.f32_bits(3.0))
        for v in (3.0, 2.5, 0.0, -4.0, -20.0):
            want = math.exp((v - 3.0) * inv)
            assert abs(rel(sv.f32_bits(v)) / sv.ONE30 - want) < 2e-8, (scale_t, v)


def test_uniforms_quantize_down_to_q24():
    assert sv.uniform_q24(0.0) == 0
    assert sv.uniform_q24(0.5) == 1 << 23
    assert sv.uniform_q24(float(torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)))) == (1 << 24) - 1
    assert sv.uniform_q24(1.0) == (1 << 24) - 1


def test_param_words_follow_the_sampler_layout():
    w = sv.param_words(0.7, 20, 0.95, VOCAB)
    inv = f32(1.0 / 0.7)
    assert w[0] == sv.f32_bits(inv * sv.LOG2E) and w[4] == sv.f32_bits(inv)
    assert w[1] == round(0.95 * (1 << 24)) and w[2] == 20 and w[3] == 0
    # top_k at or past the vocabulary, or below 1, is none; temperature 0 or top_k 1 is greedy.
    assert sv.param_words(1.0, VOCAB, 0.9, VOCAB)[2] == 0 and sv.param_words(1.0, -1, 0.9, VOCAB)[2] == 0
    assert sv.param_words(0.0, 0, 1.0, VOCAB)[3] == 1 and sv.param_words(1.0, 1, 1.0, VOCAB)[3] == 1


# ---------------------------------------------------------------------- acceptance
@pytest.mark.parametrize("top_k, top_p", [(0, 0.95), (20, 1.0), (20, 0.8), (5, 0.9), (0, 0.5)])
def test_random_requests_match_the_host_walk(top_k, top_p):
    rng = random.Random(top_k * 1000 + int(top_p * 100))
    checked = 0
    for trial in range(60):
        n = rng.randrange(0, MAX_DRAFTS + 1)
        slot_logits = [make_logits(rng, rng.choice((3.0, 5.0, 8.0))) for _ in range(n + 1)]
        temperature = rng.choice((0.6, 1.0, 1.3))
        # Drafts the target likes (so some are accepted), sometimes any token.
        drafts = [argmax_of(slot_logits[j]) if rng.random() < 0.6 else rng.randrange(VOCAB) for j in range(n)]
        u = [rng.random() for _ in range(n + 2)]
        got, near = compare(slot_logits, (temperature, top_k, top_p), drafts, False, u[0], u[1 : 1 + n], u[n + 1])
        checked += not got[2] and not near
        assert 1 <= got[0] <= n + 1 and len(got[1]) == got[0]
        assert got[1][: got[0] - 1] == drafts[: got[0] - 1], "a committed prefix that is not the drafts"
    # Longer walks (7 drafts with 8 slots) leave the certified candidates more often.
    assert checked >= (40 if MAX_DRAFTS <= 3 else 30), f"only {checked} of 60 requests compared exactly"


def test_accept_boundary_is_p_of_the_draft():
    rng = random.Random(1)
    row = make_logits(rng, 4.0)
    probs, _, _ = ref_dist(row, 1.0, 0, 0.95)
    draft = sorted(probs, key=lambda i: -probs[i])[1]  # the second most likely token
    p = probs[draft]
    for u, accepted in ((p - 1e-4, True), (p + 1e-4, False), (0.0, True)):
        count, tokens, _ = mirror_walk([row, row], (1.0, 0, 0.95), [draft], False, 0.3, [u], 0.4)
        assert (count == 2) == accepted and (tokens[0] == draft) == accepted, (u, p, tokens)


def test_a_rejection_draws_from_the_residual_without_the_draft():
    rng = random.Random(2)
    row = make_logits(rng, 6.0)  # a nucleus inside the candidates
    probs, order, cut = ref_dist(row, 1.0, 0, 0.95)
    assert len(order) > 2
    draft = order[0]  # the smallest id of the nucleus: the first bucket of an id-order draw
    for u_corr in (0.0, 0.25, 0.5, 0.99):
        count, tokens, truncated = mirror_walk([row, row], (1.0, 0, 0.95), [draft], False, 0.1, [0.999999], u_corr)
        assert count == 1 and tokens[0] != draft and not truncated
        want, near = ref_draw(probs, order, q24(u_corr), exclude=draft)
        assert near or tokens[0] == want


def test_a_draft_outside_the_nucleus_is_rejected_even_at_u_zero():
    rng = random.Random(3)
    row = make_logits(rng, 6.0)
    probs, _, _ = ref_dist(row, 1.0, 5, 1.0)
    outside = next(i for i in range(VOCAB) if i not in probs)
    count, tokens, _ = mirror_walk([row, row], (1.0, 5, 1.0), [outside], False, 0.5, [0.0], 0.5)
    assert count == 1 and tokens[0] in probs


def test_every_draft_accepted_commits_the_bonus_from_the_last_slot():
    rng = random.Random(4)
    slots = [make_logits(rng, 5.0) for _ in range(MAX_DRAFTS + 1)]
    drafts = [argmax_of(slots[j]) for j in range(MAX_DRAFTS)]
    for u_bonus in (0.0, 0.3, 0.97):
        count, tokens, _ = mirror_walk(slots, (1.0, 10, 1.0), drafts, False, u_bonus, [0.0] * MAX_DRAFTS, 0.5)
        assert count == MAX_DRAFTS + 1 and tokens[:MAX_DRAFTS] == drafts
        probs, order, cut = ref_dist(slots[-1], 1.0, 10, 1.0)
        want, near = ref_draw(probs, order, q24(u_bonus))
        assert near or tokens[-1] == want


def test_top_k_rows_draw_in_logit_order_and_nucleus_rows_in_id_order():
    """The same uniform picks the same token as the host walk, whose top-k table is ordered by logit and whose
    top-p-only table by id: two orders that pick different tokens for one uniform."""
    rng = random.Random(5)
    row = make_logits(rng, 1.0)
    for top_k, top_p in ((30, 1.0), (0, 0.3)):
        probs, order, cut = ref_dist(row, 1.0, top_k, top_p)
        assert len(order) > 3 and not cut
        for u in (0.05, 0.4, 0.8):
            _, tokens, _ = mirror_walk([row], (1.0, top_k, top_p), [], False, u, [], 0.0)
            want, near = ref_draw(probs, order, q24(u))
            assert near or tokens == [want], (top_k, u)


def test_greedy_rows_commit_the_argmax_chain_lowest_id_on_ties():
    rng = random.Random(6)
    slots = [make_logits(rng, 2.0) for _ in range(MAX_DRAFTS + 1)]
    # Tied maxima on two chips: the lower id is the argmax.
    for row, (a, b) in zip(slots, ((70, 3 * LOCAL + 5), (400, 9), (130, 131), (2, LOCAL + 2))):
        row[a] = row[b] = max(row) + 1.0
    argmax = [argmax_of(row) for row in slots]
    assert argmax[0] == 70 and argmax[1] == 9
    for drafts, count in ((argmax[:3], 4), ([argmax[0], 7, argmax[2]], 2), ([1, 2, 3], 1)):
        got = mirror_walk(slots, (0.0, 0, 1.0), drafts, True, 0.9, [0.9] * 3, 0.9)
        assert got == (count, argmax[:count], False)
    # The argmax_ids walk on the same ids commits the same (reference.spec_decode.accept_greedy).
    from models.experimental.mimo_v2_6_flash.reference.spec_decode import accept_greedy

    assert accept_greedy(argmax, [argmax[0], 7, argmax[2]], 3) == argmax[:2]


def test_a_nucleus_past_the_candidates_is_flagged_truncated():
    """One chip holds more than K of the row's largest logits: once its list is spent, the merge can no longer certify
    the next candidate; a wide nucleus also runs out of candidates."""
    rng = random.Random(7)
    row = make_logits(rng, 0.1)
    for i in range(2 * K):
        row[i] = 5.0 - 0.01 * i  # chip 0's 2K largest are the row's largest
    count, tokens, truncated = mirror_walk([row], (1.0, 0, 0.99), [], False, 0.5, [], 0.0)
    assert truncated and count == 1 and 0 <= tokens[0] < VOCAB
    # A narrow nucleus inside chip 0's list is exact.
    count, tokens, truncated = mirror_walk([row], (1.0, 3, 1.0), [], False, 0.5, [], 0.0)
    assert not truncated and tokens[0] in (0, 1, 2)
    # Flat logits, nucleus wider than every candidate.
    flat = [0.0] * VOCAB
    assert mirror_walk([flat], (1.0, 0, 0.95), [], False, 0.5, [], 0.0)[2]


# ---------------------------------------------------------------------- a pass: layouts and several requests
@pytest.mark.skipif(
    GROUP_SLOTS < SPEC_SLOTS, reason="draft groups place each request in a group (test_spec_groups_host)"
)
def test_a_pass_with_two_requests_round_trips_through_the_pages():
    rng = random.Random(8)
    rows = [(0, [11, 12, -1, -1], [40, 41, -1, -1], 1), (1, [13, -1, -1, -1], [9, -1, -1, -1], 0)]
    (vp,) = plan_verify_passes(rows)
    slot_logits = [make_logits(rng, 4.0) for _ in range(3)]
    slot_logits[0][12] = max(slot_logits[0]) + 10.0  # request 0's draft is all but certain
    spec_sampling = {
        "temperature": torch.tensor([1.0, 0.0]),
        "top_k": torch.tensor([0, 1], dtype=torch.int32),
        "top_p": torch.tensor([0.9, 1.0]),
        "greedy": torch.tensor([False, True]),
        "uniforms": torch.tensor([[0.3, 0.5, 0.0, 0.0, 0.7], [0.0] * 5]),
    }
    slots, groups = sv.pass_inputs(vp, spec_sampling)
    assert set(slots) == {0, 1} and [g[0] for g in groups] == [0, 2]
    assert groups[0][1] == [12] and not groups[0][2] and groups[1][1] == [] and groups[1][2]
    lists = [sv.chip_lists(row, 1.0, TP, K) for row in slot_logits] + [None]
    words = [sv.param_words(*slots.get(s, (0.0, 0, 1.0)), VOCAB) for s in range(4)]
    argmax = [argmax_of(row) for row in slot_logits] + [0]
    out = sv.unpack_out(sv.verify_pass(lists, words, argmax, sv.control_words(groups), LOCAL))
    assert out[2][0] == 0 and out[3][0] == 0  # unused groups
    committed = torch.full((2, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    sv.commit_pass(vp, out, committed, counts)
    assert counts.tolist() == [2, 1] and committed[0, 0] == 12 and committed[1, 0] == argmax[2]
    probs, order, cut = ref_dist(slot_logits[1], 1.0, 0, 0.9)
    want, near = ref_draw(probs, order, q24(0.3))
    assert near or committed[0, 1] == want
    assert committed[0, 2:].tolist() == [-1] * (MAX_DRAFTS - 1) and committed[1, 1:].tolist() == [-1] * MAX_DRAFTS


def test_against_the_plugin_walk_with_the_same_uniforms():
    """When the plugin is installed: its accept_speculated_tokens on the same logits and uniforms commits what the
    mirror commits (rows whose nucleus the candidates hold, away from rounding)."""
    spec_accept = pytest.importorskip("vllm_tt_plugin.spec_accept")
    rng = random.Random(9)
    agree = 0
    for _ in range(30):
        n = MAX_DRAFTS
        slot_logits = [make_logits(rng, 5.0) for _ in range(n + 1)]
        drafts = [argmax_of(slot_logits[j]) if rng.random() < 0.7 else rng.randrange(VOCAB) for j in range(n)]
        u = [rng.random() for _ in range(n + 2)]
        for top_k, top_p in ((0, 0.95), (12, 0.9)):
            got, near = compare(slot_logits, (1.0, top_k, top_p), drafts, False, u[0], u[1 : 1 + n], u[n + 1])
            if got[2] or near:
                continue
            result = spec_accept.accept_speculated_tokens(
                target_logits=torch.tensor([slot_logits]),
                draft_token_ids=torch.tensor([drafts], dtype=torch.int32),
                num_valid_drafts=torch.tensor([n], dtype=torch.int32),
                temperature=torch.tensor([1.0]),
                top_k=torch.tensor([top_k if top_k else VOCAB], dtype=torch.int32),
                top_p=torch.tensor([top_p]),
                uniforms=torch.tensor([[q24(x) for x in u]], dtype=torch.float64),
            )
            count = int(result.accepted_counts[0])
            assert (count, result.committed_token_ids[0, :count].tolist()) == got[:2]
            agree += 1
    assert agree >= 20
