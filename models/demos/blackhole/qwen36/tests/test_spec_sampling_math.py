# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only math regressions for exact speculative sampling (``tt/spec_sampling.py``).

No ttnn, no device: this pins the host-side distribution math and the losslessness of
rejection sampling against the drafter's greedy (deterministic, delta-proposal) drafts.

Run:
  pytest -q -p no:cacheprovider models/demos/blackhole/qwen36/tests/test_spec_sampling_math.py

The statistical tests are seeded end to end (fixed logits generator + fixed sampler seed),
so they are deterministic — the total-variation bounds below are headroom over the
sampling noise of the fixed draw, not flake budget.
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSampler, SpecSamplingParams

# (temperature, top_k, top_p)
_DIST_GRID = [
    (1.0, 0, 1.0),
    (0.7, 20, 1.0),
    (1.0, 0, 0.9),
    (1.0, 20, 0.95),
    (0.6, 50, 0.8),
    (1.5, 1, 1.0),
    (1.0, 1, 0.95),
]


def _expect_raises(exc_type, fn, *args, **kwargs):
    """Assert ``fn(*args, **kwargs)`` raises: the repo pre-commit forbids the pytest helper."""
    raised = False
    try:
        fn(*args, **kwargs)
    except exc_type:
        raised = True
    assert raised, f"expected {exc_type.__name__} from {getattr(fn, '__name__', fn)}({args}, {kwargs})"


def _ref_dense_dist(logits, temperature, top_k, top_p):
    """Naive dense reference: scale -> top-k mask -> softmax -> top-p keep -> renormalize."""
    z = logits.float() / temperature
    if top_k > 0:
        _, keep = torch.topk(z, min(top_k, z.numel()))
        masked = torch.full_like(z, -float("inf"))
        masked[keep] = z[keep]
        z = masked
    probs = torch.softmax(z, 0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    if top_p < 1.0:
        cum = torch.cumsum(sorted_probs, 0)
        sorted_probs = sorted_probs * ((cum - sorted_probs) < top_p)
    sorted_probs = sorted_probs / sorted_probs.sum()
    out = torch.zeros_like(probs)
    out[sorted_idx] = sorted_probs
    return out


def _dense_of(sampler, logits, penalize=None):
    """Scatter a sparse ``dist()`` support back into a dense [vocab] probability vector."""
    idx, probs = sampler.dist(logits, penalize)
    assert idx.dtype == torch.int64
    assert probs.dtype == torch.float32
    assert idx.dim() == 1 and probs.dim() == 1 and idx.numel() == probs.numel() >= 1
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
    out = torch.zeros(sampler.vocab_size)
    out[idx] = probs
    return out


def _tv(counts, probs):
    """Total-variation distance between an empirical histogram and a dense distribution."""
    total = float(counts.sum())
    assert total > 0
    return float(0.5 * (counts / total - probs).abs().sum())


@pytest.mark.parametrize(
    "vocab,params",
    [(1000, p) for p in _DIST_GRID] + [(248320, (1.0, 20, 0.95)), (248320, (1.0, 0, 0.95))],
    ids=lambda v: str(v),
)
def test_dist_matches_reference(vocab, params):
    temperature, top_k, top_p = params
    logits = torch.randn(vocab, generator=torch.Generator().manual_seed(vocab)) * 3
    sampler = SpecSampler(SpecSamplingParams(temperature, top_k, top_p, seed=0), vocab)

    got = _dense_of(sampler, logits)
    want = _ref_dense_dist(logits, temperature, top_k, top_p)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)

    idx, probs = sampler.dist(logits)
    if top_k == 0 and top_p == 1.0:
        # Dense path: whole vocabulary, vocabulary order, no sort.
        assert torch.equal(idx, torch.arange(vocab))
    else:
        assert torch.all(probs[:-1] >= probs[1:]), "truncated support must be sorted by probability"
        if top_k > 0:
            assert idx.numel() <= top_k
    # prob_of agrees with the dense reconstruction, inside and outside the support.
    for token in (int(logits.argmax()), int(logits.argmin()), 0, vocab - 1):
        assert sampler.prob_of((idx, probs), token) == pytest.approx(float(got[token]), abs=1e-7)


def test_dist_top_p_prefix_fast_path():
    """A peaked 248k row: top-p is exact from the 2048-prefix, no full sort.

    The scale matters here. iid N(0, 3^2) logits over a 248k vocabulary are much flatter
    than a real LM head — their 0.95 support is ~22k tokens, so the prefix genuinely
    cannot cover it (that row is checked for exactness by the parametrized test above,
    via the fallback). Scale 5 gives a realistically peaked row (p_max ~ 0.22, 0.95
    support ~430 tokens), which is what the prefix path is built for.
    """
    vocab = 248320
    logits = torch.randn(vocab, generator=torch.Generator().manual_seed(7)) * 5
    sampler = SpecSampler(SpecSamplingParams(1.0, 0, 0.95, seed=0), vocab)

    got = _dense_of(sampler, logits)
    want = _ref_dense_dist(logits, 1.0, 0, 0.95)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    assert int((got > 0).sum()) < 2048, "this row's top-p support must fit in the prefix"
    assert sampler._topp_full_sorts == 0, "peaked row must not need the full-sort fallback"


def test_dist_top_p_full_sort_fallback():
    """A near-uniform row: the top-p support runs past 2048, so the slow path must fire."""
    vocab = 5000
    logits = torch.zeros(vocab) + 1e-3 * torch.randn(vocab, generator=torch.Generator().manual_seed(8))
    sampler = SpecSampler(SpecSamplingParams(1.0, 0, 0.999, seed=0), vocab)

    got = _dense_of(sampler, logits)
    want = _ref_dense_dist(logits, 1.0, 0, 0.999)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    assert sampler._topp_full_sorts == 1, "flat row must fall back to the exact full sort"


@pytest.mark.parametrize("params", [(1.0, 0, 1.0), (1.0, 8, 0.9), (0.7, 20, 1.0)], ids=lambda v: str(v))
def test_accept_lossless_single_position(params):
    """One draft: the emitted token must be distributed exactly as dist(logits[0])."""
    vocab, num_samples = 64, 60000
    logits = torch.randn(2, vocab, generator=torch.Generator().manual_seed(0)) * 2
    order = torch.argsort(logits[0], descending=True)
    # argmax, runner-up, and (only when top-k truncates) a token with zero target mass.
    drafts = [int(order[0]), int(order[1])] + ([int(order[19])] if params[1] == 8 else [])

    for slot, draft in enumerate(drafts):
        sampler = SpecSampler(SpecSamplingParams(*params, seed=0), vocab)
        target = _dense_of(sampler, logits[0])
        if slot == 2:  # the out-of-support draft can never be accepted
            assert sampler.prob_of(sampler.dist(logits[0]), draft) == 0.0
        counts = torch.zeros(vocab)
        for _ in range(num_samples):
            accepted, next_token, p_draft = sampler.accept(logits, [draft])
            assert accepted in (0, 1)
            assert len(p_draft) == 1
            counts[draft if accepted >= 1 else next_token] += 1
        assert _tv(counts, target) < 0.015


def test_accept_lossless_chain():
    """K=3 chained argmax drafts: every position stays exact, conditioned on reaching it.

    The parameter set is truncated (top_k=8/top_p=0.95) on purpose: with an untruncated
    64-token softmax the acceptance chain leaves only ~1-2k samples on the bonus row, and
    the multinomial noise floor there (~0.05 TV) already exceeds the bounds asserted here.
    """
    vocab, num_samples = 64, 60000
    logits = torch.randn(4, vocab, generator=torch.Generator().manual_seed(3)) * 2
    drafts = [int(logits[j].argmax()) for j in range(3)]
    sampler = SpecSampler(SpecSamplingParams(1.0, 8, 0.95, seed=0), vocab)

    counts = [torch.zeros(vocab) for _ in range(4)]
    for _ in range(num_samples):
        accepted, next_token, p_draft = sampler.accept(logits, drafts)
        assert len(p_draft) == min(accepted + 1, 3)
        if accepted >= 1:
            counts[1][drafts[1] if accepted >= 2 else next_token] += 1
        if accepted >= 2:
            counts[2][drafts[2] if accepted >= 3 else next_token] += 1
        if accepted == 3:
            counts[3][next_token] += 1

    for position, bound in ((1, 0.02), (2, 0.02), (3, 0.03)):
        assert counts[position].sum() > 1000, f"position {position} got too few samples to test"
        assert _tv(counts[position], _dense_of(sampler, logits[position])) < bound


@pytest.mark.parametrize("temperature", [0.5, 1.0])
@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_top_k1_is_greedy(temperature, top_p):
    """top_k == 1 collapses rejection sampling onto the greedy accept/argmax path."""
    vocab, num_drafts, num_cases = 300, 5, 200
    gen = torch.Generator().manual_seed(4)
    sampler = SpecSampler(SpecSamplingParams(temperature, 1, top_p, seed=7), vocab)

    for _ in range(num_cases):
        logits = torch.randn(num_drafts + 1, vocab, generator=gen) * 3
        argmax = [int(logits[j].argmax()) for j in range(num_drafts + 1)]
        drafts = list(argmax[:num_drafts])
        for j in range(num_drafts):
            if float(torch.rand(1, generator=gen)) < 0.3:
                alt = int(torch.randint(vocab, (1,), generator=gen))
                drafts[j] = alt if alt != argmax[j] else (alt + 1) % vocab

        expected = 0
        while expected < num_drafts and drafts[expected] == argmax[expected]:
            expected += 1

        accepted, next_token, p_draft = sampler.accept(logits, drafts)
        assert accepted == expected
        assert next_token == argmax[accepted]
        assert len(p_draft) == min(accepted + 1, num_drafts)
        assert all(p == 1.0 for p in p_draft[:accepted])
        if accepted < num_drafts:
            assert p_draft[accepted] == 0.0
        assert sampler.pick(logits[0]) == argmax[0]


def _run_cases(sampler, cases):
    return [sampler.accept(logits, drafts) for logits, drafts in cases]


def test_seed_determinism():
    vocab, num_cases = 64, 50
    gen = torch.Generator().manual_seed(11)
    cases = []
    for case in range(num_cases):
        logits = torch.randn(4, vocab, generator=gen) * 2
        if case % 2 == 0:  # mix chains that accept with drafts that get rejected
            drafts = [int(logits[j].argmax()) for j in range(3)]
        else:
            drafts = [int(torch.randint(vocab, (1,), generator=gen)) for _ in range(3)]
        cases.append((logits, drafts))

    def sampler_with(seed):
        return SpecSampler(SpecSamplingParams(1.0, 0, 1.0, seed=seed), vocab)

    same = _run_cases(sampler_with(123), cases)
    assert same == _run_cases(sampler_with(123), cases)
    assert same != _run_cases(sampler_with(124), cases)

    unseeded = SpecSampler(SpecSamplingParams(1.0, 0, 1.0, seed=None), vocab)
    assert isinstance(unseeded.seed, int)
    drawn = _run_cases(unseeded, cases)
    assert drawn == _run_cases(sampler_with(unseeded.seed), cases)


def test_params_validation():
    SpecSamplingParams(1.0)  # the permissive default must stay legal
    for bad in (
        dict(temperature=0.0),
        dict(temperature=-1.0),
        dict(temperature=1.0, top_p=0.0),
        dict(temperature=1.0, top_p=1.5),
        dict(temperature=1.0, top_k=-1),
        dict(temperature=1.0, presence_penalty=-0.1),
    ):
        _expect_raises(AssertionError, SpecSamplingParams, **bad)


# --------------------------------------------------------------------------- presence penalty

# The output set the presence-penalty tests below penalize: 10 fixed ids, none of which is the
# draft token, so a variant can swap one for the draft to watch its target probability collapse.
_PP_BASE = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def _pp_chain_logits(vocab):
    """Two correlated target rows + a bonus row, for the presence-penalty chain check.

    Row 1 is a perturbation of row 0, so row 0's draft is a HIGH-probability token in row 1 as
    well — which is exactly the case the penalty exists for, and the only case where forgetting
    ``drafts[:j]`` on row 1 is observable.
    """
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(3, vocab, generator=gen) * 0.5
    logits[1] = logits[0] + 0.5 * torch.randn(vocab, generator=gen)
    return logits


@pytest.mark.parametrize("presence", [0.5, 1.5])
def test_presence_penalty_dist_matches_reference(presence):
    """``dist(logits, penalize)`` is the plain distribution of ``logits - pp * onehot(penalize)``."""
    vocab, num_penalized = 300, 12
    temperature, top_k, top_p = 0.7, 20, 0.8
    gen = torch.Generator().manual_seed(87)
    logits = torch.randn(vocab, generator=gen)
    penalize = torch.randperm(vocab, generator=gen)[:num_penalized].to(torch.int64)
    sampler = SpecSampler(SpecSamplingParams(temperature, top_k, top_p, presence_penalty=presence, seed=0), vocab)

    penalized = logits.clone()
    penalized[penalize] -= presence
    want = _ref_dense_dist(penalized, temperature, top_k, top_p)
    untouched = logits.clone()
    got = _dense_of(sampler, logits, penalize)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    # A float32 row is the CALLER's (the verify block is read back once and its rows handed out as
    # views), so the penalty must land on a copy.
    assert torch.equal(logits, untouched), "dist() mutated the caller's logits row"
    # This seed's penalized set overlaps the target's support, so the penalty is not a no-op here —
    # otherwise the reference above would agree even with `penalize` ignored.
    unpenalized = _dense_of(sampler, logits)
    assert _tv(got, unpenalized) > 0.02, "penalized set must reach the support, or the test is vacuous"

    # pp == 0 ignores the set entirely, bit for bit.
    off = SpecSampler(SpecSamplingParams(temperature, top_k, top_p, seed=0), vocab)
    assert torch.equal(_dense_of(off, logits, penalize), _dense_of(off, logits))


@pytest.mark.parametrize("draft_penalized", [False, True], ids=["draft_not_in_set", "draft_in_set"])
def test_presence_penalty_accept_lossless(draft_penalized):
    """The emitted token is distributed exactly as the PENALIZED target row.

    Rejection sampling is lossless w.r.t. whatever ``p`` the accept test uses, so with the penalty
    applied to the verify rows the emitted token must follow ``dist(logits[0], penalize_base)`` —
    including when the penalty is what pushed the draft out of the support (``draft_in_set``).
    """
    vocab, num_samples, presence = 64, 40000, 1.5
    params = (0.7, 8, 0.8)
    logits = _pp_chain_logits(vocab)[:2]  # one draft, so T == 2: the target row + the bonus row
    draft = int(logits[0].argmax())  # the drafter does not know about the penalty: plain argmax
    base = sorted(set(_PP_BASE[:9] + [draft])) if draft_penalized else list(_PP_BASE)
    assert len(base) == 10 and (draft in base) == draft_penalized
    penalize_base = torch.tensor(base, dtype=torch.int64)

    sampler = SpecSampler(SpecSamplingParams(*params, presence_penalty=presence, seed=0), vocab)
    target = _dense_of(sampler, logits[0], penalize_base)
    p_draft_pen = sampler.prob_of(sampler.dist(logits[0], penalize_base), draft)
    p_draft_raw = sampler.prob_of(sampler.dist(logits[0]), draft)
    if draft_penalized:
        assert p_draft_pen < p_draft_raw, "penalizing the draft must lower its target probability"

    counts = torch.zeros(vocab)
    for _ in range(num_samples):
        accepted, next_token, p_draft = sampler.accept(logits, [draft], penalize_base)
        assert accepted in (0, 1)
        assert p_draft == [p_draft_pen], "the accept test must run on the PENALIZED row"
        counts[draft if accepted >= 1 else next_token] += 1
    tv = _tv(counts, target)
    assert tv < 0.015, f"draft_penalized={draft_penalized}: TV={tv:.4f}"


def test_presence_penalty_accept_chain_penalizes_drafts():
    """Verify row 1 is penalized on ``penalize_base ∪ drafts[:1]``, not just on ``penalize_base``.

    Conditioned on reaching it (draft 0 accepted), the position-1 token must follow
    ``dist(logits[1], penalize_base ∪ {drafts[0]})``. Row 1 here gives ``drafts[0]`` ~0.2 of its
    unpenalized mass, so dropping that extra id from the set moves the row by TV ~0.2 — an order of
    magnitude past the bound below.
    """
    vocab, num_samples, presence = 64, 40000, 1.5
    logits = _pp_chain_logits(vocab)
    drafts = [int(logits[0].argmax()), int(logits[1].argmax())]
    penalize_base = torch.tensor(_PP_BASE, dtype=torch.int64)
    row1_set = torch.cat([penalize_base, torch.tensor([drafts[0]], dtype=torch.int64)]).unique()

    sampler = SpecSampler(SpecSamplingParams(0.7, 8, 0.8, presence_penalty=presence, seed=0), vocab)
    want = _dense_of(sampler, logits[1], row1_set)
    without = _dense_of(sampler, logits[1], penalize_base)
    assert _tv(want, without) > 0.05, "drafts[0] must matter on row 1, or this test is vacuous"

    counts = torch.zeros(vocab)
    for _ in range(num_samples):
        accepted, next_token, _ = sampler.accept(logits, drafts, penalize_base)
        if accepted >= 1:  # position 1 exists only when draft 0 was accepted
            counts[drafts[1] if accepted >= 2 else next_token] += 1
    reached = int(counts.sum())
    tv = _tv(counts, want)
    assert reached > 1000, f"position 1 was reached only {reached}/{num_samples} times"
    assert tv < 0.02, f"position 1: TV={tv:.4f} over {reached}/{num_samples} surviving chains"


def test_presence_penalty_zero_is_noop():
    """``presence_penalty == 0`` ignores ``penalize_base``: same tokens, same RNG stream."""
    vocab, num_drafts, num_cases = 300, 4, 100
    gen = torch.Generator().manual_seed(71)
    penalize_base = torch.randperm(vocab, generator=gen)[:16].to(torch.int64)
    cases = []
    for case in range(num_cases):
        logits = torch.randn(num_drafts + 1, vocab, generator=gen) * 2
        if case % 2 == 0:  # mix chains that accept with drafts that get rejected
            drafts = [int(logits[j].argmax()) for j in range(num_drafts)]
        else:
            drafts = [int(torch.randint(vocab, (1,), generator=gen)) for _ in range(num_drafts)]
        cases.append((logits, drafts))

    def sampler():
        return SpecSampler(SpecSamplingParams(1.0, 20, 0.95, seed=13), vocab)

    with_set, without_set = sampler(), sampler()
    for case, (logits, drafts) in enumerate(cases):
        got = with_set.accept(logits, drafts, penalize_base)
        want = without_set.accept(logits, drafts)
        assert got == want, f"case {case}: {got} != {want}"
