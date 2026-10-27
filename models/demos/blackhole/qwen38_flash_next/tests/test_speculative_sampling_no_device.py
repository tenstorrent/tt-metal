# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact speculative sampling on the MTP verify rows, without a device.

The point-mass acceptance's output law, integrated exactly over the uniforms, is the target's per row and per pass.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step
from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import accept_select
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2, sampling
from models.demos.blackhole.qwen38_flash_next.ttnn import speculative_sampling as spec
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    VOCAB_SIZE,
    ZERO_EMBEDDING_TOKEN,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38RowDistribution,
    Qwen38SamplingParameters,
    candidate_distribution,
    full_distribution,
    sample_candidates,
    sample_full_vocabulary,
)

PEAK_LIMIT = 200_000  # peaks below the tokenizer size: a synthetic stream never ends with ``error``
EXACT = 1e-12


# --- synthetic distributions and the exact law of the algorithm -----------------------------------------------


def _distribution(seed: int, size: int = 6, *, zeros: int = 0, dtype=torch.float64) -> Qwen38RowDistribution:
    """A random distribution over ``size`` distinct tokens, ``zeros`` of them with probability 0 (a nucleus cut)."""

    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(size, generator=generator, dtype=torch.float64) + 0.05
    if zeros:
        weights[-zeros:] = 0.0
    tokens = torch.randperm(50, generator=generator)[:size].to(torch.int64)
    return Qwen38RowDistribution(tokens, (weights / weights.sum()).to(dtype))


def _draw_law(distribution: Qwen38RowDistribution) -> dict[int, float]:
    """P(draw(u) = token) integrated exactly over u in [0, 1): the inverse CDF's intervals, the last positive
    token absorbing the tail the clip sends to it."""

    cumulative = torch.cumsum(distribution.probabilities, dim=-1)
    last_positive = int(torch.nonzero(distribution.probabilities > 0).max())
    law: dict[int, float] = {}
    previous = 0.0
    for index, token in enumerate(distribution.tokens.tolist()):
        if index > last_positive:
            law[token] = 0.0
            continue
        upper = 1.0 if index == last_positive else float(cumulative[index])
        law[token] = max(upper - previous, 0.0)
        previous = upper
    return law


def _row_law(distribution: Qwen38RowDistribution, draft: int) -> dict[int, float]:
    """The algorithm's output law on one row: accept the draft with p(d), else draw from the residual."""

    accept = distribution.probability(draft)
    law = {token: 0.0 for token in distribution.tokens.tolist()}
    law[draft] = law.get(draft, 0.0) + accept
    if accept < 1.0:
        for token, probability in _draw_law(distribution.without(draft)).items():
            law[token] += (1.0 - accept) * probability
    return law


def _pass_law(rows: list[Qwen38RowDistribution], drafts: list[int]) -> dict[tuple[int, ...], float]:
    """The joint law of the emitted tokens ``[d_1 .. d_a, x]`` of one pass, integrated exactly over the draws."""

    law: dict[tuple[int, ...], float] = {}
    survive = 1.0
    for row, draft in enumerate(drafts):
        accept = rows[row].probability(draft)
        if accept < 1.0:
            for token, probability in _draw_law(rows[row].without(draft)).items():
                if token == draft:
                    continue  # the rejected draft is never re-drawn: an accepted draft continues to the next row
                law[(*drafts[:row], token)] = (
                    law.get((*drafts[:row], token), 0.0) + survive * (1.0 - accept) * probability
                )
        survive *= accept
    for token, probability in _draw_law(rows[len(drafts)]).items():
        law[(*drafts, token)] = law.get((*drafts, token), 0.0) + survive * probability
    return law


@pytest.mark.parametrize("zeros", [0, 2])
def test_one_row_law_is_the_target_for_every_draft_in_and_out_of_the_kept_set(expect_error, zeros: int) -> None:
    for seed in range(24):
        distribution = _distribution(seed, zeros=zeros)
        tokens = distribution.tokens.tolist()
        for draft in [*tokens, 1000 + seed]:  # every kept token, a zero-probability token, a token never read
            law = _row_law(distribution, draft)
            assert sum(law.values()) == pytest.approx(1.0, abs=EXACT)
            for token in tokens:
                assert abs(law[token] - distribution.probability(token)) < EXACT, (seed, draft, token)
            if draft not in tokens:
                assert law[draft] == 0.0  # a draft outside the kept set is never emitted
            elif distribution.probability(draft) == 0.0:
                assert law[draft] == 0.0
    # A kept set of one token: p(d) = 1, the draft is always accepted and no residual is ever formed.
    one = Qwen38RowDistribution(torch.tensor([7]), torch.tensor([1.0], dtype=torch.float64))
    assert spec.accept_point_mass(lambda _: one, [7], lambda: 0.999999).accepted == 1
    with expect_error(ValueError, match="no probability mass"):
        one.without(7)


@pytest.mark.parametrize("k", mtp_v2.SUPPORTED_DRAFTS)
def test_pass_law_is_the_product_of_the_target_conditionals(k: int) -> None:
    for seed in range(8):
        rows = [_distribution(100 * seed + row, size=5, zeros=row % 2) for row in range(k + 1)]
        # Drafts inside and outside the kept sets, some with probability 0.
        drafts = [rows[row].tokens[(seed + row) % 5].item() if (seed + row) % 3 else 900 + row for row in range(k)]
        law = _pass_law(rows, drafts)
        assert sum(law.values()) == pytest.approx(1.0, abs=EXACT)
        for outcome, probability in law.items():
            expected = 1.0
            for row, token in enumerate(outcome):
                expected *= rows[row].probability(token)
            assert abs(probability - expected) < EXACT, (k, seed, outcome)
        # The first emitted token's marginal is p_0 (the plain sampler's law at that position).
        first = {}
        for outcome, probability in law.items():
            first[outcome[0]] = first.get(outcome[0], 0.0) + probability
        for token in rows[0].tokens.tolist():
            assert abs(first.get(token, 0.0) - rows[0].probability(token)) < EXACT


def test_fp32_rows_keep_the_law_within_the_single_precision_renormalisation() -> None:
    for seed in range(12):
        distribution = _distribution(seed, size=8, zeros=1, dtype=torch.float32)
        for draft in distribution.tokens.tolist():
            law = _row_law(distribution, draft)
            for token in distribution.tokens.tolist():
                assert abs(law[token] - distribution.probability(token)) < 2e-6


# --- the acceptance function: draws, counts, determinism ---------------------------------------------------------


def _uniforms(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return lambda: float(torch.rand((), generator=generator, dtype=torch.float32))


@pytest.mark.parametrize("k", mtp_v2.SUPPORTED_DRAFTS)
def test_accept_point_mass_draw_count_is_a_plus_two_or_k_plus_one_and_seeds_reproduce(expect_error, k: int) -> None:
    rows = [_distribution(7 + row, size=6) for row in range(k + 1)]
    drafts = [int(rows[row].tokens[0]) for row in range(k)]  # the most likely token of each row
    seen = set()
    for seed in range(200):
        counter = [0]
        uniform = _uniforms(seed)

        def counted() -> float:
            counter[0] += 1
            return uniform()

        result = spec.accept_point_mass(lambda row: rows[row], drafts, counted)
        assert result.draws == counter[0] == (result.accepted + 2 if result.resampled else k + 1)
        assert result.resampled == (result.accepted < k)
        assert len(result.acceptance_probabilities) == min(result.accepted + 1, k)
        assert result.acceptance_probabilities == tuple(
            rows[row].probability(drafts[row]) for row in range(len(result.acceptance_probabilities))
        )
        if result.resampled:
            assert result.token != drafts[result.accepted]  # a rejected draft is never re-drawn
        assert rows[result.accepted].probability(result.token) > 0
        seen.add(result.accepted)
        again = spec.accept_point_mass(lambda row: rows[row], drafts, _uniforms(seed))
        assert (again.accepted, again.token, again.draws) == (result.accepted, result.token, result.draws)
    assert len(seen) > 1  # the drafts are accepted sometimes and rejected sometimes
    outcomes = {
        (result.accepted, result.token)
        for result in (spec.accept_point_mass(lambda row: rows[row], drafts, _uniforms(seed)) for seed in range(40))
    }
    assert len(outcomes) > 1  # different seeds give different verdicts
    with expect_error(ValueError, match="at least one draft"):
        spec.accept_point_mass(lambda row: rows[row], [], _uniforms(0))


def test_monte_carlo_first_token_frequencies_match_the_target_within_five_sigma() -> None:
    k = 4
    rows = [_distribution(31 + row, size=6, dtype=torch.float32) for row in range(k + 1)]
    drafts = [int(rows[row].tokens[1]) for row in range(k)]
    trials = 20_000
    counts: dict[int, int] = {}
    accepted_total = 0
    uniform = _uniforms(2026)
    for _ in range(trials):
        result = spec.accept_point_mass(lambda row: rows[row], drafts, uniform)
        emitted = [*drafts[: result.accepted], result.token]
        counts[emitted[0]] = counts.get(emitted[0], 0) + 1
        accepted_total += result.accepted
    for token in rows[0].tokens.tolist():
        expected = rows[0].probability(token) * trials
        sigma = math.sqrt(max(expected * (1 - rows[0].probability(token)), 1.0))
        assert abs(counts.get(token, 0) - expected) < 5 * sigma, (
            token,
            counts.get(token, 0),
            expected,
        )  # tolerance: 5 sigma
    # The expected accepted length is the chain of the acceptance probabilities (the design's E[a*]).
    expected_accepted = 0.0
    survive = 1.0
    for row in range(k):
        survive *= rows[row].probability(drafts[row])
        expected_accepted += survive
    assert abs(accepted_total / trials - expected_accepted) < 0.05


# --- the row distributions: the samplers before their draw ----------------------------------------------------------


def _logits(seed: int, *, peaks: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.5 + 2.0
    ids = torch.randperm(PEAK_LIMIT, generator=generator)[:peaks]
    row[ids] = 16.0 + torch.rand(peaks, generator=generator) * 8.0
    return row.to(torch.bfloat16)


def _custom(**fields) -> Qwen38SamplingParameters:
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0, "seed": 0}
    base.update(fields)
    return Qwen38SamplingParameters(**base)


PROFILES = {
    "thinking": Qwen38SamplingParameters.official_thinking(seed=3),
    "non_thinking": Qwen38SamplingParameters.official_non_thinking(seed=5),
    "top_k_limit_min_p": _custom(top_k=32, top_p=0.9, min_p=0.05, temperature=0.8, seed=7),
    "top_k_one": _custom(top_k=1, temperature=2.0, seed=9),
    "frequency_repetition": _custom(top_k=30, temperature=0.9, frequency_penalty=0.3, repetition_penalty=1.2, seed=11),
    "cold": _custom(top_k=20, top_p=0.5, temperature=0.3, seed=13),
}


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_candidate_distribution_is_the_full_distribution_and_the_samplers_draw_from_it(
    expect_error, profile: str
) -> None:
    parameters = PROFILES[profile]
    exact = 0
    for seed in range(12):
        bf16 = _logits(seed)
        full = bf16.to(torch.float32)
        row = Qwen38CandidateRow.emulate(bf16)
        history = tuple(full.topk(6).indices.tolist()) * 2 if parameters.penalizes else ()
        prompt_tokens = min(3, len(history))
        try:
            candidates = candidate_distribution(row, parameters, token_history=history, prompt_tokens=prompt_tokens)
        except Qwen38CandidateFallback:
            with expect_error(Qwen38CandidateFallback):  # exactly where the sampler falls back
                sample_candidates(row, parameters, token_history=history, prompt_tokens=prompt_tokens)
            continue
        exact += 1
        reference = full_distribution(full, parameters, token_history=history, prompt_tokens=prompt_tokens)
        kept = reference.probabilities > 0
        assert torch.equal(candidates.tokens[candidates.probabilities > 0], reference.tokens[kept])
        assert torch.equal(candidates.probabilities[candidates.probabilities > 0], reference.probabilities[kept])
        # The samplers' token is the distribution's draw at their uniform (the request's generator, its seed).
        generator = torch.Generator().manual_seed(parameters.seed)
        sampled = sample_candidates(
            row, parameters, token_history=history, prompt_tokens=prompt_tokens, generator=generator
        )
        assert sampled.token_id == candidates.draw(sampled.uniform) == reference.draw(sampled.uniform)
        generator = torch.Generator().manual_seed(parameters.seed)
        assert (
            sample_full_vocabulary(
                full, parameters, token_history=history, prompt_tokens=prompt_tokens, generator=generator
            ).token_id
            == sampled.token_id
        )
    assert exact >= 10


def test_distribution_functions_refuse_temperature_zero_and_the_unbounded_candidate_cases(expect_error) -> None:
    row = Qwen38CandidateRow.emulate(_logits(1))
    greedy = _custom(temperature=0.0, top_k=1)
    with expect_error(ValueError, match="temperature > 0"):
        candidate_distribution(row, greedy)
    with expect_error(ValueError, match="temperature > 0"):
        full_distribution(_logits(1).to(torch.float32), greedy)
    with expect_error(Qwen38CandidateFallback, match="top_k 0"):
        candidate_distribution(row, _custom(top_k=0))
    with expect_error(Qwen38CandidateFallback, match="raises logits"):
        candidate_distribution(row, _custom(presence_penalty=-0.5), token_history=(1,))
    with expect_error(sampling.Qwen38SamplingError, match="exceeds the candidate limit"):
        candidate_distribution(row, _custom(top_k=33))


def test_row_distribution_validates_and_conditions(expect_error) -> None:
    distribution = _distribution(3, size=4)
    for token in distribution.tokens.tolist():
        residual = distribution.without(token)
        assert residual.probability(token) == 0.0 and residual.probabilities.sum() == pytest.approx(1.0, abs=EXACT)
        assert torch.equal(residual.tokens, distribution.tokens)
    assert distribution.probability(12345) == 0.0
    assert distribution.draw(0.0) == int(distribution.tokens[0]) and distribution.draw(0.999999) == int(
        distribution.tokens[-1]
    )
    with expect_error(ValueError, match="nonempty int64"):
        Qwen38RowDistribution(torch.tensor([], dtype=torch.int64), torch.tensor([]))
    with expect_error(ValueError, match="floating vector"):
        Qwen38RowDistribution(torch.tensor([1, 2]), torch.tensor([1.0]))
    with expect_error(ValueError, match="nonnegative"):
        Qwen38RowDistribution(torch.tensor([1, 2]), torch.tensor([1.5, -0.5]))
    with expect_error(ValueError, match="positive mass"):
        Qwen38RowDistribution(torch.tensor([1, 2]), torch.tensor([0.0, 0.0]))
    with expect_error(ValueError, match="every token once"):
        Qwen38RowDistribution(torch.tensor([1, 1]), torch.tensor([0.5, 0.5]))


# --- accept_pass: the per-row histories, the fallback, the ledger ------------------------------------------------------


class _FakeChain:
    def __init__(self, full_rows: torch.Tensor) -> None:
        self.full_rows = full_rows
        self.full_reads = 0

    def mtp_read_full_logits_rows(self) -> torch.Tensor:
        self.full_reads += 1
        return self.full_rows


def _head(rows_logits: list[torch.Tensor], drafts: list[int]) -> mtp_v2.Qwen38TTNNVerifyHeadReadback:
    argmaxes = [int(torch.argmax(logits.to(torch.float32))) for logits in rows_logits]
    accepted = 0
    while accepted < len(drafts) and argmaxes[accepted] == drafts[accepted]:
        accepted += 1
    candidate_rows = torch.stack(
        [Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits]
    )
    return mtp_v2.Qwen38TTNNVerifyHeadReadback(accepted, argmaxes[accepted], tuple(argmaxes), candidate_rows)


def test_accept_pass_uses_the_committed_stream_plus_the_earlier_rows_as_each_rows_history() -> None:
    k = 4
    parameters = _custom(presence_penalty=1.5, frequency_penalty=0.5, repetition_penalty=1.1, top_k=20, seed=17)
    prompt = [5, 6, 7]
    committed = prompt + [11, 12]
    rows_logits = [_logits(200 + row, peaks=8) for row in range(k + 1)]
    tokens = [int(torch.argmax(rows_logits[0].float()))] + [
        int(torch.argmax(rows_logits[row].float())) for row in range(1, k + 1)
    ]
    tokens[1] = tokens[2]  # a draft repeated inside the pass (row 2's own argmax): row 2's history counts it twice
    session = SimpleNamespace(committed=committed, chain=_FakeChain(torch.stack([l.float() for l in rows_logits])))
    request = step.Qwen38SamplingRequest(parameters)
    head = _head(rows_logits, tokens[1:])
    decision = step.accept_pass(session, request, len(prompt), tokens, head)
    # The re-derivation: the same rows through the pure functions with the design's histories and draws.
    generator = torch.Generator().manual_seed(parameters.seed)
    expected = spec.accept_point_mass(
        lambda row: candidate_distribution(
            Qwen38CandidateRow.emulate(rows_logits[row]),
            parameters,
            token_history=committed + tokens[: row + 1],
            prompt_tokens=len(prompt),
        ),
        tokens[1:],
        lambda: float(torch.rand((), generator=generator, dtype=torch.float32)),
    )
    assert (decision.accepted, decision.next_token) == (expected.accepted, expected.token)
    assert decision.alignment_tokens == (*tokens[1 : expected.accepted + 1], expected.token) + (
        ZERO_EMBEDDING_TOKEN,
    ) * (k - expected.accepted)
    assert decision.statistics["draws"] == expected.draws and decision.statistics["fallbacks"] == 0
    assert request.mtp.passes == 1 and request.mtp.draws == expected.draws and session.chain.full_reads == 0
    assert list(request.mtp.acceptance_probabilities) == list(expected.acceptance_probabilities)
    # Row 2's history holds the draft twice: its presence and frequency penalties lower that token's probability.
    plain = candidate_distribution(
        Qwen38CandidateRow.emulate(rows_logits[2]),
        _custom(top_k=20, seed=17),
        token_history=committed + tokens[:3],
        prompt_tokens=len(prompt),
    )
    penalized = candidate_distribution(
        Qwen38CandidateRow.emulate(rows_logits[2]),
        parameters,
        token_history=committed + tokens[:3],
        prompt_tokens=len(prompt),
    )
    assert penalized.probability(tokens[1]) < plain.probability(tokens[1])
    # The prompt's tokens are exempt from the additive penalties but not from the repetition rule.
    exempt = candidate_distribution(
        Qwen38CandidateRow.emulate(rows_logits[0]),
        parameters,
        token_history=committed + tokens[:1],
        prompt_tokens=len(committed) + 1,
    )
    assert exempt.probability(tokens[0]) >= candidate_distribution(
        Qwen38CandidateRow.emulate(rows_logits[0]), parameters, token_history=committed + tokens[:1], prompt_tokens=0
    ).probability(tokens[0])
    assert (
        request.as_dict()["mtp"] is None
    )  # the session marks the request drafted; until then the counters are not reported
    request.mtp_drafting = "drafted"
    reported = request.as_dict()["mtp"]
    assert reported["passes"] == 1 and reported["rows_drawn"] == len(expected.acceptance_probabilities)
    assert sum(reported["acceptance_probability_histogram"]) == reported["rows_drawn"]


def test_accept_pass_falls_back_to_the_full_rows_exactly_where_the_candidate_guard_fails(expect_error) -> None:
    k = 3
    parameters = _custom(top_k=32, seed=23)
    rows_logits = [_logits(300 + row, peaks=8) for row in range(k + 1)]
    tied = rows_logits[1].clone()
    tied[:32] = 25.0  # shard 0's k-th value ties with unread ids: the guard fails on row 1
    tied[32:42] = 25.0
    rows_logits[1] = tied
    tokens = [int(torch.argmax(l.float())) for l in rows_logits]
    rows_logits[0][tokens[1]] = 60.0  # row 0 all but surely accepts d_1, so row 1 is built and falls back
    session = SimpleNamespace(committed=[1, 2], chain=_FakeChain(torch.stack([l.float() for l in rows_logits])))
    request = step.Qwen38SamplingRequest(parameters)
    with expect_error(Qwen38CandidateFallback):
        candidate_distribution(Qwen38CandidateRow.emulate(tied), parameters, token_history=[1, 2, *tokens[:2]])
    decision = step.accept_pass(session, request, 2, tokens, _head(rows_logits, tokens[1:]))
    assert decision.accepted >= 1 and decision.statistics["fallbacks"] == 1 == session.chain.full_reads
    generator = torch.Generator().manual_seed(parameters.seed)

    def reference(row: int) -> Qwen38RowDistribution:
        history = [1, 2, *tokens[: row + 1]]
        try:
            return candidate_distribution(
                Qwen38CandidateRow.emulate(rows_logits[row]), parameters, token_history=history, prompt_tokens=2
            )
        except Qwen38CandidateFallback:
            return full_distribution(rows_logits[row].float(), parameters, token_history=history, prompt_tokens=2)

    expected = spec.accept_point_mass(
        reference, tokens[1:], lambda: float(torch.rand((), generator=generator, dtype=torch.float32))
    )
    assert (decision.accepted, decision.next_token) == (expected.accepted, expected.token)
    assert request.mtp.fallbacks == 1 and decision.statistics["draws"] == expected.draws
    with expect_error(ValueError, match="candidate rows"):
        step.accept_pass(session, request, 2, tokens[:-1], _head(rows_logits, tokens[1:]))


def test_request_draw_is_the_samplers_uniform_stream() -> None:
    request = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=99))
    generator = torch.Generator().manual_seed(99)
    draws = [request.draw() for _ in range(5)]
    assert draws == [float(torch.rand((), generator=generator, dtype=torch.float32)) for _ in range(5)]
    assert all(0.0 <= draw < 1.0 for draw in draws)
    assert step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=98)).draw() != draws[0]


def test_drafting_admission_names_the_refused_requests() -> None:
    sampled = SimpleNamespace(sampled=True)
    off = SimpleNamespace(sampled=False)
    thinking = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=1))
    assert step.drafting_admission(sampled, None) is None and step.drafting_admission(None, None) is None
    assert step.drafting_admission(None, thinking) == "refused: no MTP chain"
    assert step.drafting_admission(off, thinking) == "refused: no split verify captured"
    assert step.drafting_admission(sampled, thinking) is None
    assert (
        step.drafting_admission(
            sampled, step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_non_thinking(seed=1))
        )
        is None
    )
    assert step.drafting_admission(sampled, thinking, device_loop=True) == "refused: device sampler loop"
    assert step.drafting_admission(sampled, step.Qwen38SamplingRequest(_custom(top_k=0))) == "refused: top_k 0"
    assert (
        step.drafting_admission(sampled, step.Qwen38SamplingRequest(_custom(presence_penalty=-0.5)))
        == "refused: penalty raises logits"
    )
    assert (
        step.drafting_admission(sampled, step.Qwen38SamplingRequest(_custom(repetition_penalty=0.9)))
        == "refused: penalty raises logits"
    )
    assert (
        step.drafting_admission(sampled, step.Qwen38SamplingRequest(thinking.parameters, logprobs=True))
        == "refused: logprobs"
    )
    assert (
        step.drafting_admission(sampled, step.Qwen38SamplingRequest(thinking.parameters, top_logprobs=2))
        == "refused: logprobs"
    )


# --- the greedy host decision and the head readback -------------------------------------------------------------------

SMALL_IDS = (17, 15, 16, 21, 12, 20, 11, 0, 2047)
LARGE_IDS = (2048, 95859, 62086, 248044, 248319)


def _pattern(pattern: tuple[int, ...], ids: tuple[int, ...]) -> tuple[list[int], list[int]]:
    cycle = itertools.cycle(ids)
    targets = [next(cycle) for _ in range(len(pattern) + 1)]
    drafts = []
    for row, match in enumerate(pattern):
        drafts.append(targets[row] if match else next(value for value in ids if value != targets[row]))
    return targets, drafts


@pytest.mark.parametrize("k", mtp_v2.SUPPORTED_DRAFTS)
def test_decide_greedy_equals_the_device_accept_model_for_every_pattern(expect_error, k: int) -> None:
    for pattern in itertools.product((0, 1), repeat=k):
        targets, drafts = _pattern(pattern, SMALL_IDS + LARGE_IDS)
        reference = accept_select(
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(drafts, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        head = mtp_v2.Qwen38TTNNVerifyHeadReadback(
            reference["accepted"], reference["next_token_gather"], tuple(targets), torch.zeros(k + 1, 256)
        )
        decision = mtp_v2.decide_greedy([targets[0] + 1, *drafts], head)
        assert (decision.accepted, decision.next_token) == (reference["accepted"], reference["next_token_gather"])
        assert decision.alignment_tokens == tuple(targets)  # the fused body's alignment tokens: the argmax lanes
        assert decision.statistics == {"accept_checks": 1}
        wrong = mtp_v2.Qwen38TTNNVerifyHeadReadback(
            (reference["accepted"] + 1) % (k + 1),
            reference["next_token_gather"],
            tuple(targets),
            torch.zeros(k + 1, 256),
        )
        with expect_error(RuntimeError, match="host greedy accept"):
            mtp_v2.decide_greedy([targets[0] + 1, *drafts], wrong)
    with expect_error(ValueError, match="verify rows"):
        mtp_v2.decide_greedy([1, 2], mtp_v2.Qwen38TTNNVerifyHeadReadback(0, 1, (1, 2, 3), torch.zeros(3, 256)))


@pytest.mark.parametrize("k", mtp_v2.SUPPORTED_DRAFTS)
def test_head_readback_parses_and_its_width_is_pinned(expect_error, k: int) -> None:
    rows = k + 1
    assert mtp_v2.head_readback_width(rows) == 2 + 32 + rows * 256
    assert mtp_v2.CANDIDATE_LANES_PER_ROW == SAMPLING_CANDIDATE_ROW_SHAPE[3] == 256
    assert mtp_v2.HEAD_READBACK_FIXED_LANES == ("accepted", "next_token")
    rows_logits = [_logits(400 + row, peaks=8) for row in range(rows)]
    candidate_rows = torch.stack(
        [Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits]
    )
    argmaxes = [int(torch.argmax(logits.float())) for logits in rows_logits] + [ZERO_EMBEDDING_TOKEN] * (32 - rows)
    values = torch.cat(
        [
            torch.tensor([2.0, float(argmaxes[2])]),
            torch.tensor(argmaxes, dtype=torch.float32),
            candidate_rows.reshape(-1),
        ]
    )
    assert values.numel() == mtp_v2.head_readback_width(rows) and values.numel() * 4 == 4 * (34 + rows * 256)
    parsed = mtp_v2._verify_head_readback(values, rows=rows)
    assert (parsed.accepted, parsed.next_token) == (2, argmaxes[2]) and parsed.argmaxes == tuple(argmaxes[:rows])
    assert torch.equal(parsed.candidate_rows, candidate_rows)
    for row, logits in enumerate(rows_logits):
        parsed_row = Qwen38CandidateRow.from_host_row(parsed.candidate_rows[row].reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
        assert torch.equal(parsed_row.ids, Qwen38CandidateRow.emulate(logits).ids)
    with expect_error(RuntimeError, match="lanes"):
        mtp_v2._verify_head_readback(values[:-1], rows=rows)


def test_write_verify_decision_validates_before_any_device_write(expect_error) -> None:
    verify = SimpleNamespace(split=object(), rows=5)
    model = SimpleNamespace(mesh_device=None)
    good = mtp_v2.Qwen38TTNNVerifyDecision(1, 9, (8, 9, -1, -1, -1))
    for decision, message in (
        (mtp_v2.Qwen38TTNNVerifyDecision(5, 9, (8, 9, -1, -1, -1)), "accept count"),
        (mtp_v2.Qwen38TTNNVerifyDecision(1, -2, (8, 9, -1, -1, -1)), "next token"),
        (mtp_v2.Qwen38TTNNVerifyDecision(1, 9, (8, 9, -1, -1)), "alignment tokens"),
        (mtp_v2.Qwen38TTNNVerifyDecision(1, 9, (8, 7, -1, -1, -1)), "is not the next token"),
    ):
        with expect_error(ValueError, match=message):
            mtp_v2.write_verify_decision(model, verify, decision)
    with expect_error(ValueError, match="no split buffers"):
        mtp_v2.write_verify_decision(model, SimpleNamespace(split=None, rows=5), good)


# --- the split verify traces and the forms they count --------------------------------------------------------------


def test_traces_admit_the_split_form_only_with_the_commit_form(expect_error) -> None:
    traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=2, commit=3, verify_head=4, verify_tail=5)
    assert traces.split and traces.ids() == [4, 5, 3, 2]
    fused = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
    assert not fused.split and fused.ids() == [1, 3, 2]
    for kwargs in (
        {"verify_first": 1, "draft": 2, "commit": 3, "verify_head": 4, "verify_tail": 5},  # two verify forms
        {"verify_first": None, "draft": 2, "commit": 3},  # no verify form
        {"verify_first": None, "draft": 2, "commit": 3, "verify_head": 4},  # a head without its tail
        {"verify_first": None, "draft": 2, "verify_catch_up": 1, "verify_head": 4, "verify_tail": 5},  # no commit
    ):
        with expect_error(ValueError):
            mtp_v2.Qwen38TTNNMTPTraces(**kwargs)


def test_verify_forms_table_and_the_default_records_capacity_admission(expect_error) -> None:
    """The admission counts the verify forms a chain captures, from one table keyed by (mtp_sampled,
    mtp_device_accept): the fused verify alone, or the fused verify and the split head + tail beside it."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module

    assert session_module.MTP_VERIFY_FORMS_BY_SWITCH[(False, False)] == ("fused",)
    assert session_module.MTP_VERIFY_FORMS_BY_SWITCH[(True, False)] == ("fused", "split")
    assert session_module.mtp_verify_forms(False) == ("fused",) and session_module.mtp_verify_forms(True) == (
        "fused",
        "split",
    )
    with expect_error(ValueError, match="must be a bool"):
        session_module.mtp_verify_forms(1)
    # The default configuration's record (k = 4, both forms) fits the estimate.
    record = session_module.mtp_capacity_admission(32768, drafts=4, verify_forms=2)
    assert record["verify_forms"] == 2 and record["mtp_growth_remainders_bytes_per_bank"]["traces"] == 11_107_904
    assert record["mtp_growth_remainders_bytes_per_bank"]["traces_per_additional_verify_form"] == 4_717_760
    assert record["fits"] and record["required_free_bytes_per_bank"] >= 74_514_048
    assert session_module.mtp_capacity_admission(32768, drafts=4, verify_forms=1)["required_free_bytes_per_bank"] < (
        record["required_free_bytes_per_bank"]
    )
