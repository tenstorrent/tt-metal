# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The batched candidate distributions of one MTP pass, without a device.

``candidate_distributions`` builds the k + 1 verify rows' distributions in one batched pass; here every row is the
per-row ``candidate_distribution`` bitwise (positions and probabilities, fp32 and float64 rows) for every processor the
server admits and every history shape a pass produces, falls back on exactly the rows the per-row function falls back
on (the shard-floor guard, ``top_k`` 0, a boosting penalty) with its messages, and refuses what it refuses.  On those
rows ``accept_pass`` is today's per-row acceptance to the draw: the same accepted count, tokens, draws, fallbacks and
acceptance probabilities for every seed, the generator left in the same state, the verify logits read for the same
passes.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import speculative_sampling as spec
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    SAMPLING_CANDIDATES_PER_DEVICE,
    VOCAB_SIZE,
    ZERO_EMBEDDING_TOKEN,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateDistributions,
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38CandidateRows,
    Qwen38RowDistribution,
    Qwen38SamplingError,
    Qwen38SamplingParameters,
    candidate_distribution,
    candidate_distributions,
    full_distribution,
)

K = 4
PEAK_LIMIT = 200_000  # peaks below the tokenizer size


# --- synthetic passes ---------------------------------------------------------------------------------------------------


def _logits(seed: int, *, peaks: int = 24, lift: float = 16.0, spread: float = 8.0, shift: float = 2.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.5 + shift
    ids = torch.randperm(PEAK_LIMIT, generator=generator)[:peaks]
    row[ids] = lift + torch.rand(peaks, generator=generator) * spread
    return row.to(torch.bfloat16)


def _tie_row(seed: int) -> torch.Tensor:
    """Shard 0's k-th value ties with unread ids: the guard fails at ``top_k`` 32 (and below, the kept minimum sits at
    the floor whenever the tie fills the kept set)."""

    row = _logits(seed)
    row[: SAMPLING_CANDIDATES_PER_DEVICE + 10] = 25.0
    return row


def _custom(**fields) -> Qwen38SamplingParameters:
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0, "seed": 0}
    base.update(fields)
    return Qwen38SamplingParameters(**base)


def _pass_logits(seed: int, *, rows: int = K + 1, **fields) -> list[torch.Tensor]:
    return [_logits(seed * 100 + row, **fields) for row in range(rows)]


def _negative_pass_logits(seed: int) -> list[torch.Tensor]:
    """Rows whose every candidate is negative (the noise around -40, forty peaks in [-20, -12]: about ten per shard, so
    the shards' top-32 hold them all and the kept set stays above the floor): the repetition rule multiplies them."""

    return _pass_logits(seed, peaks=40, lift=-20.0, spread=8.0, shift=-40.0)


def _candidate_rows(rows_logits: list[torch.Tensor]) -> Qwen38CandidateRows:
    host = torch.stack([Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits])
    return Qwen38CandidateRows.from_host_rows(host)


def _argmaxes(rows_logits: list[torch.Tensor]) -> list[int]:
    return [int(torch.argmax(logits.to(torch.float32))) for logits in rows_logits]


def _argmax_drafts(rows_logits: list[torch.Tensor]) -> list[int]:
    """The pass tokens ``[t_P, d_1 .. d_k]`` with every draft the argmax of the row that judges it (row j proposes
    ``d_{j + 1}``): the accepted length then follows the rows' argmax probabilities."""

    argmaxes = _argmaxes(rows_logits)
    return [argmaxes[-1], *argmaxes[:-1]]


def _runner_up(logits: torch.Tensor) -> int:
    return int(torch.topk(logits.to(torch.float32), 2).indices[1])


def _committed(rows: Qwen38CandidateRows, tokens: list[int]) -> list[int]:
    """A committed stream whose tokens the rows' candidates hold (the penalties bite): candidates of several rows and
    shards, a pass token, a repeated token, and ids no row reads."""

    return [
        *rows.ids[0, 0, :2].tolist(),
        *rows.ids[3, 1, :3].tolist(),
        5,
        6,
        7,
        tokens[1],
        tokens[1],
        rows.ids[4, 2, 0].item(),
    ]


# The processors the server admits to the pass loop (temperature > 0, top_k 1..32, nucleus, min-p, non-boosting
# penalties), the two card profiles first.
PROCESSORS = {
    "thinking": Qwen38SamplingParameters.official_thinking(seed=3),
    "non_thinking": Qwen38SamplingParameters.official_non_thinking(seed=5),
    "top_k_1": _custom(top_k=1, temperature=2.0, seed=9),
    "top_k_32_nucleus_min_p": _custom(top_k=32, top_p=0.9, min_p=0.05, temperature=0.8, seed=7),
    "cold_nucleus": _custom(top_k=20, top_p=0.5, temperature=0.3, seed=13),
    "presence": _custom(top_k=20, presence_penalty=0.9, seed=15),
    "frequency": _custom(top_k=24, frequency_penalty=0.4, temperature=1.1, seed=17),
    "repetition": _custom(top_k=28, repetition_penalty=1.25, seed=19),
    "all_penalties_min_p": _custom(
        top_k=30,
        top_p=0.95,
        min_p=0.02,
        temperature=0.9,
        presence_penalty=0.5,
        frequency_penalty=0.3,
        repetition_penalty=1.2,
        seed=21,
    ),
}


# --- the per-row reference and the bitwise comparison ------------------------------------------------------------------


def _reference(
    rows: Qwen38CandidateRows,
    parameters: Qwen38SamplingParameters,
    committed: list[int],
    tokens: list[int],
    prompt: int,
) -> list[Qwen38RowDistribution | Exception]:
    """``candidate_distribution`` per row with the pass's histories, the exception where it raises."""

    results: list[Qwen38RowDistribution | Exception] = []
    for row in range(rows.rows):
        try:
            results.append(
                candidate_distribution(
                    rows.row(row), parameters, token_history=[*committed, *tokens[: row + 1]], prompt_tokens=prompt
                )
            )
        except Exception as error:  # noqa: BLE001 - the reference's exception is the expectation
            results.append(error)
    return results


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32 if values.dtype == torch.float32 else torch.int64)


def _assert_bitwise(batched: Qwen38CandidateDistributions, reference: list) -> tuple[int, int]:
    """Every row equal to the bit, or the same exception with the same message; ``(exact rows, fallback rows)``."""

    exact = fallbacks = 0
    for row, expected in enumerate(reference):
        if isinstance(expected, Exception):
            with pytest.raises(type(expected)) as caught:  # allow-pytest.raises: inspect the captured exception object
                batched.row(row)
            assert type(caught.value) is type(expected) and str(caught.value) == str(expected), row
            fallbacks += isinstance(expected, Qwen38CandidateFallback)
            continue
        actual = batched.row(row)
        assert actual.tokens.dtype == expected.tokens.dtype and torch.equal(actual.tokens, expected.tokens), row
        assert actual.probabilities.dtype == expected.probabilities.dtype, row
        assert torch.equal(_bits(actual.probabilities), _bits(expected.probabilities)), row
        exact += 1
    return exact, fallbacks


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "float64"])
@pytest.mark.parametrize("processor", sorted(PROCESSORS))
def test_batched_rows_are_the_per_row_distributions_bitwise(processor: str, dtype: torch.dtype) -> None:
    parameters = PROCESSORS[processor]
    exact = fallbacks = 0
    for seed in range(6):
        rows_logits = _pass_logits(seed)
        if seed == 4:
            rows_logits[2] = _tie_row(seed * 100 + 2)  # one row the guard cannot bound
        rows = _candidate_rows(rows_logits)
        if dtype is torch.float64:
            rows = Qwen38CandidateRows(rows.values.to(torch.float64), rows.ids)
        tokens = _argmaxes(rows_logits)
        tokens[2] = tokens[1]  # a draft repeated inside the pass: rows 2.. count it twice
        committed = _committed(rows, tokens)
        # the prompt covers nothing, part of the stream, the whole stream, or the stream and the pass's first token
        for prompt in (0, 3, len(committed), len(committed) + 1):
            batched = candidate_distributions(
                rows, parameters, token_history=committed, row_tokens=tokens, prompt_tokens=prompt
            )
            assert batched.rows == K + 1 and batched.probabilities.dtype == dtype
            counts = _assert_bitwise(batched, _reference(rows, parameters, committed, tokens, prompt))
            exact, fallbacks = exact + counts[0], fallbacks + counts[1]
    assert exact >= 100 and fallbacks >= 4, (exact, fallbacks)  # the tie row falls back under every prompt shape


@pytest.mark.parametrize("processor", ["repetition", "all_penalties_min_p"])
def test_negative_candidate_scores_take_the_repetition_rules_multiplying_branch_bitwise(processor: str) -> None:
    parameters = PROCESSORS[processor]
    exact = 0
    for seed in range(4):
        rows_logits = _negative_pass_logits(60 + seed)
        rows = _candidate_rows(rows_logits)
        assert bool((rows.values < 0).all())  # every candidate score negative: a hit is multiplied, not divided
        tokens = _argmaxes(rows_logits)
        committed = _committed(rows, tokens)
        # the per-row reference lowers row 0's argmax (in its own history): the multiplying branch acts (under the
        # nucleus and min-p cut the penalized token may leave the kept set altogether)
        unpenalized = replace(parameters, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)
        plain = candidate_distribution(rows.row(0), unpenalized, token_history=[*committed, tokens[0]])
        penalized = candidate_distribution(rows.row(0), parameters, token_history=[*committed, tokens[0]])
        assert penalized.probability(tokens[0]) < plain.probability(tokens[0])
        for prompt in (0, 3, len(committed), len(committed) + 1):
            batched = candidate_distributions(
                rows, parameters, token_history=committed, row_tokens=tokens, prompt_tokens=prompt
            )
            exact += _assert_bitwise(batched, _reference(rows, parameters, committed, tokens, prompt))[0]
    assert exact >= 64, exact


def test_every_admitted_top_k_matches_the_per_row_function() -> None:
    rows_logits = _pass_logits(7)
    rows = _candidate_rows(rows_logits)
    tokens = _argmaxes(rows_logits)
    committed = _committed(rows, tokens)
    exact = 0
    for top_k in range(1, SAMPLING_CANDIDATES_PER_DEVICE + 1):
        for parameters in (
            _custom(top_k=top_k),
            _custom(top_k=top_k, top_p=0.85, min_p=0.03, temperature=0.75, presence_penalty=0.6, seed=top_k),
            _custom(top_k=top_k, frequency_penalty=0.2, repetition_penalty=1.1, temperature=1.3, seed=top_k),
        ):
            batched = candidate_distributions(
                rows, parameters, token_history=committed, row_tokens=tokens, prompt_tokens=3
            )
            exact += _assert_bitwise(batched, _reference(rows, parameters, committed, tokens, 3))[0]
    assert exact >= 32 * 3 * 4  # at most one row per pass sits at the floor


def test_guard_failures_fall_back_on_the_same_rows_with_the_same_message() -> None:
    rows_logits = _pass_logits(30)
    rows_logits[1] = _tie_row(3001)
    rows_logits[3] = _tie_row(3003)
    rows = _candidate_rows(rows_logits)
    tokens = _argmaxes(rows_logits)
    for parameters in (_custom(top_k=32, seed=23), _custom(top_k=32, top_p=0.9, presence_penalty=1.0, seed=24)):
        batched = candidate_distributions(rows, parameters, token_history=[1, 2, 3], row_tokens=tokens, prompt_tokens=2)
        assert [isinstance(outcome, Qwen38CandidateFallback) for outcome in batched.outcomes] == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert _assert_bitwise(batched, _reference(rows, parameters, [1, 2, 3], tokens, 2)) == (3, 2)
        assert "shard floor" in str(batched.outcomes[1])


def test_refusals_and_whole_pass_fallbacks_match_the_per_row_function(expect_error) -> None:
    rows_logits = _pass_logits(8)
    rows = _candidate_rows(rows_logits)
    tokens = _argmaxes(rows_logits)
    committed = [1, 2, 3]
    for parameters in (_custom(top_k=0), _custom(presence_penalty=-0.5), _custom(frequency_penalty=-0.1)):
        batched = candidate_distributions(rows, parameters, token_history=committed, row_tokens=tokens, prompt_tokens=1)
        assert _assert_bitwise(batched, _reference(rows, parameters, committed, tokens, 1)) == (0, K + 1)
    with expect_error(Qwen38SamplingError, match="exceeds the candidate limit"):
        candidate_distributions(rows, _custom(top_k=33), token_history=committed, row_tokens=tokens)
    with expect_error(ValueError, match="temperature > 0"):
        candidate_distributions(rows, _custom(temperature=0.0), token_history=committed, row_tokens=tokens)
    with expect_error(ValueError, match="prompt_tokens must be an integer in \\[0, 4\\]"):
        candidate_distributions(rows, _custom(), token_history=committed, row_tokens=tokens, prompt_tokens=5)
    with expect_error(ValueError, match="prompt_tokens"):  # the per-row function raises for row 0 too
        candidate_distribution(rows.row(0), _custom(), token_history=[*committed, tokens[0]], prompt_tokens=5)
    with expect_error(ValueError, match="one token per row"):
        candidate_distributions(rows, _custom(), token_history=committed, row_tokens=tokens[:-1])
    with expect_error(TypeError, match="integer token IDs"):
        candidate_distributions(rows, _custom(), token_history=[1, 2.5], row_tokens=tokens)
    with expect_error(ValueError, match="outside"):
        candidate_distributions(rows, _custom(), token_history=[1, VOCAB_SIZE], row_tokens=tokens)
    with expect_error(TypeError, match="integer token IDs"):
        candidate_distributions(rows, _custom(), token_history=committed, row_tokens=[*tokens[:-1], True])
    with expect_error(TypeError, match="Qwen38CandidateRows"):
        candidate_distributions(rows.row(0), _custom(), token_history=committed, row_tokens=tokens)
    # an empty committed stream: the rows' histories are the pass's own tokens
    batched = candidate_distributions(rows, PROCESSORS["all_penalties_min_p"], row_tokens=tokens, prompt_tokens=0)
    assert _assert_bitwise(batched, _reference(rows, PROCESSORS["all_penalties_min_p"], [], tokens, 0))[0] >= K


def test_candidate_rows_parse_like_the_per_row_parser(expect_error) -> None:
    rows_logits = _pass_logits(9)
    host = torch.stack([Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits])
    rows = Qwen38CandidateRows.from_host_rows(host)
    assert rows.rows == K + 1 and rows.values.dtype == torch.float32 and rows.ids.dtype == torch.int64
    for row in range(K + 1):
        single = Qwen38CandidateRow.from_host_row(host[row].reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
        assert torch.equal(rows.row(row).values, single.values) and torch.equal(rows.row(row).ids, single.ids)
        assert torch.equal(rows.shard_floors[row], single.shard_floor)
    # every row is checked, not only the first: (the message, the corrupted row, the lane and its value)
    k = SAMPLING_CANDIDATES_PER_DEVICE
    for label, row, lane, value in (
        ("NaN", 3, 0, float("nan")),
        ("not integers", 2, k, 1.5),
        ("leave their shards", 4, k, float(VOCAB_SIZE)),
        ("repeat", 1, k + 1, float(host[1, k])),
        ("descending", 2, 1, float(host[2, 0]) + 1.0),
    ):
        bad = host.clone()
        bad[row, lane] = value
        with expect_error(ValueError, match=label):
            Qwen38CandidateRows.from_host_rows(bad)
        with expect_error(ValueError):
            Qwen38CandidateRow.from_host_row(bad[row].reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
        Qwen38CandidateRows.from_host_rows(torch.cat([bad[:row], bad[row + 1 :]]))  # the other rows are sound
    with expect_error(ValueError, match="fp32 \\[rows"):
        Qwen38CandidateRows.from_host_rows(host[0])
    with expect_error(ValueError, match="fp32 \\[rows"):
        Qwen38CandidateRows.from_host_rows(host.to(torch.float64))


def test_row_distribution_unchecked_construction_is_the_batched_builders_shortcut(expect_error) -> None:
    tokens, probabilities = torch.tensor([1, 1]), torch.tensor([0.5, 0.5])
    with expect_error(ValueError, match="every token once"):
        Qwen38RowDistribution(tokens, probabilities)
    assert Qwen38RowDistribution(tokens, probabilities, False).tokens is tokens
    rows_logits = _pass_logits(10)
    rows = _candidate_rows(rows_logits)
    batched = candidate_distributions(
        rows, PROCESSORS["thinking"], token_history=[1], row_tokens=_argmaxes(rows_logits)
    )
    for row in range(K + 1):  # what the builder hands out passes the validation it skipped
        distribution = batched.row(row)
        Qwen38RowDistribution(distribution.tokens, distribution.probabilities)


# --- accept_pass on the batched rows is the per-row acceptance to the draw ---------------------------------------------


class _FakeChain:
    def __init__(self, full_rows: torch.Tensor) -> None:
        self.full_rows = full_rows
        self.full_reads = 0

    def mtp_read_full_logits_rows(self) -> torch.Tensor:
        self.full_reads += 1
        return self.full_rows


def _head(rows_logits: list[torch.Tensor], drafts: list[int]) -> mtp_v2.Qwen38TTNNVerifyHeadReadback:
    argmaxes = _argmaxes(rows_logits)
    accepted = 0
    while accepted < len(drafts) and argmaxes[accepted] == drafts[accepted]:
        accepted += 1
    candidate_rows = torch.stack(
        [Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits]
    )
    return mtp_v2.Qwen38TTNNVerifyHeadReadback(accepted, argmaxes[accepted], tuple(argmaxes), candidate_rows)


def _accept_pass_per_row(session, request, prompt_tokens, tokens, head) -> mtp_v2.Qwen38TTNNVerifyDecision:
    """The per-row ``accept_pass`` this commit replaced, verbatim: the reference law of the pass decision."""

    rows = len(tokens)
    if tuple(head.candidate_rows.shape) != (rows, SAMPLING_CANDIDATE_ROW_SHAPE[3]):
        raise ValueError(f"{tuple(head.candidate_rows.shape)} candidate rows for a {rows}-row pass")
    committed = list(session.committed)
    fallbacks = 0
    full_rows: torch.Tensor | None = None

    def distribution(row: int) -> Qwen38RowDistribution:
        nonlocal fallbacks, full_rows
        history = committed + [int(token) for token in tokens[: row + 1]]
        candidate_row = Qwen38CandidateRow.from_host_row(head.candidate_rows[row].reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
        try:
            return candidate_distribution(
                candidate_row, request.parameters, token_history=history, prompt_tokens=prompt_tokens
            )
        except Qwen38CandidateFallback:
            fallbacks += 1
            if full_rows is None:
                full_rows = session.chain.mtp_read_full_logits_rows()
            return full_distribution(
                full_rows[row], request.parameters, token_history=history, prompt_tokens=prompt_tokens
            )

    acceptance = spec.accept_point_mass(distribution, tokens[1:], request.draw)
    request.mtp.record(acceptance, fallbacks)
    alignment = [int(token) for token in tokens[1 : acceptance.accepted + 1]] + [acceptance.token]
    alignment += [ZERO_EMBEDDING_TOKEN] * (rows - len(alignment))
    return mtp_v2.Qwen38TTNNVerifyDecision(
        acceptance.accepted,
        acceptance.token,
        tuple(alignment),
        {
            "sampled": True,
            "draws": acceptance.draws,
            "fallbacks": fallbacks,
            "resampled": acceptance.resampled,
            "acceptance_probabilities": acceptance.acceptance_probabilities,
        },
    )


def _same_decision(
    parameters: Qwen38SamplingParameters, committed: list[int], prompt: int, tokens: list[int], rows_logits: list
) -> tuple[mtp_v2.Qwen38TTNNVerifyDecision, int]:
    """Both implementations on the same pass from the same seed; the shared decision and the full-logit reads."""

    full_rows = torch.stack([logits.to(torch.float32) for logits in rows_logits])
    head = _head(rows_logits, tokens[1:])
    sessions = [SimpleNamespace(committed=list(committed), chain=_FakeChain(full_rows)) for _ in range(2)]
    requests = [step.Qwen38SamplingRequest(parameters) for _ in range(2)]
    new = step.accept_pass(sessions[0], requests[0], prompt, tokens, head)
    old = _accept_pass_per_row(sessions[1], requests[1], prompt, tokens, head)
    assert (new.accepted, new.next_token, new.alignment_tokens) == (old.accepted, old.next_token, old.alignment_tokens)
    assert dict(new.statistics) == dict(old.statistics)  # draws, fallbacks, resampled, acceptance probabilities
    assert torch.equal(requests[0].generator.get_state(), requests[1].generator.get_state())  # the same draws consumed
    assert requests[0].mtp == requests[1].mtp
    assert sessions[0].chain.full_reads == sessions[1].chain.full_reads == int(bool(new.statistics["fallbacks"]))
    return new, sessions[0].chain.full_reads


@pytest.mark.parametrize("processor", ["thinking", "non_thinking", "all_penalties_min_p", "top_k_1", "cold_nucleus"])
def test_accept_pass_is_the_per_row_acceptance_for_every_seed(processor: str) -> None:
    accepted_seen: set[int] = set()
    fallback_passes = 0
    for seed in range(48):
        parameters = replace(PROCESSORS[processor], seed=1000 + seed)
        rows_logits = _pass_logits(500 + seed, lift=14.0, spread=6.0)
        if seed % 5 == 0:
            rows_logits[3] = _tie_row(5000 + seed)  # a row the guard cannot bound, reached only past three accepts
        if seed % 7 == 0:
            rows_logits[1] = _tie_row(7000 + seed)
        tokens = _argmax_drafts(rows_logits)
        if seed % 3 == 0:
            tokens[2] = _runner_up(rows_logits[1])  # a weaker draft at row 1
        if seed % 4 == 1:
            tokens[4] = _runner_up(rows_logits[3])  # and at row 3
        rows = _candidate_rows(rows_logits)
        # candidates the penalties lower without touching the drafts; every sixth pass penalizes the first draft too
        committed = [4, 5, 6, *rows.ids[0, 0, 3:5].tolist(), *rows.ids[3, 1, 5:8].tolist(), 7, 8]
        if seed % 6 == 2:
            committed.append(tokens[1])
        prompt = (3, len(committed), len(committed) + 1)[seed % 3]
        decision, full_reads = _same_decision(parameters, committed, prompt, tokens, rows_logits)
        accepted_seen.add(decision.accepted)
        fallback_passes += full_reads
    assert len(accepted_seen) >= 3, accepted_seen  # rejections at several rows and passes accepting every draft
    assert fallback_passes >= 1


def test_a_fallback_row_past_the_first_rejection_is_neither_counted_nor_read() -> None:
    rows_logits = _pass_logits(40)
    rows_logits[2] = _tie_row(4002)
    tokens = _argmax_drafts(rows_logits)
    tokens[1] = PEAK_LIMIT + 1  # a draft no candidate row holds: row 0 rejects it with probability 1
    parameters = _custom(top_k=32, seed=41)
    rows = _candidate_rows(rows_logits)
    batched = candidate_distributions(rows, parameters, token_history=[1, 2], row_tokens=tokens, prompt_tokens=2)
    assert isinstance(batched.outcomes[2], Qwen38CandidateFallback)  # known to the builder
    decision, full_reads = _same_decision(parameters, [1, 2], 2, tokens, rows_logits)
    assert decision.accepted == 0 and decision.statistics["fallbacks"] == 0 and full_reads == 0  # never asked for
    assert decision.statistics["draws"] == 2 and decision.statistics["acceptance_probabilities"] == (0.0,)
