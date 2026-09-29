# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The candidate sampler equals the full-vocabulary sampler; the readback row is exact; the TAIL epilogue's op list."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    LOCAL_VOCAB_SIZE,
    SAMPLING_CANDIDATE_ROW_SHAPE,
    SAMPLING_CANDIDATES_PER_DEVICE,
    TILE_SIZE,
    TP_SIZE,
    VOCAB_SIZE,
    Qwen38TTNNLMHead,
    Qwen38TTNNSamplingCandidateConstants,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    CANDIDATE_TOP_K_LIMIT,
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38SamplingError,
    Qwen38SamplingParameters,
    Qwen38SamplingProfile,
    sample_candidates,
    sample_full_vocabulary,
    sample_host_logits,
)

K = SAMPLING_CANDIDATES_PER_DEVICE
PEAK_IDS = (95_859, 248_044, 248_319, 7, 62_086, 131_071)


def _logits(seed: int, *, peaks: int = 24) -> torch.Tensor:
    """A bf16 logit row at model scale: a wide body plus a few dozen peaks (some at the probe ids)."""

    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.5 + 2.0
    ids = torch.randperm(VOCAB_SIZE, generator=generator)[:peaks].tolist() + list(PEAK_IDS)
    row[ids] = 16.0 + torch.rand(len(ids), generator=generator) * 8.0  # above the body's maximum (about 13)
    return row.to(torch.bfloat16)


def _custom(**fields) -> Qwen38SamplingParameters:
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0, "seed": 0}
    base.update(fields)
    return Qwen38SamplingParameters(**base)


PROFILES = {
    "thinking": Qwen38SamplingParameters.official_thinking(seed=3),
    "non_thinking": Qwen38SamplingParameters.official_non_thinking(seed=5),
    "top_k_limit_min_p": _custom(top_k=CANDIDATE_TOP_K_LIMIT, top_p=0.9, min_p=0.05, temperature=0.8, seed=7),
    "top_k_one": _custom(top_k=1, temperature=2.0, seed=9),
    "frequency_repetition": _custom(top_k=30, temperature=0.9, frequency_penalty=0.3, repetition_penalty=1.2, seed=11),
}


def _unread_mass_nats(full: torch.Tensor, row: Qwen38CandidateRow) -> float:
    """-log of the probability the read candidates hold under the full softmax: the row's logprob offset."""

    log_z = torch.logsumexp(full.double(), 0)
    return float(log_z - torch.logsumexp(full.double()[row.ids.reshape(-1)], 0))


def _pair(row: Qwen38CandidateRow, full: torch.Tensor, parameters, history=(), top_logprobs=5, prompt_tokens=0):
    left = torch.Generator().manual_seed(parameters.seed)
    right = torch.Generator().manual_seed(parameters.seed)
    penalized = {"token_history": history, "prompt_tokens": prompt_tokens}
    candidate = sample_candidates(row, parameters, **penalized, generator=left, top_logprobs=top_logprobs)
    reference = sample_full_vocabulary(full, parameters, **penalized, generator=right, top_logprobs=top_logprobs)
    return candidate, reference


# --- exactness: the candidate sampler is the full sampler on the kept vector --------------------------


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_candidate_sampler_equals_full_vocabulary_sampler(profile) -> None:
    parameters = PROFILES[profile]
    exact = 0
    for seed in range(12):
        bf16 = _logits(seed)
        row = Qwen38CandidateRow.from_host_row(Qwen38CandidateRow.emulate(bf16).to_host_row())
        full = bf16.to(torch.float32)
        history = tuple(full.topk(6).indices.tolist()) * 2 if parameters.penalizes else ()
        try:
            candidate, reference = _pair(row, full, parameters, history)
        except Qwen38CandidateFallback:
            continue
        exact += 1
        assert candidate.token_id == reference.token_id
        assert candidate.uniform == reference.uniform
        assert candidate.kept == reference.kept
        # The row's log-probabilities are the full ones plus the unread mass (no normalizer is read back).
        offset = _unread_mass_nats(full, row)
        assert 0 <= offset < 0.2 and abs(candidate.logprob - reference.logprob - offset) < 1e-5
        assert [token for token, _ in candidate.top_logprobs] == [token for token, _ in reference.top_logprobs]
        assert all(abs(a - b - offset) < 1e-5 for (_, a), (_, b) in zip(candidate.top_logprobs, reference.top_logprobs))
    assert exact >= 10  # the guard fires on rare near-ties only


def test_request_generator_stream_over_many_steps_matches_the_full_sampler() -> None:
    parameters = Qwen38SamplingParameters.official_thinking(seed=21)
    left = torch.Generator().manual_seed(21)
    right = torch.Generator().manual_seed(21)
    history: list[int] = []
    for seed in range(64):
        bf16 = _logits(100 + seed, peaks=8)
        row = Qwen38CandidateRow.emulate(bf16)
        candidate = sample_candidates(row, parameters, token_history=history, generator=left)
        reference = sample_full_vocabulary(bf16.to(torch.float32), parameters, token_history=history, generator=right)
        assert candidate.token_id == reference.token_id
        history.append(candidate.token_id)
    assert torch.equal(left.get_state(), right.get_state())


def test_sample_host_logits_agrees_with_sample_full_vocabulary_row_by_row() -> None:
    parameters = _custom(top_k=0, top_p=0.9, temperature=0.7, seed=4)
    rows = torch.stack([_logits(s).to(torch.float32) for s in range(3)]).reshape(1, 1, 3, VOCAB_SIZE)
    generator = torch.Generator().manual_seed(4)
    tokens = sample_host_logits(rows, parameters, token_histories=((), (5,), (95_859, 7)), generator=generator)
    expected = []
    generator = torch.Generator().manual_seed(4)
    for row, history in zip(rows.reshape(3, VOCAB_SIZE), ((), (5,), (95_859, 7))):
        expected.append(sample_full_vocabulary(row, parameters, token_history=history, generator=generator).token_id)
    assert tokens.tolist() == [[expected]]


def test_temperature_zero_candidates_equal_the_full_argmax_including_a_lowest_id_tie() -> None:
    bf16 = _logits(1)
    bf16[95_859] = bf16[248_044] = 40.0
    row = Qwen38CandidateRow.emulate(bf16)
    parameters = _custom(temperature=0.0, top_k=1)
    candidate, reference = _pair(row, bf16.to(torch.float32), parameters)
    assert candidate.token_id == reference.token_id == 95_859
    assert candidate.uniform is None and candidate.kept == 1
    greedy = sample_host_logits(bf16.to(torch.float32).reshape(1, 1, 1, -1), Qwen38SamplingParameters.greedy())
    assert greedy.item() == 95_859


# --- the guard: it fires exactly when the full sampler could keep an unread token -------------------


def test_guard_falls_back_when_penalties_push_more_than_k_minus_top_k_candidates_below_the_floor(expect_error) -> None:
    bf16 = _logits(2, peaks=0)
    # ids 0..99 of shard 0 hold a slope of exact bf16 steps 28.0, 27.875, ... (above the random body): the
    # shard's top-k are ids 0..k-1 and a presence penalty of 2 pushes a penalized candidate below the next 16.
    bf16[:100] = (28.0 - torch.arange(100, dtype=torch.float32) * 0.125).to(torch.bfloat16)
    row = Qwen38CandidateRow.emulate(bf16)
    assert set(row.ids[0].tolist()) == set(range(K)) and float(row.shard_floor) == 28.0 - (K - 1) * 0.125
    parameters = _custom(top_k=20, temperature=1.0, presence_penalty=2.0, seed=1)
    history = tuple(range(K - 1))  # every read candidate but the k-th is lowered: the kept 20 reach below the floor
    with expect_error(Qwen38CandidateFallback, match="shard floor"):
        sample_candidates(row, parameters, token_history=history, generator=torch.Generator().manual_seed(1))
    # The fallback was necessary: the full sampler's kept set holds ids the row never read.
    full = bf16.to(torch.float32).clone()
    full[list(history)] -= 2.0
    kept = set(torch.topk(full, 20).indices.tolist())
    assert kept - set(row.ids.reshape(-1).tolist())
    # A milder history leaves the kept vector inside the row and the two agree.
    candidate, reference = _pair(row, bf16.to(torch.float32), parameters, tuple(range(10)))
    assert candidate.token_id == reference.token_id and candidate.kept == reference.kept


def test_guard_falls_back_on_a_tie_at_the_kept_boundary_and_the_shard_floor(expect_error) -> None:
    bf16 = _logits(3, peaks=0)
    bf16[:K] = 25.0  # shard 0's top-k all tie: the k-th read value equals unread ties beyond the row
    bf16[K : K + 10] = 25.0
    row = Qwen38CandidateRow.emulate(bf16)
    with expect_error(Qwen38CandidateFallback):
        sample_candidates(row, _custom(top_k=K), generator=torch.Generator().manual_seed(0))
    with expect_error(Qwen38CandidateFallback):
        sample_candidates(row, _custom(top_k=1, temperature=0.0))


def test_top_k_zero_and_boosting_penalties_fall_back_before_any_draw(expect_error) -> None:
    row = Qwen38CandidateRow.emulate(_logits(4))
    generator = torch.Generator().manual_seed(8)
    state = generator.get_state().clone()
    with expect_error(Qwen38CandidateFallback, match="top_k 0"):
        sample_candidates(row, _custom(top_k=0, seed=8), generator=generator)
    with expect_error(Qwen38CandidateFallback, match="raises logits"):
        sample_candidates(row, _custom(presence_penalty=-0.5, seed=8), token_history=(1,), generator=generator)
    assert torch.equal(generator.get_state(), state)
    # Without a history a boosting penalty cannot act, so the row is still exact.
    sample_candidates(row, _custom(presence_penalty=-0.5, seed=8), generator=generator)


def test_top_k_above_the_candidate_limit_is_refused_not_a_fallback(expect_error) -> None:
    row = Qwen38CandidateRow.emulate(_logits(5))
    with expect_error(Qwen38SamplingError, match="exceeds the candidate limit"):
        sample_candidates(row, _custom(top_k=CANDIDATE_TOP_K_LIMIT + 1))
    assert CANDIDATE_TOP_K_LIMIT == K == 32  # the card profiles' top_k 20 fits; 64 costs the same on device if needed


# --- the transformers reference ----------------------------------------------------------------------


def test_filters_match_transformers_logits_processors() -> None:
    lp = pytest.importorskip("transformers.generation.logits_process")
    for seed, (temperature, top_k, top_p, min_p, repetition) in enumerate(
        [(0.7, 20, 0.8, 0.0, 1.0), (1.0, 20, 0.95, 0.0, 1.0), (0.9, 30, 0.9, 0.1, 1.3), (1.3, 0, 0.99, 0.0, 1.0)]
    ):
        full = _logits(40 + seed).to(torch.float32)
        history = tuple(full.topk(5).indices.tolist()) + (11, 11)
        parameters = _custom(
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition, seed=seed
        )
        scores = full.reshape(1, -1).clone()
        input_ids = torch.tensor([list(history)], dtype=torch.int64)
        if repetition != 1.0:
            scores = lp.RepetitionPenaltyLogitsProcessor(penalty=repetition)(input_ids, scores)
        scores = lp.TemperatureLogitsWarper(temperature)(input_ids, scores)
        if top_k:
            scores = lp.TopKLogitsWarper(top_k=top_k)(input_ids, scores)
        if top_p < 1:
            scores = lp.TopPLogitsWarper(top_p=top_p)(input_ids, scores)
        if min_p > 0:
            scores = lp.MinPLogitsWarper(min_p=min_p)(input_ids, scores)
        reference = torch.softmax(scores.reshape(-1), dim=-1)
        # Our filtered distribution, placed on the vocabulary axis.
        from models.demos.blackhole.qwen38_flash_next.ttnn import sampling

        penalized = sampling._penalize(full.clone(), None, history, parameters)
        positions, _scaled, probabilities = sampling._filter(penalized, parameters)
        ours = torch.zeros(VOCAB_SIZE)
        ours[positions] = probabilities
        # transformers re-softmaxes the -inf-masked row where we renormalize the softmax (fp32 rounding), and
        # its top-p sorts ascending and thresholds the tail mass where we use the descending shifted mask: the
        # same set in exact arithmetic, but with top_k 0 the cumsum over 248k tokens can flip boundary tokens of
        # probability about 1e-6.  Finite top_k vectors agree exactly on the support.
        both = (ours > 0) & (reference > 0)
        boundary = (ours < 2e-5) & (reference < 2e-5)
        assert bool((((ours > 0) == (reference > 0)) | boundary).all()), f"support differs in case {seed}"
        assert torch.allclose(ours[both], reference[both], atol=1e-6, rtol=2e-4), f"case {seed}"
        if top_k:
            assert torch.equal(ours > 0, reference > 0), f"support differs in case {seed}"
        # The row sampler's token has positive probability under the transformers distribution.
        row = Qwen38CandidateRow.emulate(full.to(torch.bfloat16))
        if top_k:
            token = sample_candidates(row, parameters, token_history=history).token_id
            assert reference[token] > 0


# --- the penalties' history: the output, and the prompt only under the repetition rule -----------------------


def test_presence_and_frequency_penalize_the_output_only_and_repetition_the_prompt_too(expect_error) -> None:
    from models.demos.blackhole.qwen38_flash_next.ttnn import sampling

    full = _logits(60).to(torch.float32)
    prompt = tuple(full.topk(6).indices.tolist())  # the prompt holds the six strongest peaks
    output = (prompt[0], prompt[0], PEAK_IDS[0])
    history = prompt + output
    parameters = _custom(presence_penalty=1.5, frequency_penalty=0.5, repetition_penalty=1.2)
    penalized = sampling._penalize(full.clone(), None, history, parameters, prompt_tokens=len(prompt))
    expected = full.clone()
    for token in set(output):
        expected[token] -= 1.5
        expected[token] -= output.count(token) * 0.5
    for token in set(history):
        expected[token] = expected[token] * 1.2 if expected[token] < 0 else expected[token] / 1.2
    assert torch.equal(penalized, expected)
    # prompt_tokens 0 is the whole history as output: the prompt's words lose the additive penalties too (their
    # prompt occurrences count), a token the output alone holds is penalized the same.
    everything = sampling._penalize(full.clone(), None, history, parameters)
    assert everything[PEAK_IDS[0]] == penalized[PEAK_IDS[0]]
    assert bool((everything[list(prompt)] < penalized[list(prompt)]).all())
    # The candidate sampler applies the same rule to the same effect.
    row = Qwen38CandidateRow.emulate(full.to(torch.bfloat16))
    candidate, reference = _pair(row, full, parameters, history=history, prompt_tokens=len(prompt))
    assert (candidate.token_id, candidate.uniform, candidate.kept) == (
        reference.token_id,
        reference.uniform,
        reference.kept,
    )
    # A reply's first token is drawn as if there were no history: the prompt's words are not penalized before the
    # output starts.  Under the whole-history rule the instruct profile's presence 1.5 would have moved the argmax.
    instruct = Qwen38SamplingParameters.official_non_thinking(seed=21)
    first = sample_full_vocabulary(
        full, instruct, token_history=prompt, prompt_tokens=len(prompt), generator=torch.Generator().manual_seed(21)
    )
    unpenalized = sample_full_vocabulary(full, instruct, generator=torch.Generator().manual_seed(21))
    assert (first.token_id, first.uniform) == (unpenalized.token_id, unpenalized.uniform)
    argmax = _custom(top_k=1, presence_penalty=2.0)
    top = int(torch.argmax(full))
    assert sample_full_vocabulary(full, argmax, token_history=(top,), prompt_tokens=1).token_id == top
    assert sample_full_vocabulary(full, argmax, token_history=(top,)).token_id != top
    for bad in (-1, len(history) + 1, True, 1.0):
        with expect_error(ValueError, match="prompt_tokens must be an integer"):
            sample_full_vocabulary(full, parameters, token_history=history, prompt_tokens=bad)
        with expect_error(ValueError, match="prompt_tokens must be an integer"):
            sample_candidates(row, parameters, token_history=history, prompt_tokens=bad)


# --- determinism ------------------------------------------------------------------------------------------


def _stream(seed: int, parameters: Qwen38SamplingParameters, steps: int = 128) -> list[int]:
    generator = torch.Generator().manual_seed(seed) if parameters.temperature else None
    history: list[int] = []
    for step in range(steps):
        # The next row depends on the history, as a model's would.
        row = Qwen38CandidateRow.emulate(_logits(1000 + sum(history) % 977 + step, peaks=6))
        history.append(sample_candidates(row, parameters, token_history=history, generator=generator).token_id)
    return history


def test_same_seed_same_stream_different_seed_differs_and_temperature_zero_is_greedy() -> None:
    thinking = Qwen38SamplingParameters.official_thinking(seed=1234)
    assert _stream(1234, thinking) == _stream(1234, thinking)
    other = Qwen38SamplingParameters.official_thinking(seed=1235)
    assert _stream(1235, other) != _stream(1234, thinking)
    greedy = _custom(temperature=0.0, top_k=1)
    stream = _stream(0, greedy, steps=16)
    history: list[int] = []
    for step in range(16):
        row = _logits(1000 + sum(history) % 977 + step, peaks=6).to(torch.float32)
        history.append(int(torch.argmax(row)))
    assert stream == history


# --- the readback row ----------------------------------------------------------------------------------------


def test_row_layout_and_parser_round_trip_with_exact_ids_and_values() -> None:
    assert SAMPLING_CANDIDATE_ROW_SHAPE == (1, 1, 1, 2 * TP_SIZE * K) == (1, 1, 1, 256)
    assert K % TILE_SIZE == 0  # the per-shard [values | ids] pack is a tile-aligned concat
    assert 4 * SAMPLING_CANDIDATE_ROW_SHAPE[-1] == 1024  # the FP32 ROW_MAJOR readback in bytes
    bf16 = _logits(6)
    emulated = Qwen38CandidateRow.emulate(bf16)
    host_row = emulated.to_host_row()
    packs = host_row.reshape(TP_SIZE, 2, K)  # shard d: its k values, then its k ids
    assert torch.equal(packs[:, 0], emulated.values) and torch.equal(packs[:, 1], emulated.ids.to(torch.float32))
    parsed = Qwen38CandidateRow.from_host_row(host_row)
    assert torch.equal(parsed.ids, emulated.ids) and torch.equal(parsed.values, emulated.values)
    ids = set(parsed.ids.reshape(-1).tolist())
    assert set(PEAK_IDS) <= ids  # 95859 / 248044 / 248319 survive the fp32 lanes
    full = bf16.to(torch.float32)
    for shard in range(TP_SIZE):
        lo, hi = shard * LOCAL_VOCAB_SIZE, (shard + 1) * LOCAL_VOCAB_SIZE
        assert set(parsed.ids[shard].tolist()) == set((torch.topk(full[lo:hi], K).indices + lo).tolist())
        assert torch.equal(parsed.values[shard], full[parsed.ids[shard]])
        assert float(parsed.values[shard, 0]) == float(full[lo:hi].max())
    read = torch.logsumexp(full.double()[parsed.ids.reshape(-1)], 0)
    assert abs(parsed.log_normalizer - float(read)) < 1e-9 and parsed.log_normalizer <= float(
        torch.logsumexp(full.double(), 0)
    )


def test_candidate_logprobs_are_the_full_ones_shifted_by_the_unread_mass() -> None:
    for seed in range(6):
        bf16 = _logits(60 + seed)
        full = bf16.to(torch.float32)
        row = Qwen38CandidateRow.emulate(bf16)
        offset = _unread_mass_nats(full, row)
        assert 0 <= offset < 0.2, offset  # the read candidates hold most of the mass of a peaked row
        candidate, reference = _pair(row, full, Qwen38SamplingParameters.official_thinking(seed=seed))
        assert abs(candidate.logprob - reference.logprob - offset) < 1e-5
        assert candidate.logprob >= reference.logprob


def test_agreement_accepts_id_sets_that_differ_only_at_the_kth_value() -> None:
    bf16 = _logits(8, peaks=0)
    bf16[:K] = 25.0  # shard 0: ids 0..k-1 tie at the k-th value together with the unread ids k..k+9
    bf16[K : K + 10] = 25.0
    truth = Qwen38CandidateRow.emulate(bf16)
    assert set(truth.ids[0].tolist()) <= set(range(K + 10)) and float(truth.values[0, -1]) == 25.0
    exact = truth.agreement(truth)
    assert exact["values_bitwise"] == exact["ids_equal"] == exact["ids_equal_up_to_boundary_ties"] == [True] * TP_SIZE
    assert exact["boundary_ties"][0] == K and exact["ids_exchanged"] == [0] * TP_SIZE
    assert all(1 <= ties <= K for ties in exact["boundary_ties"])
    # The device keeps another tie member at the boundary: values bitwise, ids differ, the tie rule accepts it.
    swapped_ids = truth.ids.clone()
    unread = sorted(set(range(K + 10)) - set(truth.ids[0].tolist()))
    swapped_ids[0, -1] = unread[0]
    swapped = Qwen38CandidateRow(truth.values.clone(), swapped_ids)
    report = swapped.agreement(truth)
    assert report["values_bitwise"][0] and not report["ids_equal"][0] and report["ids_equal_up_to_boundary_ties"][0]
    assert report["ids_exchanged"][0] == 1 and report["boundary_ties"][0] == K
    assert report["ids_equal"][1:] == [True] * (TP_SIZE - 1)
    # An exchange away from the k-th value is a real disagreement.
    wrong = Qwen38CandidateRow(truth.values.clone(), truth.ids.clone())
    wrong.ids[1, 0] = truth.ids[1, 0] + 1 if truth.ids[1, 0] + 1 not in truth.ids[1].tolist() else truth.ids[1, 0] - 1
    report = wrong.agreement(truth)
    assert report["values_bitwise"][1] and not report["ids_equal"][1] and not report["ids_equal_up_to_boundary_ties"][1]
    # Different values are never accepted, whatever the ids.
    off = Qwen38CandidateRow(truth.values.clone(), truth.ids.clone())
    off.values[2, 3] += 0.5
    assert (
        not off.agreement(truth)["values_bitwise"][2] and not off.agreement(truth)["ids_equal_up_to_boundary_ties"][2]
    )


def test_from_host_row_rejects_inconsistent_rows(expect_error) -> None:
    good = Qwen38CandidateRow.emulate(_logits(7)).to_host_row()
    Qwen38CandidateRow.from_host_row(good)
    with expect_error(ValueError, match="must be fp32"):
        Qwen38CandidateRow.from_host_row(good.to(torch.bfloat16))
    bad = good.clone().reshape(-1)
    bad[K] += 0.5  # shard 0's first id, a non-integer
    with expect_error(ValueError, match="not integers"):
        Qwen38CandidateRow.from_host_row(bad.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
    bad = good.clone().reshape(-1)
    bad[2 * K + K] = 5.0  # shard 1's first id inside shard 0
    with expect_error(ValueError, match="leave their shards"):
        Qwen38CandidateRow.from_host_row(bad.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
    bad = good.clone().reshape(-1)
    bad[K + 1] = bad[K]  # shard 0 names one id twice
    with expect_error(ValueError, match="repeat"):
        Qwen38CandidateRow.from_host_row(bad.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
    bad = good.clone().reshape(-1)
    bad[0], bad[1] = bad[1].item(), bad[0].item()  # shard 0's values out of order
    with expect_error(ValueError, match="not descending"):
        Qwen38CandidateRow.from_host_row(bad.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
    bad = good.clone().reshape(-1)
    bad[3] = float("inf")
    with expect_error(ValueError, match="NaN or infinity"):
        Qwen38CandidateRow.from_host_row(bad.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))


# --- parameters ---------------------------------------------------------------------------------------------


def test_new_parameter_fields_default_to_no_op_and_the_official_profiles_forbid_them(expect_error) -> None:
    thinking = Qwen38SamplingParameters.official_thinking(seed=1)
    assert (thinking.min_p, thinking.frequency_penalty, thinking.repetition_penalty) == (0.0, 0.0, 1.0)
    assert not thinking.penalizes and not thinking.raises_logits
    assert Qwen38SamplingParameters.official_non_thinking(seed=1).penalizes
    with expect_error(ValueError, match="thinking profile requires"):
        Qwen38SamplingParameters(1.0, 0.95, 20, 0.0, 1, Qwen38SamplingProfile.THINKING, min_p=0.1)
    for fields, message in (
        ({"min_p": 1.5}, "min_p"),
        ({"frequency_penalty": 2.5}, "frequency_penalty"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
    ):
        with expect_error(ValueError, match=message):
            _custom(**fields)
    assert _custom(repetition_penalty=0.9).raises_logits and _custom(frequency_penalty=-1.0).raises_logits


# --- the device epilogue: op list and constants ---------------------------------------------------------------

EPILOGUE_DEVICE_OPS = (
    "topk",  # per shard: k bf16 values and UINT16 local ids, sorted descending (the stock path, no indices_tensor)
    "typecast",  # values bf16 -> fp32 (exact)
    "typecast",  # ids uint16 -> fp32 (exact)
    "add",  # + the shard's first global id (SFPU fp32)
    "concat",  # [values | global ids], two tile-aligned pieces
    "to_layout",  # -> padding-free ROW_MAJOR
    "all_gather",  # the one collective: the four shards' packs
    "copy",  # into the persistent readback row
)
FPU_STAGE_OPS = frozenset({"sum", "mean", "prod", "max", "min", "matmul", "linear", "bmm"})


def test_sampling_candidates_is_the_emulated_op_sequence_and_the_greedy_path_is_untouched() -> None:
    body = inspect.getsource(Qwen38TTNNLMHead.sampling_candidates)
    ops = re.findall(r"ttnn\.((?:\w+\.)*\w+)\(", body)
    assert tuple(ops) == EPILOGUE_DEVICE_OPS
    assert FPU_STAGE_OPS.isdisjoint(ops) and "compute_kernel_config" not in body  # no reduce reads an id or a value
    assert ops.count("all_gather") == 1 and "indices_tensor" not in body.split('"""')[2]
    for forbidden in (
        "to_torch(",
        "from_torch(",
        "ConcatMeshToTensor",
        "synchronize",
        "topk_large_indices",
        "ttnn.gather(",
    ):
        assert forbidden not in body.split('"""')[2]
    assert "No id enters an FPU or reduce stage" in " ".join(body.split())  # the docstring wraps the phrase
    # The production greedy path does not know the epilogue exists.
    for name in ("greedy_candidates", "resolve_greedy_on_device", "greedy_token", "__call__"):
        assert "sampling" not in inspect.getsource(getattr(Qwen38TTNNLMHead, name))


def test_candidate_constants_are_a_vocab_sharded_start_scalar_and_a_replicated_row() -> None:
    build = inspect.getsource(Qwen38TTNNSamplingCandidateConstants.build)
    assert "ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1))" in build
    assert "reshape(1, TP_SIZE, 1, 1) * LOCAL_VOCAB_SIZE" in build and "ttnn.TILE_LAYOUT" in build
    assert "torch.zeros(SAMPLING_CANDIDATE_ROW_SHAPE" in build and "replicate_tensor_2d_mesh_mapper" in build
    assert "dtype=ttnn.float32" in build
    validate = inspect.getsource(Qwen38TTNNSamplingCandidateConstants.validate)
    assert "placement=TensorPlacement.VOCAB_SHARDED, shard_dim=1" in validate
    assert "placement=TensorPlacement.REPLICATED" in validate
    assert "SAMPLING_CANDIDATE_ROW_SHAPE" in validate and "ttnn.ROW_MAJOR_LAYOUT" in validate
    starts = torch.arange(TP_SIZE, dtype=torch.float32).reshape(1, TP_SIZE, 1, 1) * LOCAL_VOCAB_SIZE
    assert starts.reshape(-1).tolist() == [0.0, 62080.0, 124160.0, 186240.0]


def test_sampling_candidates_rejects_foreign_vocabulary_ownership(expect_error) -> None:
    from types import SimpleNamespace

    from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import Qwen38ShardedLogits

    shell = object.__new__(Qwen38TTNNLMHead)
    shell.weights = SimpleNamespace(
        vocab_ranges=tuple((s, s + LOCAL_VOCAB_SIZE) for s in range(0, VOCAB_SIZE, LOCAL_VOCAB_SIZE))
    )
    foreign = Qwen38ShardedLogits(tensor=None, vocab_ranges=((0, 1),), global_shape=(1, 1, 1, VOCAB_SIZE))
    with expect_error(ValueError, "differs from the exact TP4 contract"):
        shell.sampling_candidates(foreign, None)
