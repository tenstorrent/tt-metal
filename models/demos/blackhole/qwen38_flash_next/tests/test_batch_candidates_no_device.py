# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3g-1: the batch adapter's sampled lanes draw from the device candidate rows (no device).

Per-row routing of one mixed step (greedy, candidate draw, each fallback reason) on a fake lane decoder whose candidate
rows are ``torch.topk`` per shard of the bf16 logits it also serves as the full gather; the knob
(``QWEN38_BATCH_CANDIDATES``: 1 / unset on, 0 the P2 path, anything else refused); the candidate law against the
full-row law of ``sample_from_row`` (probability vectors, then the drawn tokens under equal generators); a seeded row
draws the same token alone and in a full batch; the counters and the periodic log split.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as adapter_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401  (fixture)
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import (
    LANES,
    FakeDecoder,
    Rig,
    _prompt,
    _rows_for,
    _sampling,
    _three,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    BATCH_CANDIDATES_VARIABLE,
    CandidateFallback,
    Qwen38ForCausalLM,
    Qwen38HostSamplingPolicy,
    batch_candidates_enabled,
    candidate_law,
    candidate_route,
    parse_candidate_rows,
    position_generator,
    sample_from_candidates,
    sample_from_row,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38CandidateRow


@pytest.fixture(autouse=True)
def _single_path_prefill(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.delenv(BATCH_CANDIDATES_VARIABLE, raising=False)


def bf16_row(seed: int, boost: tuple[int, float] | None = None) -> torch.Tensor:
    row = torch.randn(VOCAB_SIZE, generator=torch.Generator().manual_seed(seed))
    if boost is not None:
        row[boost[0]] += boost[1]
    return row.to(torch.bfloat16).to(torch.float32)


def distinct_row(seed: int, tokens: tuple[int, ...] = ()) -> torch.Tensor:
    """A bf16 row with 40 distinct top values (4.0 + j / 32, above every bulk value) at random places, ``tokens`` among
    them: no ties in the region the filters read, so a token-level comparison is exact."""

    generator = torch.Generator().manual_seed(seed)
    row = 0.5 * torch.randn(VOCAB_SIZE, generator=generator)
    others = [t for t in torch.randperm(TOKENIZER_SIZE, generator=generator).tolist() if t not in tokens]
    places = list(tokens) + others[: 40 - len(tokens)]
    for rank, place in enumerate(torch.randperm(40, generator=generator).tolist()):
        row[places[rank]] = 4.0 + place / 32
    return row.to(torch.bfloat16).to(torch.float32)


def candidate_row_of(logits: torch.Tensor) -> torch.Tensor:
    return Qwen38CandidateRow.emulate(logits.to(torch.bfloat16)).to_host_row().reshape(-1)


class CandidateDecoder(FakeDecoder):
    """The fake lane decoder with the candidate surface: the full gather and the candidate rows are two views of the same
    bf16 logits (``overrides`` replaces a lane's row; ``corrupt`` poisons a lane's candidate row)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.has_candidates = True
        self.candidate_reads = 0
        self.full_reads = 0
        self.overrides: dict[int, torch.Tensor] = {}
        self.corrupt: dict[int, float] = {}
        self.identical_lanes = False

    def logits_for(self, lane: int, step_no: int) -> torch.Tensor:
        if lane in self.overrides:
            return self.overrides[lane].clone()
        key = 0 if self.identical_lanes else lane
        generator = torch.Generator().manual_seed(1000 + 31 * key + step_no)
        row = torch.randn(VOCAB_SIZE, generator=generator)
        row[self.last_ids[lane]] += 12.0
        return row.to(torch.bfloat16).to(torch.float32)

    def read_full_logits(self) -> torch.Tensor:
        self.full_reads += 1
        return super().read_full_logits()

    def read_candidates(self) -> torch.Tensor:
        self.candidate_reads += 1
        rows = torch.stack(
            [candidate_row_of(self.logits_for(lane, len(self.steps))) for lane in range(self.lanes)]
        ).clone()
        for lane, value in self.corrupt.items():
            rows[lane, 3] = value
        return rows


@pytest.fixture
def crig(monkeypatch) -> Rig:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", CandidateDecoder)
    return Rig(monkeypatch)


def _full(rig: Rig, lane: int, step_no: int, policy, generator, history=None) -> int:
    row = rig.decoder.logits_for(lane, step_no).clone()
    row[TOKENIZER_SIZE:] = float("-inf")
    if history is not None:
        vllm_module.apply_penalties(row, *history, policy)
    return sample_from_row(row, policy, generator)


# -- the knob -----------------------------------------------------------------------------------------------------------


def test_the_knob_is_on_by_default_and_refuses_other_values(expect_error) -> None:
    assert batch_candidates_enabled({}) is True
    assert batch_candidates_enabled({BATCH_CANDIDATES_VARIABLE: "1"}) is True
    assert batch_candidates_enabled({BATCH_CANDIDATES_VARIABLE: "0"}) is False
    for bad in ("", "2", "true", "on", " 1"):
        with expect_error(ValueError, BATCH_CANDIDATES_VARIABLE):
            batch_candidates_enabled({BATCH_CANDIDATES_VARIABLE: bad})


def test_a_bad_knob_value_is_refused_at_adapter_construction(monkeypatch, expect_error) -> None:
    monkeypatch.setenv(BATCH_CANDIDATES_VARIABLE, "yes")
    with expect_error(ValueError, BATCH_CANDIDATES_VARIABLE):
        Qwen38ForCausalLM(None, allocated_context=adapter_tests.ALLOCATED_CONTEXT, batch_lanes=LANES)


def test_the_single_slot_adapter_ignores_the_knob(monkeypatch) -> None:
    monkeypatch.setenv(BATCH_CANDIDATES_VARIABLE, "yes")
    # batch_lanes 1 never reads it: the single-slot path is unchanged
    model = Qwen38ForCausalLM(object(), allocated_context=adapter_tests.ALLOCATED_CONTEXT, batch_lanes=1)
    assert model.batch_candidates is False


# -- per-row routing in one mixed step ----------------------------------------------------------------------------------


def _mixed_params(prompt_tokens=None):
    """Rows 0..7: greedy, candidates (seeded), candidates (unseeded-looking but seeded), top_k off, top_k 50, a raising
    penalty, presence penalty over a history (candidates), a flat row for the guard (set by the test)."""

    return _sampling(
        8,
        temperature=[0.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.7, 1.0],
        top_k=[1, 20, 32, VOCAB_SIZE, 50, 20, 20, 20],
        top_p=[0.0, 0.95, 1.0, 0.95, 0.9, 0.95, 0.8, 1.0],
        seed=[None, 11, 22, 33, 44, 55, 66, 77],
        presence_penalty=[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.5, 0.0],
    )


def _run_mixed(rig: Rig, flat_lane: int | None = None, corrupt_lane: int | None = None):
    prompts = [_prompt(20 + 4 * i, 100 * (i + 1)) for i in range(8)]
    blocks = [[1 + i] for i in range(8)]
    rig.prefill(prompts, list(range(8)), blocks, adapter_tests.greedy_batch_sampling_params(8))
    if flat_lane is not None:
        rig.decoder.overrides[flat_lane] = torch.zeros(VOCAB_SIZE)  # every value equal: the k-th ties the shard floor
    if corrupt_lane is not None:
        rig.decoder.corrupt[corrupt_lane] = float("nan")
    firsts = [1000 + i for i in range(8)]
    prompt_tokens = torch.full((8, 8), -1, dtype=torch.int32)
    output_tokens = torch.full((8, 8), -1, dtype=torch.int32)
    for row in (5, 6):
        prompt_tokens[row, :3] = torch.tensor([10, 11, 12])
    output_tokens[6, :4] = torch.tensor([13, 14, 14, 15])
    output_tokens[5, :2] = torch.tensor([13, 14])
    params = _mixed_params()
    out = rig.decode(
        _rows_for(prompts, blocks, firsts, list(range(8))),
        sampling_params=params,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    return prompts, params, out, (prompt_tokens, output_tokens)


def _policy(params, row) -> Qwen38HostSamplingPolicy:
    return Qwen38HostSamplingPolicy.from_contract(params, row)


def test_a_mixed_step_routes_every_row_and_pays_one_gather_for_the_fallbacks(crig: Rig) -> None:
    prompts, params, out, (prompt_tokens, output_tokens) = _run_mixed(crig)
    decoder = crig.decoder
    assert decoder.candidate_reads == 1
    assert decoder.full_reads == 1  # rows 3 (top_k off), 4 (top_k 50), 5 (raising penalty): one gather for all three
    counters = crig.model.candidate_counters
    assert counters["candidate_steps"] == 1 and counters["candidate_rows"] == 4  # rows 1, 2, 6, 7
    assert counters["fallback_steps"] == 1
    assert (counters["fallback_top_k"], counters["fallback_penalty"]) == (2, 1)
    assert counters["fallback_guard"] == 0 and counters["fallback_nonfinite"] == 0
    histories = {
        5: (torch.tensor([10, 11, 12]), torch.tensor([13, 14])),
        6: (torch.tensor([10, 11, 12]), torch.tensor([13, 14, 14, 15])),
    }
    assert out[0, 0] == FakeDecoder.next_id(1000, len(prompts[0]))  # greedy: the device id, untouched
    for row in range(1, 8):
        policy = _policy(params, row)
        expected = _full(crig, row, 1, policy, position_generator(policy.seed, len(prompts[row])), histories.get(row))
        assert out[row, 0] == expected, row  # candidate rows draw the token the full path draws


def test_a_flat_lane_trips_the_guard_and_a_poisoned_row_the_finite_check(crig: Rig) -> None:
    prompts, params, out, _ = _run_mixed(crig, flat_lane=1, corrupt_lane=2)
    counters = crig.model.candidate_counters
    assert counters["fallback_guard"] == 1 and counters["fallback_nonfinite"] == 1
    assert counters["candidate_rows"] == 2 and counters["fallback_steps"] == 1
    assert crig.decoder.full_reads == 1
    for row in (1, 2):  # both fell back and drew over the full logits, with their own fresh generator
        policy = _policy(params, row)
        assert out[row, 0] == _full(crig, row, 1, policy, position_generator(policy.seed, len(prompts[row])))


def test_all_candidate_rows_pay_no_full_gather(crig: Rig) -> None:
    prompts, blocks, firsts = _three(crig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(8, temperature=[1.0, 0.0, 0.9], top_k=[20, 1, 5], top_p=[0.95, 0.0, 1.0], seed=[1, None, 3])
    out = crig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5, sampling_params=params)
    assert crig.decoder.candidate_reads == 1 and crig.decoder.full_reads == 0
    assert out[1, 0] == FakeDecoder.next_id(firsts[1], len(prompts[1]))
    counters = crig.model.candidate_counters
    assert counters["candidate_rows"] == 2 and counters["fallback_steps"] == 0


def test_greedy_only_steps_pay_nothing_new(crig: Rig) -> None:
    prompts, blocks, firsts = _three(crig)
    firsts = firsts.reshape(-1).tolist()
    out = crig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5)
    assert crig.decoder.candidate_reads == 0 and crig.decoder.full_reads == 0
    assert crig.model.candidate_counters["candidate_steps"] == 0
    assert out[0, 0] == FakeDecoder.next_id(firsts[0], len(prompts[0]))


def test_host_logits_without_sampling_params_still_gather_everything(crig: Rig) -> None:
    prompts, blocks, firsts = _three(crig)
    firsts = firsts.reshape(-1).tolist()
    crig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5, sampling_params=None)
    assert crig.decoder.full_reads == 1 and crig.decoder.candidate_reads == 0


def test_knob_zero_is_the_full_logits_path(monkeypatch) -> None:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", CandidateDecoder)
    monkeypatch.setenv(BATCH_CANDIDATES_VARIABLE, "0")
    rig = Rig(monkeypatch)
    assert rig.model.batch_candidates is False
    prompts, params, out, (prompt_tokens, output_tokens) = _run_mixed(rig)
    assert rig.decoder.candidate_reads == 0 and rig.decoder.full_reads == 1
    assert all(value == 0 for value in rig.model.candidate_counters.values())
    for row in range(1, 8):
        policy = _policy(params, row)
        history = None
        if row in (5, 6):
            history = (
                torch.tensor([10, 11, 12]),
                torch.tensor([13, 14]) if row == 5 else torch.tensor([13, 14, 14, 15]),
            )
        assert out[row, 0] == _full(rig, row, 1, policy, position_generator(policy.seed, len(prompts[row])), history)


def test_a_decoder_without_the_candidate_surface_keeps_the_old_path(monkeypatch) -> None:
    rig = Rig(monkeypatch)  # a plain FakeDecoder: no has_candidates
    assert rig.model.batch_candidates is True
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(8, temperature=[1.0], top_k=[20], top_p=[0.95], seed=[9])
    out = rig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5, sampling_params=params)
    policy = Qwen38HostSamplingPolicy(1.0, 20, 0.95, seed=9)
    assert out[0, 0] == adapter_tests._expected_draw(rig, 0, 1, policy, position_generator(9, len(prompts[0])))


# -- a seeded row draws the same token alone and in a full batch --------------------------------------------------------


def _seeded_run(monkeypatch, live: int, params):
    """A fresh rig whose lanes all serve the same logits; ``live`` requests (the first rows) decode one step."""

    monkeypatch.setattr(adapter_tests, "FakeDecoder", CandidateDecoder)
    rig = Rig(monkeypatch)
    rig.decoder.identical_lanes = True  # only the batch composition differs between the runs
    prompts = [_prompt(40, 500 + i) for i in range(live)]
    blocks = [[1 + i] for i in range(live)]
    rig.prefill(prompts, list(range(live)), blocks, adapter_tests.greedy_batch_sampling_params(live))
    rows = _rows_for(prompts, blocks, [1000] * live, list(range(live))) + [None] * (LANES - live)
    return rig, rig.decode(rows, sampling_params=params)


def test_a_seeded_row_draws_the_same_token_alone_and_in_a_full_batch(monkeypatch) -> None:
    full = _sampling(
        8,
        temperature=[1.0, 1.0, 0.7, 0.0, 1.0, 1.0, 0.9, 1.0],
        top_k=[20, 20, 10, 1, VOCAB_SIZE, 50, 32, 5],
        top_p=[0.95, 0.95, 0.9, 0.0, 0.9, 1.0, 1.0, 0.8],
        seed=[7, 7, 8, None, 9, 10, 11, 12],
    )
    rig, batch = _seeded_run(monkeypatch, 8, full)
    assert rig.decoder.candidate_reads == 1 and rig.decoder.full_reads == 1  # rows 4 and 5 (top_k) fell back
    assert batch[0, 0] == batch[1, 0]  # same seed, same position, same logits: the same token in two lanes
    alone = _sampling(8, temperature=[1.0], top_k=[20], top_p=[0.95], seed=[7])
    rig_alone, only = _seeded_run(monkeypatch, 1, alone)
    assert rig_alone.decoder.candidate_reads == 1 and rig_alone.decoder.full_reads == 0
    assert only[0, 0] == batch[0, 0]
    # and the same through the full-logits path (knob 0): the token does not depend on the path either
    monkeypatch.setenv(BATCH_CANDIDATES_VARIABLE, "0")
    rig_off, off = _seeded_run(monkeypatch, 1, alone)
    assert rig_off.decoder.candidate_reads == 0 and off[0, 0] == only[0, 0]


def test_the_candidate_draw_equals_the_full_row_draw_per_seed_over_many_rows() -> None:
    mismatches = 0
    for seed in range(200):
        row = distinct_row(seed)
        masked = row.clone()
        masked[TOKENIZER_SIZE:] = float("-inf")
        cands = candidate_row_of(row).reshape(1, -1)
        values, ids, valid = parse_candidate_rows(cands)
        assert bool(valid[0])
        policy = Qwen38HostSamplingPolicy(
            temperature=(0.5, 0.8, 1.0, 1.3)[seed % 4],
            top_k=(1, 2, 5, 20, 32)[seed % 5],
            top_p=(1.0, 0.95, 0.8)[seed % 3],
            seed=seed,
        )
        expected = sample_from_row(masked, policy, position_generator(seed, 100 + seed))
        try:
            got = sample_from_candidates(values[0], ids[0], policy, position_generator(seed, 100 + seed))
        except CandidateFallback:
            continue
        mismatches += got != expected
    assert mismatches == 0


# -- the law ------------------------------------------------------------------------------------------------------------


def full_law(row: torch.Tensor, policy: Qwen38HostSamplingPolicy) -> dict[int, float]:
    """vLLM's law over the whole row for ``0 < top_k <= 32``: the top-k set (ties at the k-th kept), temperature softmax
    over it, the nucleus.  Written out without ``sample_from_row``'s code."""

    masked = row.clone()
    masked[TOKENIZER_SIZE:] = float("-inf")
    order = torch.argsort(masked, descending=True, stable=True)
    kth = masked[order[policy.top_k - 1]]
    kept = order[masked[order] >= kth]
    probabilities = torch.softmax(masked[kept] / policy.temperature, 0)
    if policy.top_p < 1.0:
        keep = torch.cumsum(probabilities, 0) - probabilities < policy.top_p
        keep[0] = True
        kept, probabilities = kept[keep], probabilities[keep]
    probabilities = probabilities / probabilities.sum()  # the draw is scale-free (argmax of p / noise)
    return {int(t): float(p) for t, p in zip(kept, probabilities)}


def assert_same_law(row: torch.Tensor, tokens: torch.Tensor, probabilities: torch.Tensor, expected: dict) -> None:
    """The same law up to the order of exactly tied logits: the probabilities as sorted vectors and the logits of the
    kept tokens as sorted vectors (a nucleus that ends inside a tie group keeps either member of the group)."""

    probabilities = probabilities / probabilities.sum()
    got = torch.sort(probabilities, descending=True).values
    want = torch.sort(torch.tensor(list(expected.values())), descending=True).values
    assert got.shape == want.shape and float((got - want).abs().max()) < 1e-6
    masked = row.clone()
    masked[TOKENIZER_SIZE:] = float("-inf")
    kept_logits = torch.sort(masked[tokens], descending=True).values
    want_logits = torch.sort(masked[torch.tensor(list(expected))], descending=True).values
    assert torch.equal(kept_logits, want_logits)


@pytest.mark.parametrize("top_k", (1, 2, 7, 20, 32))
@pytest.mark.parametrize("temperature, top_p", ((1.0, 0.95), (0.7, 0.8), (1.5, 1.0), (0.3, 0.5)))
def test_candidate_law_is_the_full_row_law_for_top_k_up_to_32(top_k: int, temperature: float, top_p: float) -> None:
    checked = 0
    for seed in range(40):
        row = bf16_row(seed + 1000 * top_k)
        policy = Qwen38HostSamplingPolicy(temperature, top_k, top_p, seed=seed)
        values, ids, valid = parse_candidate_rows(candidate_row_of(row).reshape(1, -1))
        assert bool(valid[0])
        try:
            probabilities, tokens = candidate_law(values[0], ids[0], policy)
        except CandidateFallback as fallback:
            assert fallback.reason == "guard"
            continue
        expected = full_law(row, policy)
        assert_same_law(row, tokens, probabilities, expected)
        checked += 1
    assert checked >= 30


@pytest.mark.parametrize(
    "fields",
    (
        {"presence_penalty": 1.5},
        {"frequency_penalty": 0.4},
        {"repetition_penalty": 1.3},
        {"presence_penalty": 0.5, "frequency_penalty": 0.2, "repetition_penalty": 1.1},
    ),
)
def test_candidate_law_with_penalties_is_the_penalized_full_row_law(fields) -> None:
    history = (torch.tensor([3, 17, 90]), torch.tensor([5, 5, 6, 200]))
    checked = 0
    for seed in range(40):
        row = distinct_row(seed, tokens=(3, 5, 6, 200))  # the history's tokens sit among the candidates
        policy = Qwen38HostSamplingPolicy(0.9, 20, 0.95, seed=seed, **fields)
        values, ids, valid = parse_candidate_rows(candidate_row_of(row).reshape(1, -1))
        try:
            probabilities, tokens = candidate_law(values[0], ids[0], policy, history)
        except CandidateFallback as fallback:
            assert fallback.reason == "guard"
            continue
        penalized = row.clone()
        penalized[TOKENIZER_SIZE:] = float("-inf")
        vllm_module.apply_penalties(penalized, *history, policy)
        reference = full_law(penalized, policy.__class__(0.9, 20, 0.95, seed=seed))  # penalties already applied
        assert_same_law(penalized, tokens, probabilities, reference)
        generator = position_generator(seed, 5)
        assert sample_from_candidates(values[0], ids[0], policy, generator, history) == sample_from_row(
            penalized, policy, position_generator(seed, 5)
        )
        checked += 1
    assert checked >= 30


@pytest.mark.parametrize("fields", ({"presence_penalty": 0.5}, {"repetition_penalty": 1.3}, {"frequency_penalty": 0.4}))
def test_a_penalized_greedy_row_draws_the_penalized_full_row_argmax(fields) -> None:
    history = (torch.tensor([3, 17, 90]), torch.tensor([5, 5, 6, 200]))
    for seed in range(20):
        row = distinct_row(seed, tokens=(3, 5, 6, 200))
        policy = Qwen38HostSamplingPolicy(0.0, 20, 1.0, seed=seed, **fields)
        assert policy.greedy and policy.penalized and candidate_route(policy) is None
        values, ids, _ = parse_candidate_rows(candidate_row_of(row).reshape(1, -1))
        penalized = row.clone()
        penalized[TOKENIZER_SIZE:] = float("-inf")
        vllm_module.apply_penalties(penalized, *history, policy)
        probabilities, tokens = candidate_law(values[0], ids[0], policy, history)
        assert probabilities.tolist() == [1.0] and int(tokens[0]) == int(penalized.argmax())
        assert sample_from_candidates(values[0], ids[0], policy, position_generator(seed, 5), history) == int(
            penalized.argmax()
        )


def test_a_penalized_greedy_tie_takes_the_lowest_id() -> None:
    row = distinct_row(7)
    top = float(row.max())
    row[[900, 70000, 130000]] = top + 1.0  # three equal maxima on three shards
    policy = Qwen38HostSamplingPolicy(0.0, 20, 1.0, presence_penalty=0.5)
    values, ids, _ = parse_candidate_rows(candidate_row_of(row).reshape(1, -1))
    history = (torch.tensor([1]), torch.tensor([2]))
    assert sample_from_candidates(values[0], ids[0], policy, None, history) == 900 == int(row.argmax())


def test_routing_by_policy_alone() -> None:
    def policy(**fields):
        return Qwen38HostSamplingPolicy(**{"temperature": 1.0, "top_k": 20, "top_p": 0.95, **fields})

    assert candidate_route(policy()) is None
    assert candidate_route(policy(top_k=32)) is None and candidate_route(policy(top_k=1)) is None
    assert candidate_route(policy(top_k=33)) == "top_k"
    assert candidate_route(policy(top_k=VOCAB_SIZE)) == "top_k"  # the plugin turns top_k <= 0 into the vocabulary
    assert candidate_route(policy(top_k=0)) == "top_k"
    for raising in ({"presence_penalty": -0.1}, {"frequency_penalty": -0.1}, {"repetition_penalty": 0.9}):
        assert candidate_route(policy(**raising)) == "penalty"
    for lowering in ({"presence_penalty": 1.0}, {"frequency_penalty": 0.5}, {"repetition_penalty": 1.2}):
        assert candidate_route(policy(**lowering)) is None


def test_parse_flags_malformed_rows() -> None:
    good = candidate_row_of(bf16_row(1))
    rows = torch.stack([good, good.clone(), good.clone(), good.clone(), good.clone()])
    rows[1, 0] = float("inf")  # non-finite value
    rows[2, 32 + 0] = 0.5  # a fractional id (the first shard's ids start at lane 32)
    rows[3, 32 + 1] = rows[3, 32]  # a repeated id inside a shard
    rows[4, 96 + 5] = 3.0  # shard 1's ids start at lane 96: an id outside its range
    values, ids, valid = parse_candidate_rows(rows)
    assert valid.tolist() == [True, False, False, False, False]
    assert tuple(values.shape) == (5, 4, 32) and tuple(ids.shape) == (5, 4, 32)


# -- the periodic log line ----------------------------------------------------------------------------------------------


def test_the_log_line_carries_the_split_and_the_counters(crig: Rig, monkeypatch) -> None:
    monkeypatch.setattr(vllm_module, "DECODE_LOG_INTERVAL", 1)
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        _run_mixed(crig)
    finally:
        logger.remove(sink)
    line = next(text for text in lines if "batch decode:" in text)
    for part in ("replay", "candidates ", "host draw", "fallback gather", "fallback steps 1", "top_k 2, penalty 1"):
        assert part in line, (part, line)
    stats = crig.model.decode_stats[-1]
    assert stats["candidates_s"] > 0 and stats["draw_s"] > 0 and stats["fallback_s"] > 0
