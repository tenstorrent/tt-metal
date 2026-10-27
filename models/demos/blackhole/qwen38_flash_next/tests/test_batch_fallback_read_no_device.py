# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3g-3: the cheaper fallback read and the faster host full-row law (no device).

Part A: the knob (``QWEN38_BATCH_ROW_READ``: 1 / unset on, 0 the full gather, anything else refused), the row-read
buckets, and the adapter's routing of a step with fallback rows on a fake decoder (the fallback lanes only, the bucket
limit, the knob, padding columns masked, tokens equal to the full-gather path).  Part B: ``sample_from_row`` against a
frozen copy of the code it replaces (bit-identical tokens wherever the old code took the top-1024 or a top-k branch),
the empirical law of the full-row nucleus and of the inverse-CDF draw against the exact law, a masked token never
drawn, seeded determinism.
"""

from __future__ import annotations

import math

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as adapter_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_candidates_no_device import (
    CandidateDecoder,
    _full,
    _policy,
    _run_mixed,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401  (fixture)
    _chi_square_ok,
    _vllm_sampler_probabilities,
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import Rig
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import row_read_bucket, row_read_buckets
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    BATCH_CANDIDATES_VARIABLE,
    BATCH_NORMALIZER_VARIABLE,
    BATCH_ROW_READ_VARIABLE,
    HOST_DRAW_BRANCHES,
    Qwen38ForCausalLM,
    Qwen38HostSamplingPolicy,
    batch_row_read_enabled,
    position_generator,
    sample_from_row,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.delenv(BATCH_CANDIDATES_VARIABLE, raising=False)
    monkeypatch.delenv(BATCH_ROW_READ_VARIABLE, raising=False)
    monkeypatch.setenv(BATCH_NORMALIZER_VARIABLE, "0")  # a top-k-off row is a "top_k" fallback row: P3g-1's routing


# -- the knob and the buckets
# -------------------------------------------------------------------------------------------


def test_the_row_read_knob_is_on_by_default_and_refuses_other_values(expect_error) -> None:
    assert batch_row_read_enabled({}) is True
    assert batch_row_read_enabled({BATCH_ROW_READ_VARIABLE: "1"}) is True
    assert batch_row_read_enabled({BATCH_ROW_READ_VARIABLE: "0"}) is False
    for bad in ("", "2", "true", "on", " 1"):
        with expect_error(ValueError, BATCH_ROW_READ_VARIABLE):
            batch_row_read_enabled({BATCH_ROW_READ_VARIABLE: bad})


def test_the_adapter_reads_the_row_read_knob(monkeypatch, expect_error) -> None:
    def build(lanes: int = adapter_tests.LANES) -> Qwen38ForCausalLM:
        return Qwen38ForCausalLM(None, allocated_context=adapter_tests.ALLOCATED_CONTEXT, batch_lanes=lanes)

    assert build().batch_row_read is True
    monkeypatch.setenv(BATCH_ROW_READ_VARIABLE, "0")
    assert build().batch_row_read is False
    monkeypatch.setenv(BATCH_ROW_READ_VARIABLE, "1")
    monkeypatch.setenv(BATCH_CANDIDATES_VARIABLE, "0")
    assert build().batch_row_read is False  # no candidate rows: no fallback step to serve
    monkeypatch.setenv(BATCH_ROW_READ_VARIABLE, "maybe")
    with expect_error(ValueError, BATCH_ROW_READ_VARIABLE):
        build()
    assert Qwen38ForCausalLM(object(), allocated_context=adapter_tests.ALLOCATED_CONTEXT).batch_row_read is False


def test_the_buckets_are_the_powers_of_two_up_to_the_lanes() -> None:
    assert row_read_buckets(32) == (1, 2, 4, 8, 16, 32)
    assert row_read_buckets(8) == (1, 2, 4, 8)
    assert row_read_buckets(12) == (1, 2, 4, 8)  # nine lanes and more of a 12-lane decoder take the full gather
    assert row_read_buckets(2) == (1, 2)
    buckets = row_read_buckets(32)
    assert [row_read_bucket(count, buckets) for count in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)] == [
        1,
        2,
        4,
        4,
        8,
        8,
        16,
        16,
        32,
        32,
    ]
    assert row_read_bucket(33, buckets) is None and row_read_bucket(0, buckets) is None
    assert row_read_bucket(9, row_read_buckets(8)) is None


# -- the adapter's routing
# ----------------------------------------------------------------------------------------------


class RowReadDecoder(CandidateDecoder):
    """The candidate fake with the row-read surface: ``read_logits_rows`` returns the rows of the same bf16 logits the
    full gather serves (``row_reads`` records the lanes asked for).  ``has_row_read`` is always true: the adapter's knob
    and its
    bucket check decide whether it is used."""

    has_row_read = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row_read_buckets = row_read_buckets(self.lanes)
        self.row_reads: list[list[int]] = []

    def read_logits_rows(self, lanes) -> torch.Tensor:
        self.row_reads.append(list(lanes))
        return torch.stack([self.logits_for(lane, len(self.steps)) for lane in lanes]).clone()


@pytest.fixture
def rrig(monkeypatch) -> Rig:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", RowReadDecoder)
    return Rig(monkeypatch)


def _history_of(row: int):
    return {
        5: (torch.tensor([10, 11, 12]), torch.tensor([13, 14])),
        6: (torch.tensor([10, 11, 12]), torch.tensor([13, 14, 14, 15])),
    }.get(row)


def _assert_tokens_equal_the_full_path(rig: Rig, prompts, params, out) -> None:
    for row in range(1, 8):
        policy = _policy(params, row)
        expected = _full(rig, row, 1, policy, position_generator(policy.seed, len(prompts[row])), _history_of(row))
        assert out[row, 0] == expected, row


def test_a_fallback_step_reads_only_the_fallback_lanes(rrig: Rig) -> None:
    prompts, params, out, _ = _run_mixed(rrig)
    decoder = rrig.decoder
    assert decoder.row_reads == [[3, 4, 5]]  # top_k off, top_k 50, a raising penalty: one read of those three lanes
    assert decoder.full_reads == 0 and decoder.candidate_reads == 1
    counters = rrig.model.candidate_counters
    assert counters["row_read_steps"] == 1 and counters["row_read_rows"] == 3 and counters["full_read_steps"] == 0
    assert counters["fallback_steps"] == 1
    _assert_tokens_equal_the_full_path(rrig, prompts, params, out)


def test_the_row_read_masks_the_padding_columns(rrig: Rig) -> None:
    row = torch.randn(VOCAB_SIZE, generator=torch.Generator().manual_seed(5))
    row[TOKENIZER_SIZE:] = 50.0  # a padded column above every real logit: unmasked, every fallback draw would pick one
    rrig.decoder.overrides[3] = row.to(torch.bfloat16).to(torch.float32)
    rrig.decoder.overrides[4] = rrig.decoder.overrides[3].clone()
    prompts, params, out, _ = _run_mixed(rrig)
    assert rrig.decoder.row_reads == [[3, 4, 5]]
    assert int(out[3, 0]) < TOKENIZER_SIZE and int(out[4, 0]) < TOKENIZER_SIZE
    _assert_tokens_equal_the_full_path(rrig, prompts, params, out)


def test_the_knob_zero_reads_the_full_gather(rrig: Rig, monkeypatch) -> None:
    monkeypatch.setenv(BATCH_ROW_READ_VARIABLE, "0")
    monkeypatch.setattr(adapter_tests, "FakeDecoder", RowReadDecoder)
    rig = Rig(monkeypatch)
    assert rig.model.batch_row_read is False
    prompts, params, out, _ = _run_mixed(rig)
    assert rig.decoder.row_reads == [] and rig.decoder.full_reads == 1
    counters = rig.model.candidate_counters
    assert counters["row_read_steps"] == 0 and counters["full_read_steps"] == 1 and counters["candidate_rows"] == 4
    _assert_tokens_equal_the_full_path(rig, prompts, params, out)


def test_more_lanes_than_the_largest_bucket_read_the_full_gather(rrig: Rig) -> None:
    rrig.decoder.row_read_buckets = (1, 2)  # three fallback lanes (3, 4, 5) do not fit
    prompts, params, out, _ = _run_mixed(rrig)
    assert rrig.decoder.row_reads == [] and rrig.decoder.full_reads == 1
    counters = rrig.model.candidate_counters
    assert counters["row_read_steps"] == 0 and counters["full_read_steps"] == 1
    _assert_tokens_equal_the_full_path(rrig, prompts, params, out)


def test_a_decoder_without_the_row_read_traces_reads_the_full_gather(monkeypatch) -> None:
    class NoRowRead(RowReadDecoder):
        has_row_read = False

    monkeypatch.setattr(adapter_tests, "FakeDecoder", NoRowRead)
    rig = Rig(monkeypatch)
    prompts, params, out, _ = _run_mixed(rig)
    assert rig.decoder.row_reads == [] and rig.decoder.full_reads == 1
    _assert_tokens_equal_the_full_path(rig, prompts, params, out)


def test_a_candidate_only_step_reads_no_logits_at_all(rrig: Rig) -> None:
    prompts, blocks, firsts = adapter_tests._three(rrig)
    firsts = firsts.reshape(-1).tolist()
    params = adapter_tests._sampling(
        8, temperature=[1.0, 0.0, 0.9], top_k=[20, 1, 5], top_p=[0.95, 0.0, 1.0], seed=[1, None, 3]
    )
    rrig.decode(adapter_tests._rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5, sampling_params=params)
    assert rrig.decoder.row_reads == [] and rrig.decoder.full_reads == 0


def test_the_log_line_carries_the_read_and_draw_split_and_the_host_branches(rrig: Rig, monkeypatch) -> None:
    monkeypatch.setattr(vllm_module, "DECODE_LOG_INTERVAL", 1)
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        _run_mixed(rrig)
    finally:
        logger.remove(sink)
    line = next(text for text in lines if "batch decode:" in text)
    for part in ("fallback steps 1 of the last 1", "device read", "host draw", "row read on: 1 steps / 3 lanes"):
        assert part in line, (part, line)
    assert "full gather 0 steps" in line and "host branches:" in line
    stats = rrig.model.decode_stats[-1]
    assert stats["fallback_read_s"] > 0 and stats["fallback_draw_s"] > 0
    assert stats["fallback_s"] >= stats["fallback_read_s"] + stats["fallback_draw_s"] - 1e-9
    branches = rrig.model.host_branch_counts
    assert set(branches) == set(HOST_DRAW_BRANCHES)
    assert branches["top_k"] == 2 and sum(branches.values()) == 3  # rows 4 and 5 (top_k 50, 20); row 3 is a nucleus row


# -- Part B: the host law
# -----------------------------------------------------------------------------------------------


def old_sample_full_row(scaled, top_k, top_p, generator) -> int:
    """The code P3g-3 replaced (``_sample_full_row`` of 19804bc62c4), frozen."""

    if top_k < scaled.numel():
        scaled = scaled.masked_fill(scaled < scaled.topk(top_k).values[-1], float("-inf"))
    if top_p < 1.0:
        ordered, order = scaled.sort(descending=True)
        probs = torch.softmax(ordered, 0)
        keep = torch.cumsum(probs, 0) - probs < top_p
        keep[0] = True
        scaled = scaled.clone()
        scaled[order[~keep]] = float("-inf")
    probs = torch.softmax(scaled, 0)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int((probs / noise).argmax())


def old_sample_from_row(row, policy, generator, *, candidates: int = vllm_module.REDUCED_CANDIDATES) -> int:
    """The code P3g-3 replaced (``sample_from_row`` of 19804bc62c4), frozen."""

    if policy.greedy:
        return int(row.argmax())
    vocab = row.numel()
    top_k = policy.top_k if 0 < policy.top_k < vocab else vocab
    top_p = policy.top_p
    if top_k <= candidates:
        values, indices = row.topk(top_k)
        extra = int((row >= values[-1]).sum()) - top_k
        if extra > 0:
            values, indices = row.topk(top_k + extra)
        probs = torch.softmax(values / policy.temperature, 0)
    elif top_k == vocab:
        values, indices = row.topk(candidates)
        scaled = row / policy.temperature
        probs = torch.exp(values / policy.temperature - torch.logsumexp(scaled, 0))
        if top_p >= 1.0 or float(probs.sum()) < top_p:
            return old_sample_full_row(scaled, top_k, top_p, generator)
    else:
        return old_sample_full_row(row / policy.temperature, top_k, top_p, generator)
    if top_p < 1.0:
        keep = torch.cumsum(probs, 0) - probs < top_p
        keep[0] = True
        probs, indices = probs[keep], indices[keep]
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int(indices[(probs / noise).argmax()])


UNCHANGED_BRANCHES = ("greedy", "top_k", "nucleus_1024", "top_k_threshold")


def _random_row(seed: int, vocab: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(vocab, generator=generator) * (0.8 + 3.0 * torch.rand(1, generator=generator))
    row[torch.randperm(vocab, generator=generator)[: 4 + seed % 12]] += 6.0 + 4.0 * torch.rand(1, generator=generator)
    if seed % 3 == 0:
        row = row.to(torch.bfloat16).to(torch.float32)  # the device rows are bf16 values: ties
    return row


def test_the_unchanged_branches_draw_the_old_tokens_bit_for_bit() -> None:
    vocab = 16384
    seen: dict[str, int] = {}
    escalated = 0
    for seed in range(240):
        row = _random_row(seed, vocab)
        temperature = (0.0, 0.6, 1.0, 1.3)[seed % 4]
        top_k = (0, 20, 1024, 1500, vocab, 7, 0, 5000)[seed % 8]
        top_p = (0.95, 1.0, 0.9, 0.5, 0.0)[seed % 5]
        policy = Qwen38HostSamplingPolicy(temperature, top_k, top_p)
        branches: dict[str, int] = {}
        new = sample_from_row(row, policy, torch.Generator().manual_seed(seed), branch_counts=branches)
        (branch,) = branches
        seen[branch] = seen.get(branch, 0) + 1
        if branch in UNCHANGED_BRANCHES:
            old = old_sample_from_row(row, policy, torch.Generator().manual_seed(seed))
            assert new == old, (seed, branch, new, old)
        else:
            escalated += 1
    for branch in UNCHANGED_BRANCHES:
        assert seen.get(branch, 0) >= 10, seen
    assert escalated >= 10, seen
    logger.info(f"branches over the 240 rows: {seen}")


def test_the_top_1024_branch_is_the_old_code_on_peaked_rows() -> None:
    """The branch the server's GPQA rows take most: a peaked 248,320-wide row, top-k off, top_p 0.95, T 1."""

    same = 0
    for seed in range(40):
        row = torch.randn(VOCAB_SIZE, generator=torch.Generator().manual_seed(seed))
        row[:6] += 18.0
        row[TOKENIZER_SIZE:] = float("-inf")
        policy = Qwen38HostSamplingPolicy(1.0, VOCAB_SIZE, 0.95)
        branches: dict[str, int] = {}
        new = sample_from_row(row, policy, position_generator(seed, 5), branch_counts=branches)
        assert branches == {"nucleus_1024": 1}
        same += new == old_sample_from_row(row, policy, position_generator(seed, 5))
    assert same == 40


def _tv_noise_bound(expected: torch.Tensor, draws: int) -> float:
    """1.3 x the mean total variation distance between ``draws`` samples' empirical law and ``expected``: per token the
    mean absolute error of a binomial frequency is about ``sqrt(2 p (1 - p) / (pi n))`` (normal approximation), halved
    for TV.
    """

    p = expected.to(torch.float64)
    return 1.3 * float(0.5 * torch.sqrt(2 * p * (1 - p) / (math.pi * draws)).sum())


def test_the_full_row_nucleus_draws_the_exact_law() -> None:
    """A row of 2048 whose nucleus (top_p 0.9) is wider than the 1024 candidates: the argsorted full row.  Chi-square at
    the 1e-4 level against the law of vLLM's sampler (the exact one), and the total variation distance within the
    sampling noise of 6000 draws."""

    vocab = 2048
    row = torch.randn(vocab, generator=torch.Generator().manual_seed(3))
    policy = Qwen38HostSamplingPolicy(1.0, 0, 0.9)
    expected = _vllm_sampler_probabilities(row, temperature=1.0, top_k=0, top_p=0.9)
    draws = 6000
    generator = torch.Generator().manual_seed(11)
    branches: dict[str, int] = {}
    tokens = torch.tensor([sample_from_row(row, policy, generator, branch_counts=branches) for _ in range(draws)])
    assert branches == {"nucleus_full": draws}
    counts = torch.bincount(tokens, minlength=vocab)
    assert int(counts[expected == 0].sum()) == 0
    statistic, critical = _chi_square_ok(counts, expected, draws)
    assert statistic < critical, (statistic, critical)
    assert float((counts / draws - expected).abs().sum()) / 2 < _tv_noise_bound(expected, draws)


def test_a_nucleus_wider_than_1024_takes_the_full_branch_and_a_narrower_one_does_not() -> None:
    row = torch.randn(4096, generator=torch.Generator().manual_seed(5))
    for top_p, branch in ((0.5, "nucleus_1024"), (0.97, "nucleus_full")):
        branches: dict[str, int] = {}
        token = sample_from_row(
            row, Qwen38HostSamplingPolicy(1.0, 0, top_p), torch.Generator().manual_seed(1), branch_counts=branches
        )
        assert branches == {branch: 1} and 0 <= token < 4096
    probs = torch.softmax(row, 0).sort(descending=True).values
    assert float(probs[:1024].sum()) >= 0.5 and float(probs[:1024].sum()) < 0.97  # the premise of the two cases


def test_the_full_row_nucleus_never_draws_a_masked_token_and_takes_a_cpu_view() -> None:
    row = torch.randn(4096, generator=torch.Generator().manual_seed(8))
    row[3000:] = float("-inf")
    policy = Qwen38HostSamplingPolicy(1.0, 0, 0.999)
    generator = torch.Generator().manual_seed(1)
    branches: dict[str, int] = {}
    batch = row.repeat(2, 1)  # callers pass views such as logits[lane]
    tokens = [sample_from_row(batch[1], policy, generator, branch_counts=branches) for _ in range(1500)]
    assert max(tokens) < 3000 and set(branches) <= {"nucleus_1024", "nucleus_full"}


def test_the_whole_row_is_drawn_by_inverse_cdf_with_the_exact_law() -> None:
    vocab = 2048
    row = torch.randn(vocab, generator=torch.Generator().manual_seed(21)) * 1.2
    row[row.topk(40).indices] += 2.5
    row[1900:] = float("-inf")  # masked columns: probability zero
    expected = torch.softmax(row / 0.8, 0)
    policy = Qwen38HostSamplingPolicy(0.8, 0, 1.0)
    draws = 12000
    generator = torch.Generator().manual_seed(4)
    branches: dict[str, int] = {}
    tokens = torch.tensor([sample_from_row(row, policy, generator, branch_counts=branches) for _ in range(draws)])
    assert branches == {"inverse_cdf": draws}
    counts = torch.bincount(tokens, minlength=vocab)
    assert int(counts[1900:].sum()) == 0  # a zero-probability token is never drawn
    statistic, critical = _chi_square_ok(counts, expected, draws)
    assert statistic < critical, (statistic, critical)
    assert float((counts / draws - expected).abs().sum()) / 2 < _tv_noise_bound(expected, draws)


def test_the_inverse_cdf_clamps_to_the_last_token_of_positive_probability(monkeypatch) -> None:
    row = torch.tensor([0.0, 1.0, float("-inf"), float("-inf")])
    with monkeypatch.context() as patch:
        for variate in (0.0, 0.3, 1.0):  # 1.0: the uniform variate equals the cumulative total (the clamp case)
            patch.setattr(torch, "rand", lambda *a, _v=variate, **k: torch.tensor([_v], dtype=torch.float64))
            token = vllm_module._sample_whole_row(row, None)
            assert token in (0, 1), (variate, token)
        assert token == 1  # the variate at the total lands on the last positive token, not on a masked one
    row = torch.full((8,), float("-inf"))
    row[5] = 2.0  # one token of positive probability: every draw is that token
    generator = torch.Generator().manual_seed(2)
    assert {vllm_module._sample_whole_row(row, generator) for _ in range(50)} == {5}


def test_the_whole_row_policy_with_top_p_above_one_and_a_seeded_generator_is_deterministic() -> None:
    row = _random_row(7, 4096)
    policy = Qwen38HostSamplingPolicy(1.0, 0, 1.0)
    first = [sample_from_row(row, policy, position_generator(5, position)) for position in range(30)]
    again = [sample_from_row(row, policy, position_generator(5, position)) for position in range(30)]
    other = [sample_from_row(row, policy, position_generator(6, position)) for position in range(30)]
    assert first == again and first != other


@pytest.mark.parametrize("top_p", (0.5, 0.97, 1.0))
def test_seeded_draws_are_deterministic_on_every_branch(top_p) -> None:
    row = torch.randn(4096, generator=torch.Generator().manual_seed(12)) * 1.1
    policy = Qwen38HostSamplingPolicy(1.0, 0, top_p)
    for candidates in (16, 1024):
        runs = [
            [sample_from_row(row, policy, position_generator(9, p), candidates=candidates) for p in range(20)]
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
    assert len({sample_from_row(row, policy, position_generator(s, 3), candidates=16) for s in range(40)}) > 1


def test_the_branch_counts_name_every_branch_once_per_row() -> None:
    row = torch.randn(4096, generator=torch.Generator().manual_seed(14)) * 2.0
    counts: dict[str, int] = {}
    cases = (
        (Qwen38HostSamplingPolicy(0.0, 0, 1.0), "greedy"),
        (Qwen38HostSamplingPolicy(1.0, 20, 0.9), "top_k"),
        (Qwen38HostSamplingPolicy(1.0, 2000, 0.9), "top_k_threshold"),
        (Qwen38HostSamplingPolicy(1.0, 0, 1.0), "inverse_cdf"),
        (Qwen38HostSamplingPolicy(1.0, 0, 0.5), "nucleus_1024"),
        (Qwen38HostSamplingPolicy(1.0, 0, 0.99), "nucleus_full"),
    )
    for policy, branch in cases:
        before = dict(counts)
        sample_from_row(row, policy, torch.Generator().manual_seed(1), branch_counts=counts)
        assert sum(counts.values()) - sum(before.values()) == 1, branch
        assert counts.get(branch, 0) == before.get(branch, 0) + 1, (branch, counts)
    assert math.isclose(sum(counts.values()), len(cases))
