# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 2 (c, d): device accept logic on integer ids, the argmax tie rule, and the tok/pass arithmetic.

No device.  The accept model is the design's section 2.6 op sequence in fp32 with the FPU reduce stages
read through TF32 (the greedy-resolve precision model, silicon-confirmed: fp32 integers survive an FPU
reduce only below 2048).
"""

from __future__ import annotations

import itertools

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import (
    accept_select,
    chain_alphas,
    expected_tokens_per_pass,
    tf32,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import GREEDY_TIE_BREAK_EPS, TP_SIZE

# Ids seen on the pinned prefixes and at the vocabulary edge: below and above the 2048 TF32 limit.
SMALL_IDS = (17, 15, 16, 21, 12, 20, 11, 0, 2047)
LARGE_IDS = (2048, 95859, 62086, 248044, 248319)
R6_HISTOGRAM = [74, 75, 55, 38, 92]  # accepted drafts 0..4 over 334 rounds (r6 arm, 2026-09-02)


def _rows(pattern: tuple[int, ...], ids: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify-row argmaxes, drafts and alignment argmaxes realizing one match/mismatch pattern."""

    k = len(pattern)
    cycle = itertools.cycle(ids)
    targets = torch.tensor([next(cycle) for _ in range(k + 1)], dtype=torch.float32)
    drafts = torch.empty(k, dtype=torch.float32)
    for j, match in enumerate(pattern):
        other = next(value for value in ids if value != int(targets[j]))
        drafts[j] = targets[j] if match else other
    alignment = torch.tensor([next(cycle) for _ in range(k + 1)], dtype=torch.float32)
    return targets, drafts, alignment


def _leading_matches(pattern: tuple[int, ...]) -> int:
    return next((j for j, match in enumerate(pattern) if not match), len(pattern))


@pytest.mark.parametrize("k", (3, 4, 5))
def test_accept_count_and_gather_select_are_exact_for_every_pattern(k: int) -> None:
    for pattern in itertools.product((0, 1), repeat=k):
        for ids in (SMALL_IDS, LARGE_IDS, SMALL_IDS + LARGE_IDS):
            targets, drafts, alignment = _rows(pattern, ids)
            result = accept_select(targets, drafts, alignment)
            accepted = _leading_matches(pattern)
            assert result["accepted"] == accepted, (pattern, ids, result)
            assert result["next_token_gather"] == int(targets[accepted]), (pattern, ids, result)
            assert result["first_draft_gather"] == int(alignment[accepted]), (pattern, ids, result)


@pytest.mark.parametrize("k", (3, 4, 5))
def test_sum_select_is_exact_only_below_the_tf32_integer_limit(k: int) -> None:
    """Design 2.6 writes t' = sum(one_hot * argmax rows); through an FPU reduce that is wrong from id 2048."""

    wrong_large = 0
    for pattern in itertools.product((0, 1), repeat=k):
        small = accept_select(*_rows(pattern, SMALL_IDS))
        assert small["next_token_sum"] == small["next_token_gather"], pattern
        assert small["first_draft_sum"] == small["first_draft_gather"], pattern
        large = accept_select(*_rows(pattern, LARGE_IDS))
        wrong_large += large["next_token_sum"] != large["next_token_gather"]
        wrong_large += large["first_draft_sum"] != large["first_draft_gather"]
        # Same ops with an exact reduce (fp32 accumulate of exact inputs): the arithmetic itself is fine.
        exact = accept_select(*_rows(pattern, LARGE_IDS), fpu_rounding=lambda x: x)
        assert exact["next_token_sum"] == exact["next_token_gather"], pattern
    assert wrong_large > 0, "the TF32 model must reject the sum select on large ids"
    assert tf32(torch.tensor([95859.0, 248044.0])).tolist() == [95872.0, 248064.0]


def test_accept_count_survives_the_fpu_reduce_and_position_update_is_exact() -> None:
    flags = torch.ones(5, dtype=torch.float32)
    assert tf32(flags).sum().item() == 5.0
    for position in (0, 4, 31, 2047, 2048, 32767, 2**16 - 1):
        for accepted in range(6):
            # P <- P + a + 1 as an SFPU add on fp32 (exact below 2**24) or a uint32 add.
            assert float(position) + float(accepted) + 1.0 == float(position + accepted + 1)


def _ranked_owner(values: torch.Tensor) -> torch.Tensor:
    # resolve_greedy_on_device rule, rows batched: bf16 maxima -> fp32 -> minus owner * eps -> argmax.
    assert values.dtype == torch.bfloat16
    ranked = values.to(torch.float32) - torch.arange(TP_SIZE, dtype=torch.float32) * GREEDY_TIE_BREAK_EPS
    return torch.argmax(ranked, dim=-1)


def test_row_batched_owner_tie_break_picks_the_lowest_owner_like_resolve_greedy() -> None:
    torch.manual_seed(7)
    for _ in range(2000):
        rows = torch.randint(1, 6, ()).item()
        values = (torch.randn(rows, TP_SIZE) * 8).to(torch.bfloat16)
        values[0] = values[0, 0]  # all four owners tie on row 0
        if rows > 1:
            values[1, 2] = values[1, 1]  # a two-way tie on row 1
        expected = torch.argmax(values.to(torch.float32), dim=-1)  # first owner on ties (resolve_greedy)
        assert torch.equal(_ranked_owner(values), expected)
        ranked = values.to(torch.float32) - torch.arange(TP_SIZE, dtype=torch.float32) * GREEDY_TIE_BREAK_EPS
        # No row keeps a tie after the shift, so the device argmax kernel's tie rule cannot matter.
        top = ranked.max(dim=-1, keepdim=True).values
        assert torch.all((ranked == top).sum(dim=-1) == 1)


def test_tie_break_shift_keeps_distinct_bf16_logits_ordered() -> None:
    # Adjacent bf16 values from 2**-6 up: the larger one minus 3 * eps stays above the smaller one.
    exponents = torch.arange(-6, 8, dtype=torch.float32)
    for exponent in exponents.tolist():
        base = torch.tensor(2.0**exponent, dtype=torch.bfloat16)
        step = torch.nextafter(base.to(torch.float32), torch.tensor(float("inf")))
        larger = step.to(torch.bfloat16)
        while larger == base:
            step = torch.nextafter(step, torch.tensor(float("inf")))
            larger = step.to(torch.bfloat16)
        assert larger.to(torch.float32) - 3 * GREEDY_TIE_BREAK_EPS > base.to(torch.float32), exponent


def test_r6_histogram_reproduces_the_pooled_alphas_and_tok_per_pass() -> None:
    alphas = chain_alphas(R6_HISTOGRAM)
    assert sum(R6_HISTOGRAM) == 334
    assert [round(alpha, 3) for alpha in alphas] == [0.778, 0.712, 0.703, 0.708]
    mean_accepted = sum(count * drafts for drafts, count in enumerate(R6_HISTOGRAM)) / sum(R6_HISTOGRAM)
    assert round(mean_accepted, 2) == 2.00
    assert round(expected_tokens_per_pass(alphas, 3), 2) == 2.72
    assert round(expected_tokens_per_pass(alphas, 4), 2) == 3.00
    # With the full chain, E[tok/pass] at k=4 is exactly 1 + the mean accepted count.
    assert expected_tokens_per_pass(alphas, 4) == pytest.approx(1.0 + mean_accepted, abs=1e-12)
    # The rounded alphas of the results note give the same two decimals.
    assert round(expected_tokens_per_pass([0.778, 0.712, 0.703], 3), 2) == 2.72
    assert round(expected_tokens_per_pass([0.778, 0.712, 0.703, 0.708], 4), 2) == 3.00


def test_design_section_three_chains_reproduce_the_cost_model_rows(expect_error) -> None:
    measured = [0.791, 0.768, 0.767]  # CPU free-run chain; alpha_4|3 = alpha_5|4 = 0.767 extrapolated
    assert [round(expected_tokens_per_pass(measured, k), 3) for k in (1, 2, 3, 4, 5)] == [
        1.791,
        2.398,
        2.864,
        3.222,
        3.496,
    ]
    geometric = [0.791]
    assert [round(expected_tokens_per_pass(geometric, k), 3) for k in (1, 2, 3, 4, 5)] == [
        1.791,
        2.417,
        2.912,
        3.303,
        3.613,
    ]
    with expect_error(ValueError):  # allow-pytest.raises: pure arithmetic contract
        expected_tokens_per_pass([], 3)
