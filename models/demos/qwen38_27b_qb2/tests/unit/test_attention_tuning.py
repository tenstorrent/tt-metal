# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Causal bounds, page mappings and tuning selection must survive batching."""

import torch

from models.demos.qwen38_27b_qb2.tests.attention_tuning import (
    CASES,
    CHUNKS,
    accuracy,
    geometry,
    reference,
    select_candidate,
)


def test_alignment_preserves_context_and_covers_every_candidate_tail():
    for length, batch in CASES:
        plan = geometry(length, batch)
        assert length + 127 <= plan["aligned_capacity"] <= 262144
        assert plan["native_capacity"] <= plan["aligned_capacity"]
        for chunk in CHUNKS:
            assert plan["aligned_capacity"] % chunk == 0
            assert all(
                (position + chunk) // chunk * chunk <= plan["aligned_capacity"] for position in plan["positions"]
            )
    assert geometry(55000, 1)["native_chunk"] == 32
    assert geometry(131072, 8)["extra_pool_tokens"] == 3072


def test_cpu_reference_honors_shuffled_pages_and_independent_causal_bounds():
    query = torch.zeros(1, 2, 6, 4)
    key = torch.zeros(4, 1, 32, 4)
    value = torch.full_like(key, 99)
    table = torch.tensor([[3, 0], [1, 2]])
    positions = [4, 39]
    value[3, 0, :5] = 1
    value[1] = 7
    value[2, 0, :8] = 7
    output = reference(query, key, value, table, positions)
    torch.testing.assert_close(output[0, 0], torch.ones(6, 4))
    torch.testing.assert_close(output[0, 1], torch.full((6, 4), 7.0))


def test_accuracy_rejects_scaled_output_and_one_bad_user():
    expected = torch.randn(1, 4, 6, 16, generator=torch.Generator().manual_seed(42))
    assert accuracy(expected, expected)["passed"]
    assert not accuracy(expected * 2, expected)["passed"]  # Correlation alone would pass.
    damaged = expected.clone()
    damaged[:, 3] = -damaged[:, 3]
    assert not accuracy(damaged, expected)["passed"]


def test_fast_but_inaccurate_candidate_cannot_win_and_drift_is_explicit():
    candidates = [
        dict(chunk=128, accuracy_passed=True, median_traced_call_us=20),
        dict(chunk=256, accuracy_passed=False, median_traced_call_us=5),
        dict(chunk=512, accuracy_passed=True, median_traced_call_us=10),
    ]
    result = select_candidate(candidates, 128, [20, 20, 20])
    assert result["candidate_chunk"] == 512 and result["timing_comparison_qualified"]
    assert not result["promoted_to_model"]
    assert not select_candidate(candidates, 128, [24, 24, 24])["timing_comparison_qualified"]
