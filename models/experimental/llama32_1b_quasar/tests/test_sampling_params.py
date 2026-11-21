# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.llama32_1b_quasar.sampling import (
    SamplingParams,
    format_sampling_params,
    per_slot_sampling_params,
)


def test_per_request_sampling_params_retained():
    """Verify that per-request sampling parameters are preserved across lanes."""
    B = 32
    params = SamplingParams(
        temperature=[0.2, 0.5, 0.7, 0.9],
        top_k=[5, 10, 15, 20],
        top_p=[0.8, 0.85, 0.9, 0.95],
        seed=[101, 102, 103, 104],
    )

    formatted = format_sampling_params(per_slot_sampling_params(params, B), B)

    # Lanes 0..3 must preserve their distinct per-request values
    # (temperature is inverted during formatting: 1 / temp)
    assert formatted.temperature[0] == pytest.approx(1.0 / 0.2)
    assert formatted.temperature[1] == pytest.approx(1.0 / 0.5)
    assert formatted.temperature[2] == pytest.approx(1.0 / 0.7)
    assert formatted.temperature[3] == pytest.approx(1.0 / 0.9)

    assert formatted.top_k[0] == 5
    assert formatted.top_k[1] == 10
    assert formatted.top_k[2] == 15
    assert formatted.top_k[3] == 20

    assert formatted.top_p[0] == pytest.approx(0.8)
    assert formatted.top_p[1] == pytest.approx(0.85)
    assert formatted.top_p[2] == pytest.approx(0.9)
    assert formatted.top_p[3] == pytest.approx(0.95)

    assert formatted.seed[0] == 101
    assert formatted.seed[1] == 102
    assert formatted.seed[2] == 103
    assert formatted.seed[3] == 104

    assert len(formatted.temperature) == B
    assert len(formatted.top_k) == B
    assert len(formatted.top_p) == B
    assert len(formatted.seed) == B


def test_scalar_non_greedy_sampling_params():
    """Verify that scalar non-greedy params broadcast across all 32 lanes."""
    B = 32
    params = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        seed=7,
        repetition_penalty=1.2,
    )

    formatted = format_sampling_params(per_slot_sampling_params(params, B), B)

    assert len(formatted.temperature) == B
    assert len(formatted.top_k) == B
    assert len(formatted.top_p) == B
    assert len(formatted.seed) == B
    assert len(formatted.repetition_penalty) == B

    # Every one of the 32 lanes gets 1/0.7, min(50, 32)=32, 0.9, 7, and repetition penalty
    assert all(t == pytest.approx(1.0 / 0.7) for t in formatted.temperature)
    assert all(k == min(50, 32) for k in formatted.top_k)
    assert all(p == pytest.approx(0.9) for p in formatted.top_p)
    assert all(s == 7 for s in formatted.seed)
    assert all(r == pytest.approx(1.2) for r in formatted.repetition_penalty)


def test_mixed_sampling_params():
    """Verify mixed list and scalar params: lists are preserved for lanes 0..3, scalar seed replicated."""
    B = 32
    params = SamplingParams(
        temperature=[0.2, 0.5, 0.7, 0.9],
        top_k=[5, 10, 15, 20],
        top_p=[0.8, 0.85, 0.9, 0.95],
        seed=7,
    )

    formatted = format_sampling_params(per_slot_sampling_params(params, B), B)

    # Lanes 0..3 keep their lists
    assert formatted.temperature[0] == pytest.approx(1.0 / 0.2)
    assert formatted.temperature[1] == pytest.approx(1.0 / 0.5)
    assert formatted.temperature[2] == pytest.approx(1.0 / 0.7)
    assert formatted.temperature[3] == pytest.approx(1.0 / 0.9)

    assert formatted.top_k[0] == 5
    assert formatted.top_k[1] == 10
    assert formatted.top_k[2] == 15
    assert formatted.top_k[3] == 20

    assert formatted.top_p[0] == pytest.approx(0.8)
    assert formatted.top_p[1] == pytest.approx(0.85)
    assert formatted.top_p[2] == pytest.approx(0.9)
    assert formatted.top_p[3] == pytest.approx(0.95)

    # Seed is 7 on every lane
    assert len(formatted.seed) == B
    assert all(s == 7 for s in formatted.seed)
