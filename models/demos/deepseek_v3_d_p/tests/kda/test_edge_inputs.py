# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tier-C edge inputs hit the numerical edges they are named for. CPU only."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.edge_inputs import BOUNDARY_LENGTHS, EDGE_CASES, recurrence_edge_case

SHAPE = {"heads": 2, "key_dim": 32, "value_dim": 16}


@pytest.mark.parametrize("sequence", BOUNDARY_LENGTHS)
@pytest.mark.parametrize("name", EDGE_CASES)
def test_edge_case_is_deterministic_in_domain_and_fractional(name: str, sequence: int) -> None:
    case = recurrence_edge_case(name, sequence, **SHAPE, seed=3)
    again = recurrence_edge_case(name, sequence, **SHAPE, seed=3)
    assert all(torch.equal(case[key], again[key]) for key in case)
    assert case["q"].shape == (1, sequence, 2, 32) and case["v"].shape == (1, sequence, 2, 16)
    gate = case["gate"]
    assert bool(((gate >= -5.0) & (gate < 0.0)).all()), name  # inside K3's bounded gate -5 * sigmoid(x)
    assert not bool((gate == -5.0).any()), name  # fractional: never the format-exact bound


def test_ceiling_case_decays_just_under_160_per_32_token_chunk() -> None:
    gate = recurrence_edge_case("ceiling", 64, **SHAPE)["gate"]
    chunk_decay = -gate.reshape(1, 2, 32, 2, 32).sum(dim=2)  # |G| of each 32-token chunk, per head and channel
    assert bool((chunk_decay > 159.0).all()) and bool((chunk_decay < 160.0).all())


def test_long_memory_decay_rounds_to_one_in_bf16() -> None:
    gate = recurrence_edge_case("long_memory", 64, **SHAPE)["gate"]
    assert bool((torch.exp(gate).bfloat16() == 1.0).all())
    assert bool((gate < 0.0).all())  # yet every token still decays in FP32


def test_ceiling_with_long_memory_mixes_both_in_each_head() -> None:
    case = recurrence_edge_case("ceiling_with_long_memory", 64, **SHAPE)
    long_memory = 32 // 16  # key_dim // 16 channels
    gate = case["gate"]
    assert bool((torch.exp(gate[..., :long_memory]).bfloat16() == 1.0).all())
    assert bool((gate[..., long_memory:] < -4.98).all())
    beta = torch.sigmoid(case["beta_logits"])
    assert bool((beta > 0.97).all()) and bool((beta < 0.99).all())


def test_repeated_keys_and_zero_values_have_their_structure() -> None:
    case = recurrence_edge_case("repeated_keys", 33, **SHAPE)
    assert torch.equal(case["k"], case["k"][:, :1].expand_as(case["k"]))
    assert bool((torch.sigmoid(case["beta_logits"]) >= 0.9).all())
    zeros = recurrence_edge_case("zero_values", 33, **SHAPE)
    output, state = kda_recurrent_reference(
        zeros["q"],
        zeros["k"],
        zeros["v"],
        zeros["gate"],
        torch.sigmoid(zeros["beta_logits"]),
        zeros["initial_state"],
        dtype=torch.float64,
    )
    assert not output.any() and not state.any()


def test_unknown_case_is_rejected(expect_error) -> None:
    with expect_error(ValueError, "unknown edge case"):
        recurrence_edge_case("nonexistent", 16, **SHAPE)
