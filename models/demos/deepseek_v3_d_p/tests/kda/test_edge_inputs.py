# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tier-C edge inputs hit their targets, and the FP64 golden holds on them. CPU only."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference, l2_norm_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.fla_naive_kda import naive_recurrent_kda
from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import slice_nrmse
from models.demos.deepseek_v3_d_p.tests.kda.edge_inputs import BOUNDARY_LENGTHS, EDGE_CASES, recurrence_edge_case
from models.demos.deepseek_v3_d_p.tests.kda.real_input import chunk_decay_statistics

SHAPE = {"heads": 2, "key_dim": 32, "value_dim": 16}


@pytest.mark.parametrize("name", EDGE_CASES)
def test_edge_case_is_deterministic_in_domain_and_fractional(name: str) -> None:
    case = recurrence_edge_case(name, 64, **SHAPE, seed=3)
    again = recurrence_edge_case(name, 64, **SHAPE, seed=3)
    assert all(torch.equal(case[key], again[key]) for key in case)
    gate = case["gate"]
    assert bool(((gate >= -5.0) & (gate < 0.0)).all()), name
    # Fractional, not format-exact: no gate is exactly the bound, and values vary token to token.
    assert not bool((gate == -5.0).any()) and gate.unique().numel() > gate.numel() // 2, name


def test_edge_cases_reach_their_decay_targets() -> None:
    ceiling = chunk_decay_statistics(recurrence_edge_case("ceiling", 64, **SHAPE)["gate"][0])
    assert ceiling["p50"] > 159.0 and ceiling["max"] < 160.0
    long_memory = chunk_decay_statistics(recurrence_edge_case("long_memory", 64, **SHAPE)["gate"][0])
    assert long_memory["weak_below_2^-9"] == long_memory["chunk_channels"]
    mixed = chunk_decay_statistics(recurrence_edge_case("ceiling_with_long_memory", 64, **SHAPE)["gate"][0])
    assert mixed["weak_below_2^-9"] == 2 * 2 * 2  # 2 chunks × 2 heads × (32 // 16) long-memory channels
    assert mixed["above_144"] == mixed["chunk_channels"] - mixed["weak_below_2^-9"]


def test_repeated_keys_and_zero_values_have_their_structure() -> None:
    keys = recurrence_edge_case("repeated_keys", 33, **SHAPE)["k"]
    assert torch.equal(keys, keys[:, :1].expand_as(keys))
    zeros = recurrence_edge_case("zero_values", 33, **SHAPE)
    output, state = kda_recurrent_reference(
        *(zeros[key] for key in ("q", "k", "v", "gate", "beta", "initial_state")), dtype=torch.float64
    )
    assert not output.any() and not state.any()


@pytest.mark.parametrize("sequence", BOUNDARY_LENGTHS)
@pytest.mark.parametrize("name", EDGE_CASES)
def test_golden_matches_fla_on_every_edge_case_and_boundary_length(name: str, sequence: int) -> None:
    case = {key: tensor.double() for key, tensor in recurrence_edge_case(name, sequence, **SHAPE).items()}
    golden_output, golden_state = kda_recurrent_reference(
        *(case[key] for key in ("q", "k", "v", "gate", "beta", "initial_state")), dtype=torch.float64
    )
    fla_output, fla_state = naive_recurrent_kda(
        l2_norm_reference(case["q"], dtype=torch.float64),
        l2_norm_reference(case["k"], dtype=torch.float64),
        case["v"],
        case["gate"],
        case["beta"],
        initial_state=case["initial_state"],
        output_final_state=True,
        compute_dtype=torch.float64,
    )
    assert slice_nrmse(fla_output, golden_output, keep_dims=(2,)).max() <= 1e-12
    assert slice_nrmse(fla_state, golden_state, keep_dims=(1,)).max() <= 1e-12
