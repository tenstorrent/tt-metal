# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the stateless full-layer KDA reference transition."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
)


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_split_forward_matches_full_forward_without_mutating_input_state(use_full_rank_gate: bool) -> None:
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    hidden_states = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(31))

    full_output, full_state = kda_forward_reference(hidden_states, weights, config)
    first_output, first_state = kda_forward_reference(hidden_states[:, :5], weights, config)
    input_state_snapshot = tuple(tensor.clone() for tensor in first_state.__dict__.values())
    last_output, split_state = kda_forward_reference(hidden_states[:, 5:], weights, config, first_state)

    assert_accurate(
        full_output,
        torch.cat((first_output, last_output), dim=1),
        name="split layer output",
        pcc_threshold=0.99999,
    )
    for index, (full_tensor, split_tensor) in enumerate(
        zip(full_state.__dict__.values(), split_state.__dict__.values())
    ):
        assert_accurate(full_tensor, split_tensor, name=f"split layer state {index}", pcc_threshold=0.99999)
    for input_tensor, snapshot in zip(first_state.__dict__.values(), input_state_snapshot):
        assert_equal(snapshot, input_tensor, name="reference input state unchanged")


def test_reference_layer_is_bit_identical() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(3031))
    expected_output, expected_state = kda_forward_reference(hidden_states, weights, config)
    for iteration in range(2):
        actual_output, actual_state = kda_forward_reference(hidden_states, weights, config)
        assert_bit_identical(expected_output, actual_output, name=f"output iteration {iteration}")
        for field in expected_state.__dataclass_fields__:
            assert_bit_identical(
                getattr(expected_state, field),
                getattr(actual_state, field),
                name=f"{field} iteration {iteration}",
            )


def test_reference_layer_validates_weights(expect_error) -> None:
    config = make_config()
    weights = random_weights(config)
    del weights["q_proj.weight"]

    with expect_error(ValueError, "missing KDA weight: q_proj.weight"):
        kda_forward_reference(torch.zeros(1, 1, config.hidden_size), weights, config)
