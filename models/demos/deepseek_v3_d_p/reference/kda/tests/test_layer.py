# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the stateless full-layer KDA reference transition."""

import os

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    make_config,
    random_weights,
)


def test_split_forward_matches_full_forward_without_mutating_input_state() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(31))

    full_output, full_state = kda_forward_reference(hidden_states, weights, config)
    first_output, first_state = kda_forward_reference(hidden_states[:, :5], weights, config)
    input_state_snapshot = tuple(tensor.clone() for tensor in first_state.__dict__.values())
    last_output, split_state = kda_forward_reference(hidden_states[:, 5:], weights, config, first_state)

    split_output = torch.cat((first_output, last_output), dim=1)
    assert_accurate(full_output, split_output, name="split layer output", pcc_threshold=0.99999)
    for index, (full_tensor, split_tensor) in enumerate(
        zip(full_state.__dict__.values(), split_state.__dict__.values())
    ):
        assert_accurate(full_tensor, split_tensor, name=f"split layer state {index}", pcc_threshold=0.99999)
    for input_tensor, snapshot in zip(first_state.__dict__.values(), input_state_snapshot):
        assert_equal(snapshot, input_tensor, name="reference input state unchanged")


@pytest.mark.long_running
@pytest.mark.skipif(
    os.getenv("KDA_RUN_LONG_TESTS") != "1",
    reason="set KDA_RUN_LONG_TESTS=1 to run CPU-reference determinism",
)
def test_reference_layer_determinism() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 128, config.hidden_size, generator=torch.Generator().manual_seed(3031))
    results = []
    for _ in range(3):
        output, state = kda_forward_reference(hidden_states, weights, config)
        results.append((output, *state.__dict__.values()))

    for iteration, result in enumerate(results[1:], start=1):
        for tensor_index, (expected, actual) in enumerate(zip(results[0], result)):
            assert_bit_identical(
                expected,
                actual,
                name=f"CPU reference tensor {tensor_index} iteration {iteration}",
            )
