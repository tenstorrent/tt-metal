# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the stateless full-layer KDA reference transition."""

import torch

from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.utils import make_config, random_weights


def test_split_forward_matches_full_forward_without_mutating_input_state() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(31))

    full_output, full_state = kda_forward_reference(hidden_states, weights, config)
    first_output, first_state = kda_forward_reference(hidden_states[:, :5], weights, config)
    input_state_snapshot = tuple(tensor.clone() for tensor in first_state.__dict__.values())
    last_output, split_state = kda_forward_reference(hidden_states[:, 5:], weights, config, first_state)

    assert torch.allclose(full_output, torch.cat((first_output, last_output), dim=1), rtol=1e-5, atol=1e-5)
    for full_tensor, split_tensor in zip(full_state.__dict__.values(), split_state.__dict__.values()):
        assert torch.allclose(full_tensor, split_tensor, rtol=1e-5, atol=1e-5)
    for input_tensor, snapshot in zip(first_state.__dict__.values(), input_state_snapshot):
        assert torch.equal(input_tensor, snapshot)
