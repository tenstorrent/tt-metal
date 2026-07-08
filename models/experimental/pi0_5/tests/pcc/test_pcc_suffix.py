# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for the PI0.5 reference suffix.

This is a no-device-required smoke test that exercises the time MLP +
sincos path with random weights, and checks shapes / dtypes for the
adaRMS conditioning output. A separate PCC test against TTNN should
follow once weights / a device are available.
"""

import math

import pytest
import torch

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding


def _make_random_weights(action_dim: int, expert_width: int) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "action_in_proj.weight": torch.randn(expert_width, action_dim, generator=g) * 0.02,
        "action_in_proj.bias": torch.zeros(expert_width),
        "action_out_proj.weight": torch.randn(action_dim, expert_width, generator=g) * 0.02,
        "action_out_proj.bias": torch.zeros(action_dim),
        "time_mlp_in.weight": torch.randn(expert_width, expert_width, generator=g) * 0.02,
        "time_mlp_in.bias": torch.zeros(expert_width),
        "time_mlp_out.weight": torch.randn(expert_width, expert_width, generator=g) * 0.02,
        "time_mlp_out.bias": torch.zeros(expert_width),
    }


@pytest.mark.parametrize("batch_size", [1, 2])
def test_pi0_5_suffix_shapes(batch_size):
    action_dim = 32
    action_horizon = 50
    expert_width = 1024

    cfg = SuffixConfig(
        action_dim=action_dim,
        action_horizon=action_horizon,
        expert_width=expert_width,
        pi05=True,
    )
    weights = _make_random_weights(action_dim, expert_width)
    suffix = Pi0_5SuffixEmbedding(cfg, weights)

    noisy_actions = torch.randn(batch_size, action_horizon, action_dim)
    timestep = torch.full((batch_size,), 0.5)
    state = None  # pi0.5 has no continuous state token

    embs, pad, att, adarms = suffix.embed_suffix(state, noisy_actions, timestep)

    assert embs.shape == (batch_size, action_horizon, expert_width)
    assert pad.shape == (batch_size, action_horizon) and pad.dtype == torch.bool
    assert att.shape == (batch_size, action_horizon) and att.dtype == torch.bool
    assert adarms.shape == (batch_size, expert_width)

    out = suffix.project_output(embs)
    assert out.shape == (batch_size, action_horizon, action_dim)


def test_pi0_5_suffix_time_dependence():
    """adaRMS conditioning should differ across timesteps."""
    cfg = SuffixConfig(action_dim=32, action_horizon=50, expert_width=1024, pi05=True)
    weights = _make_random_weights(cfg.action_dim, cfg.expert_width)
    suffix = Pi0_5SuffixEmbedding(cfg, weights)

    noisy_actions = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    _, _, _, a_lo = suffix.embed_suffix(None, noisy_actions, torch.tensor([0.05]))
    _, _, _, a_hi = suffix.embed_suffix(None, noisy_actions, torch.tensor([0.95]))
    assert not torch.allclose(a_lo, a_hi, atol=1e-4)
    assert math.isfinite(a_lo.float().abs().sum().item())
