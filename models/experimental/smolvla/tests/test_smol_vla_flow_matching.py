# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks that the SmolVLA action head follows LeRobot's flow-matching formulation
(create_sinusoidal_pos_embedding + embed_suffix). No device needed.

Run with:
    pytest models/experimental/smolvla/tests/test_smol_vla_flow_matching.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from models.experimental.smolvla.tt.smol_vla import SmolVLAActionHeads, get_timestep_embedding

EXPERT_DIM = 720
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0


def lerobot_time_embedding(t, dim):
    # Written out independently of the module under test: 2*pi/period, dim // 2 log-spaced periods, [sin | cos].
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    angle = t.double()[:, None] * (1.0 / period * 2 * math.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1).float()


@pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_time_embedding_matches_lerobot(t):
    timesteps = torch.tensor([t, 1.0 - t])
    got = get_timestep_embedding(timesteps, EXPERT_DIM, MIN_PERIOD, MAX_PERIOD)
    assert got.shape == (2, EXPERT_DIM)
    torch.testing.assert_close(got, lerobot_time_embedding(timesteps, EXPERT_DIM), rtol=0, atol=1e-6)


def test_embed_suffix_concatenates_action_and_time():
    torch.manual_seed(0)
    heads = SmolVLAActionHeads(expert_hidden_dim=EXPERT_DIM)
    heads.eval()
    assert heads.action_time_mlp_in.in_features == 2 * EXPERT_DIM
    noisy_actions = torch.randn(1, 50, 32)
    timestep = torch.tensor([0.3])
    with torch.no_grad():
        action_emb = heads.action_in_proj(noisy_actions)
        time_emb = lerobot_time_embedding(timestep, EXPERT_DIM)[:, None, :].expand_as(action_emb)
        hidden = heads.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))
        expected = heads.action_time_mlp_out(F.silu(hidden))
        got = heads(None, None, noisy_actions, timestep)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_embed_suffix_depends_on_time():
    torch.manual_seed(0)
    heads = SmolVLAActionHeads(expert_hidden_dim=EXPERT_DIM)
    heads.eval()
    noisy_actions = torch.randn(1, 50, 32)
    with torch.no_grad():
        early = heads(None, None, noisy_actions, torch.tensor([0.05]))
        late = heads(None, None, noisy_actions, torch.tensor([0.95]))
    assert (early - late).abs().max() > 1e-3
