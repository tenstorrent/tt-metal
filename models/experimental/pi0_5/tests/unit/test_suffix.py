# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
No-device CPU smoke tests for the Pi0.5 suffix embedding.

Exercises the Pi0.5 (``pi05=True``) path of ``SuffixEmbedding`` with random
weights: shapes/dtypes of the embedded suffix and the adaRMS conditioning
vector, the attention-mask layout, the output projection shape, and the fact
that in Pi0.5 the time signal enters *only* through ``adarms_cond`` (the action
embeddings themselves are time-independent). Runs on CPU, no device or
pretrained weights required.
"""

import pytest
import torch

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.reference.torch_suffix import SuffixEmbedding


def _suffix_weights(cfg: SuffixConfig, g: torch.Generator) -> dict:
    ew, ad = cfg.expert_width, cfg.action_dim
    return {
        "action_in_proj.weight": torch.randn(ew, ad, generator=g) * 0.02,
        "action_in_proj.bias": torch.zeros(ew),
        "action_out_proj.weight": torch.randn(ad, ew, generator=g) * 0.02,
        "action_out_proj.bias": torch.zeros(ad),
        "time_mlp_in.weight": torch.randn(ew, ew, generator=g) * 0.02,
        "time_mlp_in.bias": torch.zeros(ew),
        "time_mlp_out.weight": torch.randn(ew, ew, generator=g) * 0.02,
        "time_mlp_out.bias": torch.zeros(ew),
    }


def _make_suffix(seed: int = 0) -> SuffixEmbedding:
    cfg = SuffixConfig(action_dim=32, action_horizon=50, expert_width=1024, pi05=True)
    return SuffixEmbedding(cfg, _suffix_weights(cfg, torch.Generator().manual_seed(seed)))


@pytest.mark.parametrize("batch", [1, 2])
def test_suffix_pi05_shapes(batch):
    suffix = _make_suffix()
    cfg = suffix.config
    state = torch.zeros(batch, cfg.state_dim)  # unused on the pi05 path
    noisy_actions = torch.randn(batch, cfg.action_horizon, cfg.action_dim)
    timestep = torch.full((batch,), 0.5)

    embs, pad_masks, att_masks, adarms_cond = suffix.embed_suffix(state, noisy_actions, timestep)

    assert embs.shape == (batch, cfg.action_horizon, cfg.expert_width)
    assert pad_masks.shape == (batch, cfg.action_horizon)
    assert att_masks.shape == (batch, cfg.action_horizon)
    assert pad_masks.dtype == torch.bool and att_masks.dtype == torch.bool
    assert bool(pad_masks.all())  # suffix tokens are all valid
    # First suffix token attends (1), the rest are masked-in via block-causal (0).
    assert bool(att_masks[:, 0].all()) and not bool(att_masks[:, 1:].any())
    # Pi0.5 produces an adaRMS conditioning vector of expert width.
    assert adarms_cond is not None
    assert adarms_cond.shape == (batch, cfg.expert_width)


def test_suffix_pi05_time_enters_only_through_adarms_cond():
    suffix = _make_suffix()
    cfg = suffix.config
    state = torch.zeros(1, cfg.state_dim)
    noisy_actions = torch.randn(1, cfg.action_horizon, cfg.action_dim)

    embs_a, _, _, cond_a = suffix.embed_suffix(state, noisy_actions, torch.full((1,), 0.1))
    embs_b, _, _, cond_b = suffix.embed_suffix(state, noisy_actions, torch.full((1,), 0.9))

    # adaRMS conditioning is time-dependent ...
    assert not torch.allclose(cond_a, cond_b)
    # ... while the action-token embeddings are not (time is injected via adaRMS).
    assert torch.allclose(embs_a, embs_b)


def test_suffix_project_output_shape():
    suffix = _make_suffix()
    cfg = suffix.config
    expert_output = torch.randn(2, cfg.action_horizon, cfg.expert_width)
    actions = suffix.project_output(expert_output)
    assert actions.shape == (2, cfg.action_horizon, cfg.action_dim)
