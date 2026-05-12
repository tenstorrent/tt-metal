# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 suffix embedding (PyTorch reference).

vs PI0:
  - No state_proj token (state lives in the language tokens).
  - Time is embedded with sincos, then pushed through a small MLP
    (time_mlp_in -> swish -> time_mlp_out). The result is the per-batch
    adaRMS conditioning signal (`adarms_cond`), NOT fused into the action
    tokens.
  - Action tokens are emitted as-is from `action_in_proj`; the suffix is
    just the action tokens (length = action_horizon).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import SuffixConfig
from models.experimental.pi0.reference.torch_suffix import (
    SuffixEmbedding,
    create_sinusoidal_pos_embedding,
)


class Pi0_5SuffixEmbedding(SuffixEmbedding):
    """
    PI0.5 suffix.

    Expected weight keys in `weights`:
      - action_in_proj.{weight,bias}
      - action_out_proj.{weight,bias}
      - time_mlp_in.{weight,bias}
      - time_mlp_out.{weight,bias}
    """

    def __init__(self, config: SuffixConfig, weights: Dict[str, torch.Tensor]):
        assert config.pi05, "Pi0_5SuffixEmbedding requires config.pi05=True"
        self.config = config
        self.weights = weights

        self.action_in_weight = weights["action_in_proj.weight"]
        self.action_in_bias = weights.get("action_in_proj.bias")
        self.action_out_weight = weights["action_out_proj.weight"]
        self.action_out_bias = weights.get("action_out_proj.bias")

        self.time_mlp_in_weight = weights["time_mlp_in.weight"]
        self.time_mlp_in_bias = weights.get("time_mlp_in.bias")
        self.time_mlp_out_weight = weights["time_mlp_out.weight"]
        self.time_mlp_out_bias = weights.get("time_mlp_out.bias")

    def embed_timestep_adarms(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        sincos(t) -> Linear -> silu -> Linear -> silu -> adarms_cond
        Output shape: (batch, expert_width)

        The trailing silu matches openpi/lerobot pi05 reference
        (modeling_pi05.py time_mlp_func / pi0_pytorch.py time_mlp_func) —
        without it adarms_cond has the wrong distribution and the scale/
        shift/gate modulations produced from it flip sign of model outputs.
        """
        sincos = create_sinusoidal_pos_embedding(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
        ).to(timestep.dtype)

        w_in = self.time_mlp_in_weight.to(sincos.dtype)
        b_in = self.time_mlp_in_bias.to(sincos.dtype) if self.time_mlp_in_bias is not None else None
        x = F.linear(sincos, w_in, b_in)
        x = F.silu(x)
        w_out = self.time_mlp_out_weight.to(x.dtype)
        b_out = self.time_mlp_out_bias.to(x.dtype) if self.time_mlp_out_bias is not None else None
        x = F.linear(x, w_out, b_out)
        return F.silu(x)

    def embed_suffix(
        self,
        state: Optional[torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = noisy_actions.shape[0]
        device = noisy_actions.device

        action_emb = self.embed_actions(noisy_actions)
        adarms_cond = self.embed_timestep_adarms(timestep)

        suffix_embs = action_emb
        suffix_len = suffix_embs.shape[1]
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)

        att = [1] + [0] * (self.config.action_horizon - 1)
        suffix_att_masks = torch.tensor(att, dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks.unsqueeze(0).expand(batch_size, -1)

        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
