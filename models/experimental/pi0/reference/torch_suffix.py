# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Embedding module - PyTorch Reference Implementation.

This module handles embedding of state, noisy actions, and timestep to create
the suffix part of the sequence for expert transformer processing.

Components:
    - action_in_proj: Projects actions from action_dim to expert width
    - action_out_proj: Projects expert output back to action_dim
    - state_proj: Projects state from state_dim to expert width (PI0 only)
    - action_time_mlp: Fuses action and time embeddings (PI0 only)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import SuffixConfig


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Create sinusoidal positional embeddings for timesteps."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    device = time.device
    dtype = torch.float64 if device.type == "cpu" else time.dtype

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    embeddings = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

    return embeddings.to(time.dtype)


def safe_cat(tensors: list, dim: int = -1) -> torch.Tensor:
    """Safely concatenate tensors with dtype handling."""
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    target_dtype = tensors[0].dtype
    converted = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in tensors]
    return torch.cat(converted, dim=dim)


class SuffixEmbedding:
    """
    PyTorch implementation of suffix embedding.

    Embeds state + noisy actions + timestep for the action expert.
    """

    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize suffix embedding.

        Args:
            config: Suffix configuration
            weights: Dictionary with projection weights
        """
        self.config = config
        self.weights = weights

        # Extract weights
        self.action_in_weight = weights["action_in_proj.weight"]
        self.action_in_bias = weights["action_in_proj.bias"]
        self.action_out_weight = weights["action_out_proj.weight"]
        self.action_out_bias = weights["action_out_proj.bias"]

        if not config.pi05:
            self.state_weight = weights["state_proj.weight"]
            self.state_bias = weights["state_proj.bias"]
            self.time_mlp_in_weight = weights["action_time_mlp_in.weight"]
            self.time_mlp_in_bias = weights["action_time_mlp_in.bias"]
            self.time_mlp_out_weight = weights["action_time_mlp_out.weight"]
            self.time_mlp_out_bias = weights["action_time_mlp_out.bias"]

    def embed_actions(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        """Embed noisy actions."""
        action_in_weight = self.action_in_weight.to(noisy_actions.dtype)
        action_in_bias = self.action_in_bias.to(noisy_actions.dtype) if self.action_in_bias is not None else None
        return F.linear(noisy_actions, action_in_weight, action_in_bias)

    def embed_state(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Embed robot state (PI0 only)."""
        if self.config.pi05:
            return None

        state_weight = self.state_weight.to(state.dtype)
        state_bias = self.state_bias.to(state.dtype) if self.state_bias is not None else None
        state_emb = F.linear(state, state_weight, state_bias)
        return state_emb.unsqueeze(1)  # Add sequence dimension

    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Create timestep embedding using sinusoidal encoding."""
        return create_sinusoidal_pos_embedding(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
        ).to(timestep.dtype)

    def fuse_action_time(
        self,
        action_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse action and time embeddings."""
        if self.config.pi05:
            return action_emb, time_emb
        else:
            # PI0: Concatenate action and time, apply MLP
            time_expanded = time_emb.unsqueeze(1).expand_as(action_emb)
            concat = torch.cat([action_emb, time_expanded], dim=-1)

            # Apply MLP: Linear -> SiLU -> Linear
            time_mlp_in_weight = self.time_mlp_in_weight.to(concat.dtype)
            time_mlp_in_bias = self.time_mlp_in_bias.to(concat.dtype) if self.time_mlp_in_bias is not None else None
            x = F.linear(concat, time_mlp_in_weight, time_mlp_in_bias)
            x = F.silu(x)
            time_mlp_out_weight = self.time_mlp_out_weight.to(x.dtype)
            time_mlp_out_bias = self.time_mlp_out_bias.to(x.dtype) if self.time_mlp_out_bias is not None else None
            x = F.linear(x, time_mlp_out_weight, time_mlp_out_bias)

            return x, None

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Main embedding function for suffix.

        Returns:
            Tuple of (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond)
        """
        batch_size = noisy_actions.shape[0]
        device = noisy_actions.device

        embs = []
        att_masks = []

        # Embed state (PI0 only)
        if not self.config.pi05:
            state_emb = self.embed_state(state)
            if state_emb is not None:
                embs.append(state_emb)
                att_masks.append(1)

        # Embed timestep
        time_emb = self.embed_timestep(timestep)

        # Embed actions
        action_emb = self.embed_actions(noisy_actions)

        # Fuse action and time
        action_time_emb, adarms_cond = self.fuse_action_time(action_emb, time_emb)

        embs.append(action_time_emb)
        att_masks.append(1)
        att_masks.extend([0] * (self.config.action_horizon - 1))

        # Concatenate embeddings
        suffix_embs = safe_cat(embs, dim=1)

        # Create masks
        suffix_len = suffix_embs.shape[1]
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        suffix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks.unsqueeze(0).expand(batch_size, -1)

        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    def project_output(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Project expert output back to action dimension."""
        action_out_weight = self.action_out_weight.to(expert_output.dtype)
        action_out_bias = self.action_out_bias.to(expert_output.dtype) if self.action_out_bias is not None else None
        return F.linear(expert_output, action_out_weight, action_out_bias)
