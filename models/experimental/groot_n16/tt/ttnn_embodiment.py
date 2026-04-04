# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Embodiment-conditioned MLPs for GR00T N1.6 - TTNN Implementation.

GR00T N1.6 uses CategorySpecificMLP and MultiEmbodimentActionEncoder
to handle up to 32 different robot embodiments with separate learned
weight matrices per embodiment.

For single-embodiment inference (common case), we select the appropriate
weight slice and run a standard MLP. This avoids the batched category
indexing overhead.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.groot_n16.common.configs import EmbodimentConfig
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
    to_tt_tensor,
)


class CategorySpecificMLPTTNN:
    """
    Per-embodiment 2-layer MLP on TTNN.

    Each embodiment has its own weight matrices. At inference time,
    we select the weights for the given embodiment_id.

    Architecture: input_dim -> hidden_dim (SiLU) -> output_dim
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: Any,
    ):
        self.device = device
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weights are CategorySpecificLinear: shape [num_categories, out, in]
        # We preprocess all embodiment weights and store them indexed
        self.layer1_weights = {}  # embodiment_id -> (weight_tt, bias_tt)
        self.layer2_weights = {}

        w1 = weights.get("layers.0.weight")  # [num_cat, hidden_dim, input_dim]
        b1 = weights.get("layers.0.bias")    # [num_cat, hidden_dim]
        w2 = weights.get("layers.1.weight")  # [num_cat, output_dim, hidden_dim]
        b2 = weights.get("layers.1.bias")    # [num_cat, output_dim]

        if w1 is not None:
            # Pre-process weights for all valid embodiments
            for cat_id in range(min(num_categories, w1.shape[0])):
                self.layer1_weights[cat_id] = (
                    preprocess_linear_weight(w1[cat_id], device),
                    preprocess_linear_bias(b1[cat_id], device) if b1 is not None else None,
                )
                self.layer2_weights[cat_id] = (
                    preprocess_linear_weight(w2[cat_id], device),
                    preprocess_linear_bias(b2[cat_id], device) if b2 is not None else None,
                )

    def __call__(
        self,
        x: ttnn.Tensor,
        embodiment_id: int = 0,
    ) -> ttnn.Tensor:
        """
        Forward pass for a single embodiment.

        Args:
            x: [batch, seq_len, input_dim]
            embodiment_id: Which embodiment's weights to use

        Returns:
            [batch, seq_len, output_dim]
        """
        w1, b1 = self.layer1_weights[embodiment_id]
        w2, b2 = self.layer2_weights[embodiment_id]

        # Layer 1 with SiLU activation
        h = ttnn.linear(
            x, w1, bias=b1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        h = ttnn.silu(h)

        # Layer 2
        out = ttnn.linear(
            h, w2, bias=b2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(h)

        return out


class TimestepEncoderTTNN:
    """
    Sinusoidal timestep encoding + MLP projection.

    Encodes continuous timestep tau into embedding of dim=input_embedding_dim (1536).
    Uses 256-channel sinusoidal features -> MLP -> 1536-dim.
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: Any,
        embedding_dim: int = 256,
        output_dim: int = 1536,
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # MLP: embedding_dim -> output_dim -> output_dim
        w1 = weights.get("mlp.0.weight")
        b1 = weights.get("mlp.0.bias")
        w2 = weights.get("mlp.2.weight")
        b2 = weights.get("mlp.2.bias")

        if w1 is not None:
            self.fc1_weight = preprocess_linear_weight(w1, device)
            self.fc1_bias = preprocess_linear_bias(b1, device) if b1 is not None else None
            self.fc2_weight = preprocess_linear_weight(w2, device)
            self.fc2_bias = preprocess_linear_bias(b2, device) if b2 is not None else None

    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings on CPU, then transfer."""
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def __call__(self, timesteps: torch.Tensor) -> ttnn.Tensor:
        """
        Encode timesteps.

        Args:
            timesteps: [batch] continuous timestep values

        Returns:
            [batch, 1, output_dim] timestep embeddings
        """
        # Sinusoidal encoding on CPU
        sin_emb = self._sinusoidal_embedding(timesteps)  # [batch, embedding_dim]
        sin_emb = sin_emb.unsqueeze(1)  # [batch, 1, embedding_dim]

        # Transfer to device
        t_tt = to_tt_tensor(sin_emb, self.device)

        # MLP with SiLU
        h = ttnn.linear(
            t_tt, self.fc1_weight, bias=self.fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(t_tt)
        h = ttnn.silu(h)

        out = ttnn.linear(
            h, self.fc2_weight, bias=self.fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(h)

        return out


class MultiEmbodimentActionEncoderTTNN:
    """
    Action encoder that fuses noisy actions with timestep embeddings.

    Architecture:
        action -> W1[embodiment] -> action_emb
        time -> sinusoidal -> MLP -> time_emb
        cat(action_emb, time_emb) -> W2[embodiment] (Swish) -> W3[embodiment] -> output

    All W1, W2, W3 are CategorySpecificLinear (per-embodiment weights).
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        config: EmbodimentConfig,
        timestep_weights: Dict[str, torch.Tensor],
        device: Any,
    ):
        self.device = device
        self.config = config

        # Per-embodiment linear layers
        # W1: action_dim -> embedding_dim
        self.w1_weights = {}
        w1 = weights.get("w1.weight")  # [num_cat, embedding_dim, action_dim]
        b1 = weights.get("w1.bias")
        if w1 is not None:
            for cat_id in range(min(config.max_num_embodiments, w1.shape[0])):
                self.w1_weights[cat_id] = (
                    preprocess_linear_weight(w1[cat_id], device),
                    preprocess_linear_bias(b1[cat_id], device) if b1 is not None else None,
                )

        # W2: 2*embedding_dim -> embedding_dim (fuses action + time)
        self.w2_weights = {}
        w2 = weights.get("w2.weight")
        b2 = weights.get("w2.bias")
        if w2 is not None:
            for cat_id in range(min(config.max_num_embodiments, w2.shape[0])):
                self.w2_weights[cat_id] = (
                    preprocess_linear_weight(w2[cat_id], device),
                    preprocess_linear_bias(b2[cat_id], device) if b2 is not None else None,
                )

        # W3: embedding_dim -> embedding_dim
        self.w3_weights = {}
        w3 = weights.get("w3.weight")
        b3 = weights.get("w3.bias")
        if w3 is not None:
            for cat_id in range(min(config.max_num_embodiments, w3.shape[0])):
                self.w3_weights[cat_id] = (
                    preprocess_linear_weight(w3[cat_id], device),
                    preprocess_linear_bias(b3[cat_id], device) if b3 is not None else None,
                )

        # Timestep encoder
        self.timestep_encoder = TimestepEncoderTTNN(
            timestep_weights, device,
            output_dim=config.action_output_dim,
        )

    def __call__(
        self,
        noisy_actions: ttnn.Tensor,
        timesteps: torch.Tensor,
        embodiment_id: int = 0,
    ) -> ttnn.Tensor:
        """
        Encode noisy actions with timestep conditioning.

        Args:
            noisy_actions: [batch, action_horizon, action_dim]
            timesteps: [batch] continuous timestep values
            embodiment_id: Which embodiment

        Returns:
            [batch, action_horizon, embedding_dim]
        """
        # Action projection
        w1, b1 = self.w1_weights[embodiment_id]
        action_emb = ttnn.linear(
            noisy_actions, w1, bias=b1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        # Timestep encoding: [batch, 1, embedding_dim]
        time_emb = self.timestep_encoder(timesteps)

        # Broadcast time_emb to match action_horizon
        # time_emb is [batch, 1, dim], action_emb is [batch, H, dim]
        # Concatenate along last dim: [batch, H, 2*dim]
        # For broadcast: repeat time_emb H times
        batch_size = noisy_actions.shape[0]
        action_horizon = noisy_actions.shape[1]

        # Expand time_emb by repeating
        time_emb_expanded = ttnn.repeat(time_emb, ttnn.Shape([1, action_horizon, 1]))

        # Concatenate action and time embeddings
        fused = ttnn.concat([action_emb, time_emb_expanded], dim=-1)
        ttnn.deallocate(action_emb)
        ttnn.deallocate(time_emb_expanded)

        # W2 with Swish (SiLU)
        w2, b2 = self.w2_weights[embodiment_id]
        h = ttnn.linear(
            fused, w2, bias=b2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(fused)
        h = ttnn.silu(h)

        # W3
        w3, b3 = self.w3_weights[embodiment_id]
        out = ttnn.linear(
            h, w3, bias=b3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(h)

        return out
