# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Embodiment-conditioned MLPs for GR00T N1.6 - TTNN Implementation.

Weight key patterns (from actual model):
    State encoder:  layer1.W [num_cat, hidden, input], layer1.b [num_cat, hidden]
                    layer2.W [num_cat, output, hidden], layer2.b [num_cat, output]
    Action encoder: W1.W, W1.b, W2.W, W2.b, W3.W, W3.b
    Action decoder: layer1.W, layer1.b, layer2.W, layer2.b

For single-embodiment inference, we select the weight slice for the given
embodiment_id and run a standard MLP.
"""

import math
from typing import Any, Dict, Optional

import torch

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
    Per-embodiment 2-layer MLP: input -> hidden (SiLU) -> output.

    Weight keys: layer1.W [num_cat, hidden, input], layer1.b [num_cat, hidden],
                 layer2.W [num_cat, output, hidden], layer2.b [num_cat, output]
    """

    def __init__(self, weights: Dict[str, torch.Tensor],
                 num_categories: int, input_dim: int, hidden_dim: int,
                 output_dim: int, device: Any):
        self.device = device
        self.layer1_weights = {}
        self.layer2_weights = {}

        w1 = weights.get("layer1.W")  # [num_cat, hidden_dim, input_dim]
        b1 = weights.get("layer1.b")  # [num_cat, hidden_dim]
        w2 = weights.get("layer2.W")  # [num_cat, output_dim, hidden_dim]
        b2 = weights.get("layer2.b")  # [num_cat, output_dim]

        if w1 is not None:
            for cat_id in range(min(num_categories, w1.shape[0])):
                self.layer1_weights[cat_id] = (
                    preprocess_linear_weight(w1[cat_id], device),
                    preprocess_linear_bias(b1[cat_id], device) if b1 is not None else None,
                )
                self.layer2_weights[cat_id] = (
                    preprocess_linear_weight(w2[cat_id], device),
                    preprocess_linear_bias(b2[cat_id], device) if b2 is not None else None,
                )

    def __call__(self, x: ttnn.Tensor, embodiment_id: int = 0) -> ttnn.Tensor:
        """Forward for a single embodiment: x -> SiLU(W1*x + b1) -> W2*h + b2."""
        w1, b1 = self.layer1_weights[embodiment_id]
        w2, b2 = self.layer2_weights[embodiment_id]

        h = ttnn.linear(x, w1, bias=b1, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        h = ttnn.silu(h)

        out = ttnn.linear(h, w2, bias=b2, memory_config=ttnn.L1_MEMORY_CONFIG,
                          dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(h)
        return out


class TimestepEncoderTTNN:
    """
    Sinusoidal timestep encoding + MLP.

    Weight keys: timestep_embedder.linear_1.{weight,bias},
                 timestep_embedder.linear_2.{weight,bias}
    """

    def __init__(self, weights: Dict[str, torch.Tensor], device: Any,
                 embedding_dim: int = 256, output_dim: int = 1536):
        self.device = device
        self.embedding_dim = embedding_dim

        w1 = weights.get("timestep_embedder.linear_1.weight")
        b1 = weights.get("timestep_embedder.linear_1.bias")
        w2 = weights.get("timestep_embedder.linear_2.weight")
        b2 = weights.get("timestep_embedder.linear_2.bias")

        if w1 is not None:
            self.fc1_weight = preprocess_linear_weight(w1, device)
            self.fc1_bias = preprocess_linear_bias(b1, device) if b1 is not None else None
            self.fc2_weight = preprocess_linear_weight(w2, device)
            self.fc2_bias = preprocess_linear_bias(b2, device) if b2 is not None else None

    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def __call__(self, timesteps: torch.Tensor) -> ttnn.Tensor:
        """Encode timesteps -> [batch, 1, output_dim]."""
        sin_emb = self._sinusoidal_embedding(timesteps).unsqueeze(1)
        t_tt = to_tt_tensor(sin_emb, self.device)

        h = ttnn.linear(t_tt, self.fc1_weight, bias=self.fc1_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(t_tt)
        h = ttnn.silu(h)

        out = ttnn.linear(h, self.fc2_weight, bias=self.fc2_bias,
                          memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(h)
        return out


class MultiEmbodimentActionEncoderTTNN:
    """
    Action encoder fusing noisy actions with timestep embeddings.

    Weight keys: W1.W, W1.b (action projection)
                 W2.W, W2.b (fuse action+time, Swish)
                 W3.W, W3.b (final projection)

    Architecture:
        action -> W1[emb] -> action_emb
        time -> sinusoidal -> MLP -> time_emb
        cat(action_emb, time_emb) -> W2[emb] (SiLU) -> W3[emb] -> output
    """

    def __init__(self, weights: Dict[str, torch.Tensor], config: EmbodimentConfig,
                 timestep_weights: Dict[str, torch.Tensor], device: Any):
        self.device = device
        self.config = config

        # Per-embodiment linear layers
        self.w1 = {}
        self.w2 = {}
        self.w3 = {}

        for name, storage in [("W1", self.w1), ("W2", self.w2), ("W3", self.w3)]:
            w = weights.get(f"{name}.W")  # [num_cat, out_dim, in_dim]
            b = weights.get(f"{name}.b")  # [num_cat, out_dim]
            if w is not None:
                for cat_id in range(min(config.max_num_embodiments, w.shape[0])):
                    storage[cat_id] = (
                        preprocess_linear_weight(w[cat_id], device),
                        preprocess_linear_bias(b[cat_id], device) if b is not None else None,
                    )

        self.timestep_encoder = TimestepEncoderTTNN(
            timestep_weights, device, output_dim=config.action_output_dim,
        )

    def __call__(self, noisy_actions: ttnn.Tensor, timesteps: torch.Tensor,
                 embodiment_id: int = 0) -> ttnn.Tensor:
        """
        Encode noisy actions with timestep conditioning.

        Args:
            noisy_actions: [B, action_horizon, action_dim]
            timesteps: [B] continuous timestep values
            embodiment_id: Which robot embodiment
        Returns:
            [B, action_horizon, embedding_dim]
        """
        # Action projection via W1
        w1, b1 = self.w1[embodiment_id]
        action_emb = ttnn.linear(noisy_actions, w1, bias=b1,
                                 memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)

        # Timestep encoding: [B, 1, dim]
        time_emb = self.timestep_encoder(timesteps)

        # Broadcast time to action horizon length
        action_horizon = noisy_actions.shape[1]
        time_emb_expanded = ttnn.repeat(time_emb, ttnn.Shape([1, action_horizon, 1]))

        # Concatenate and fuse
        fused = ttnn.concat([action_emb, time_emb_expanded], dim=-1)
        ttnn.deallocate(action_emb)
        ttnn.deallocate(time_emb_expanded)

        # W2 with SiLU
        w2, b2 = self.w2[embodiment_id]
        h = ttnn.linear(fused, w2, bias=b2, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(fused)
        h = ttnn.silu(h)

        # W3
        w3, b3 = self.w3[embodiment_id]
        out = ttnn.linear(h, w3, bias=b3, memory_config=ttnn.L1_MEMORY_CONFIG,
                          dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(h)
        return out
