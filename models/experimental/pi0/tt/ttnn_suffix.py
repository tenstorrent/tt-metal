# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Embedding module - TTNN Implementation.

This module handles embedding of state, noisy actions, and timestep using
TTNN operations for efficient execution on Tenstorrent hardware.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import SuffixConfig


def create_sinusoidal_pos_embedding_ttnn(
    time: ttnn.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: ttnn.Device = None,
) -> ttnn.Tensor:
    """Create sinusoidal positional embeddings for timesteps (TTNN version)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    
    if device is None:
        device = time.device()
    
    # Compute frequencies on host
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * 2 * math.pi
    
    # Convert to TTNN
    scaling_factor_ttnn = ttnn.from_torch(
        scaling_factor.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    
    # Reshape time for broadcasting
    time_reshaped = ttnn.reshape(time, (-1, 1))
    
    # Compute sin input
    sin_input = ttnn.matmul(time_reshaped, scaling_factor_ttnn)
    
    # Compute sin and cos
    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)
    
    # Concatenate
    return ttnn.concat([sin_emb, cos_emb], dim=-1)


class TtSuffixEmbedding:
    """TTNN implementation of suffix embedding."""
    
    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, ttnn.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        self.weights = weights
    
    def embed_actions(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """Embed noisy actions using ttnn.linear."""
        return ttnn.linear(
            noisy_actions,
            self.weights["action_in_proj.weight"],
            bias=self.weights["action_in_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    
    def embed_state(self, state: ttnn.Tensor) -> Optional[ttnn.Tensor]:
        """Embed robot state (PI0 only)."""
        if self.config.pi05:
            return None
        
        state_emb = ttnn.linear(
            state,
            self.weights["state_proj.weight"],
            bias=self.weights["state_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        
        # Add sequence dimension
        shape = state_emb.shape
        return ttnn.reshape(state_emb, (shape[0], 1, shape[-1]))
    
    def embed_timestep(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """Create timestep embedding."""
        return create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
        )
    
    def fuse_action_time(
        self,
        action_emb: ttnn.Tensor,
        time_emb: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Fuse action and time embeddings using TTNN operations."""
        if self.config.pi05:
            return action_emb, time_emb
        
        # Get shapes
        batch_size = action_emb.shape[0]
        action_horizon = action_emb.shape[1]
        
        # Expand time embedding
        time_expanded = ttnn.reshape(time_emb, (batch_size, 1, -1))
        time_expanded = ttnn.repeat(time_expanded, (1, action_horizon, 1))
        
        # Concatenate
        concat = ttnn.concat([action_emb, time_expanded], dim=-1)
        
        # MLP: Linear -> SiLU -> Linear
        x = ttnn.linear(
            concat,
            self.weights["action_time_mlp_in.weight"],
            bias=self.weights["action_time_mlp_in.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.silu(x)
        x = ttnn.linear(
            x,
            self.weights["action_time_mlp_out.weight"],
            bias=self.weights["action_time_mlp_out.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        
        return x, None
    
    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Create suffix embeddings using TTNN operations."""
        batch_size = noisy_actions.shape[0]
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
        
        # Concatenate
        if len(embs) > 1:
            suffix_embs = ttnn.concat(embs, dim=1)
        else:
            suffix_embs = embs[0]
        
        # Create masks
        suffix_len = suffix_embs.shape[1]
        
        suffix_pad_masks = ttnn.ones(
            (batch_size, suffix_len),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        att_mask_tensor = torch.tensor(att_masks, dtype=torch.bool)
        att_mask_tensor = att_mask_tensor.unsqueeze(0).expand(batch_size, -1)
        suffix_att_masks = ttnn.from_torch(
            att_mask_tensor.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
    
    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
        """Project expert output back to action dimension."""
        return ttnn.linear(
            expert_output,
            self.weights["action_out_proj.weight"],
            bias=self.weights["action_out_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


def convert_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: ttnn.Device,
    dtype: ttnn.DataType = None,
) -> Dict[str, ttnn.Tensor]:
    """Convert PyTorch suffix weights to TTNN format."""
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype
    
    ttnn_weights = {}
    
    for key, value in torch_weights.items():
        if "bias" in key:
            ttnn_weights[key] = ttnn.from_torch(
                value.unsqueeze(0),
                dtype=bias_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            ttnn_weights[key] = ttnn.from_torch(
                value.T.contiguous(),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
    
    return ttnn_weights

