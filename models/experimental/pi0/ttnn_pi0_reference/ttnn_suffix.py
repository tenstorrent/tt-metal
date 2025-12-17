# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Embedding module for TTNN PI0 implementation.

This module handles embedding of state, noisy actions, and timestep to create
the suffix part of the sequence for expert transformer processing.

Components:
    - action_in_proj: Projects actions from action_dim to expert width
    - action_out_proj: Projects expert output back to action_dim
    - state_proj: Projects state from state_dim to expert width (PI0 only)
    - action_time_mlp: Fuses action and time embeddings (PI0 only)
    - time_mlp: Processes time embeddings for adaRMS (PI05 only)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from .ttnn_common import (
    create_sinusoidal_pos_embedding_torch,
    create_sinusoidal_pos_embedding_ttnn,
    safe_cat_torch,
)


@dataclass
class SuffixConfig:
    """Configuration for suffix embedding."""
    action_dim: int = 32
    action_horizon: int = 50
    expert_width: int = 1024
    pi05: bool = False  # PI05 uses different time handling


class SuffixEmbeddingTorch:
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
            weights: Dictionary with projection weights:
                - action_in_proj.weight, action_in_proj.bias
                - action_out_proj.weight, action_out_proj.bias
                - state_proj.weight, state_proj.bias (PI0 only)
                - action_time_mlp_in.weight, action_time_mlp_in.bias (PI0 only)
                - action_time_mlp_out.weight, action_time_mlp_out.bias (PI0 only)
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
        """
        Embed noisy actions.
        
        Args:
            noisy_actions: (batch_size, action_horizon, action_dim)
        
        Returns:
            (batch_size, action_horizon, expert_width)
        """
        return F.linear(noisy_actions, self.action_in_weight, self.action_in_bias)
    
    def embed_state(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Embed robot state (PI0 only).
        
        Args:
            state: (batch_size, state_dim)
        
        Returns:
            (batch_size, 1, expert_width) or None for PI05
        """
        if self.config.pi05:
            return None
        
        state_emb = F.linear(state, self.state_weight, self.state_bias)
        return state_emb.unsqueeze(1)  # Add sequence dimension
    
    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Create timestep embedding using sinusoidal encoding.
        
        Args:
            timestep: (batch_size,) with values in [0, 1]
        
        Returns:
            (batch_size, expert_width)
        """
        return create_sinusoidal_pos_embedding_torch(
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
        """
        Fuse action and time embeddings.
        
        For PI0: Concatenate and apply MLP
        For PI05: Process time separately for adaRMS
        
        Args:
            action_emb: (batch_size, action_horizon, expert_width)
            time_emb: (batch_size, expert_width)
        
        Returns:
            Tuple of (fused_emb, adarms_cond):
                - fused_emb: (batch_size, action_horizon, expert_width)
                - adarms_cond: (batch_size, expert_width) for PI05, None for PI0
        """
        if self.config.pi05:
            # PI05: Return action embedding directly, time goes to adaRMS
            # Note: time_mlp would be applied here for PI05
            return action_emb, time_emb
        else:
            # PI0: Concatenate action and time, apply MLP
            # Expand time to match action sequence length
            time_expanded = time_emb.unsqueeze(1).expand_as(action_emb)
            
            # Concatenate along feature dimension
            concat = torch.cat([action_emb, time_expanded], dim=-1)
            
            # Apply MLP: Linear -> SiLU -> Linear
            x = F.linear(concat, self.time_mlp_in_weight, self.time_mlp_in_bias)
            x = F.silu(x)
            x = F.linear(x, self.time_mlp_out_weight, self.time_mlp_out_bias)
            
            return x, None
    
    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Main embedding function for suffix (state + actions + timestep).
        
        Args:
            state: (batch_size, state_dim)
            noisy_actions: (batch_size, action_horizon, action_dim)
            timestep: (batch_size,) with values in [0, 1]
        
        Returns:
            Tuple of (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond):
                - suffix_embs: (batch_size, suffix_len, expert_width)
                - suffix_pad_masks: (batch_size, suffix_len)
                - suffix_att_masks: (batch_size, suffix_len)
                - adarms_cond: For PI05 adaRMS conditioning
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
                # State token starts causal attention boundary
                att_masks.append(1)
        
        # Embed timestep
        time_emb = self.embed_timestep(timestep)
        
        # Embed actions
        action_emb = self.embed_actions(noisy_actions)
        
        # Fuse action and time
        action_time_emb, adarms_cond = self.fuse_action_time(action_emb, time_emb)
        
        # Add action-time embeddings
        embs.append(action_time_emb)
        
        # First action token continues causal boundary, rest are causal
        att_masks.append(1)
        att_masks.extend([0] * (self.config.action_horizon - 1))
        
        # Concatenate embeddings
        suffix_embs = safe_cat_torch(embs, dim=1)
        
        # Create masks
        suffix_len = suffix_embs.shape[1]
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        suffix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks.unsqueeze(0).expand(batch_size, -1)
        
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
    
    def project_output(self, expert_output: torch.Tensor) -> torch.Tensor:
        """
        Project expert output back to action dimension.
        
        Args:
            expert_output: (batch_size, action_horizon, expert_width)
        
        Returns:
            (batch_size, action_horizon, action_dim)
        """
        return F.linear(expert_output, self.action_out_weight, self.action_out_bias)


class SuffixEmbeddingTTNN:
    """
    TTNN implementation of suffix embedding.
    
    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """
    
    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, "ttnn.Tensor"],
        device: "ttnn.Device",
    ):
        """
        Initialize suffix embedding with TTNN weights.
        
        Args:
            config: Suffix configuration
            weights: Dictionary with TTNN weight tensors
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        self.weights = weights
    
    def embed_actions(self, noisy_actions: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Embed noisy actions using ttnn.linear.
        
        Args:
            noisy_actions: TTNN tensor (batch_size, action_horizon, action_dim)
        
        Returns:
            TTNN tensor (batch_size, action_horizon, expert_width)
        """
        return ttnn.linear(
            noisy_actions,
            self.weights["action_in_proj.weight"],
            bias=self.weights["action_in_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    
    def embed_state(self, state: "ttnn.Tensor") -> Optional["ttnn.Tensor"]:
        """
        Embed robot state (PI0 only).
        
        Args:
            state: TTNN tensor (batch_size, state_dim)
        
        Returns:
            TTNN tensor (batch_size, 1, expert_width) or None for PI05
        """
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
    
    def embed_timestep(self, timestep: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Create timestep embedding.
        
        Args:
            timestep: TTNN tensor (batch_size,)
        
        Returns:
            TTNN tensor (batch_size, expert_width)
        """
        return create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
        )
    
    def fuse_action_time(
        self,
        action_emb: "ttnn.Tensor",
        time_emb: "ttnn.Tensor",
    ) -> Tuple["ttnn.Tensor", Optional["ttnn.Tensor"]]:
        """
        Fuse action and time embeddings using TTNN operations.
        
        Args:
            action_emb: TTNN tensor (batch_size, action_horizon, expert_width)
            time_emb: TTNN tensor (batch_size, expert_width)
        
        Returns:
            Tuple of (fused_emb, adarms_cond)
        """
        if self.config.pi05:
            return action_emb, time_emb
        
        # Get shapes
        batch_size = action_emb.shape[0]
        action_horizon = action_emb.shape[1]
        
        # Expand time embedding to match action shape
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
    
    def project_output(self, expert_output: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Project expert output back to action dimension.
        
        Args:
            expert_output: TTNN tensor (batch_size, action_horizon, expert_width)
        
        Returns:
            TTNN tensor (batch_size, action_horizon, action_dim)
        """
        return ttnn.linear(
            expert_output,
            self.weights["action_out_proj.weight"],
            bias=self.weights["action_out_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


def convert_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: "ttnn.Device",
    dtype: "ttnn.DataType" = None,
) -> Dict[str, "ttnn.Tensor"]:
    """
    Convert PyTorch suffix weights to TTNN format.
    
    Args:
        torch_weights: Dictionary of PyTorch weight tensors
        device: TTNN device
        dtype: TTNN data type (default: bfloat8_b for weights, bfloat16 for bias)
    
    Returns:
        Dictionary of TTNN weight tensors
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype
    
    ttnn_weights = {}
    
    for key, value in torch_weights.items():
        if "bias" in key:
            # Bias: expand to [1, out_features]
            ttnn_weights[key] = ttnn.from_torch(
                value.unsqueeze(0),
                dtype=bias_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # Weight: transpose for TTNN [in, out] format
            ttnn_weights[key] = ttnn.from_torch(
                value.T.contiguous(),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    
    return ttnn_weights


# Default to PyTorch implementation
SuffixEmbedding = SuffixEmbeddingTorch

