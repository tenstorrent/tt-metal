# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Embedding module - TTNN Implementation

This module handles embedding of state, noisy actions, and timestep to create
the suffix part of the sequence for expert transformer processing.

Components:
    - action_in_proj: Projects actions from action_dim to expert width
    - action_out_proj: Projects expert output back to action_dim
    - state_proj: Projects state from state_dim to expert width (PI0 only)
    - action_time_mlp: Fuses action and time embeddings (PI0 only)

Optimizations:
    - Pre-computed attention mask pattern (saves 10 transfers per inference)
"""

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SuffixConfig
from .ttnn_common import create_sinusoidal_pos_embedding_ttnn, tensor_1d_to_2d_ttnn


class SuffixEmbeddingTTNN:
    """
    TTNN implementation of suffix embedding.

    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """

    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, ttnn.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize suffix embedding with TTNN weights.

        Args:
            config: Suffix configuration
            weights: Dictionary with TTNN weight tensors
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.weights = weights

        # Query device grid to use all available cores (P150: up to 13x10)
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # OPTIMIZATION: Pre-compute attention mask pattern (saves 10 transfers per inference!)
        # Attention mask is constant for a given config: [1, 1, 0, ..., 0] for PI0
        # or [1, 0, ..., 0] for PI05
        att_mask_pattern = []
        if not config.pi05:
            att_mask_pattern.append(1)  # State token
        att_mask_pattern.append(1)  # First action token
        att_mask_pattern.extend([0] * (config.action_horizon - 1))  # Remaining action tokens

        # Store as TTNN tensor with batch_size=1, will repeat at runtime
        # Pad to tile-aligned size (suffix_len typically ~51 -> 64)
        suffix_len = len(att_mask_pattern)
        pad_len = ((suffix_len + 31) // 32) * 32

        # Create attention mask using ttnn (avoid torch.tensor)
        # Use ttnn.zeros and fill with ones at specific positions
        att_mask_ttnn = ttnn.zeros((1, pad_len), device=device, dtype=ttnn.bfloat16)
        att_mask_ttnn = ttnn.to_layout(att_mask_ttnn, ttnn.TILE_LAYOUT)

        # The pattern is: [1, 1, 0, ...] for PI0 or [1, 0, ...] for PI05
        # Create a mask with ones in the right positions
        num_ones = len([x for x in att_mask_pattern if x == 1])
        if num_ones > 0:
            ones_tensor = ttnn.ones((1, num_ones), device=device, dtype=ttnn.bfloat16)
            ones_tensor = ttnn.to_layout(ones_tensor, ttnn.TILE_LAYOUT)
            # Pad the ones tensor to full length
            ones_padded = ttnn.pad(ones_tensor, [(0, 0), (0, pad_len - num_ones)], value=0.0)
            ttnn.deallocate(ones_tensor)
            ttnn.deallocate(att_mask_ttnn)
            att_mask_ttnn = ones_padded

        self._att_mask_pattern = att_mask_ttnn
        self._att_mask_suffix_len = suffix_len

        self.indices = ttnn.arange(0, 512, 1, device=device, dtype=ttnn.float32)

    def embed_actions(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
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
            core_grid=self.core_grid,
        )

    def embed_state(self, state: ttnn.Tensor) -> Optional[ttnn.Tensor]:
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
            core_grid=self.core_grid,
        )

        # Add sequence dimension
        shape = state_emb.shape
        return ttnn.reshape(state_emb, (shape[0], 1, shape[-1]))

    def embed_timestep(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
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
            indices=self.indices,
        )

    def fuse_action_time(
        self,
        action_emb: ttnn.Tensor,
        time_emb: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
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
            core_grid=self.core_grid,
        )
        x = ttnn.silu(x)
        x = ttnn.linear(
            x,
            self.weights["action_time_mlp_out.weight"],
            bias=self.weights["action_time_mlp_out.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )

        return x, None

    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Create suffix embeddings using TTNN operations.

        Args:
            state: TTNN tensor (batch_size, state_dim) or None
            noisy_actions: TTNN tensor (batch_size, action_horizon, action_dim)
            timestep: TTNN tensor (batch_size,)

        Returns:
            Tuple of (suffix_embs, pad_masks, att_masks, adarms_cond)
            - suffix_embs: (batch_size, suffix_len, expert_width)
            - pad_masks: (batch_size, suffix_len) - all ones
            - att_masks: (batch_size, suffix_len) - causal attention masks
            - adarms_cond: Optional conditioning for adaptive RMSNorm
        """
        batch_size = noisy_actions.shape[0]
        embs = []

        # Embed state (PI0 only, not PI05)
        if not self.config.pi05:
            state_emb = self.embed_state(state)
            if state_emb is not None:
                embs.append(state_emb)

        # Embed timestep
        time_emb = self.embed_timestep(timestep)

        # Embed actions
        action_emb = self.embed_actions(noisy_actions)

        # Fuse action and time
        action_time_emb, adarms_cond = self.fuse_action_time(action_emb, time_emb)

        # Add action-time embeddings
        embs.append(action_time_emb)

        # Concatenate embeddings along sequence dimension
        if len(embs) > 1:
            suffix_embs = ttnn.concat(embs, dim=1)
        else:
            suffix_embs = embs[0]

        # Create masks on device
        suffix_len = suffix_embs.shape[1]

        # Padding mask: all ones (no padding)
        suffix_pad_masks = None

        # OPTIMIZATION: Use pre-computed attention mask pattern (no transfer per step!)
        # Pattern is pre-computed in __init__, just repeat for batch_size
        if batch_size == 1:
            # Most common case: batch_size=1, just slice to suffix_len
            suffix_att_masks = ttnn.slice(
                self._att_mask_pattern,
                [0, 0],
                [1, suffix_len],
            )
        else:
            # Repeat pattern for batch_size > 1
            att_mask_sliced = ttnn.slice(
                self._att_mask_pattern,
                [0, 0],
                [1, suffix_len],
            )
            suffix_att_masks = ttnn.repeat(
                att_mask_sliced,
                (batch_size, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
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
            core_grid=self.core_grid,
        )


def convert_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
) -> Dict[str, ttnn.Tensor]:
    """
    Convert PyTorch suffix weights to TTNN format.

    Args:
        torch_weights: Dictionary of PyTorch weight tensors
        device: TTNN device
        dtype: TTNN data type (default: bfloat8_b for weights, bfloat16 for bias)

    Returns:
        Dictionary of TTNN weight tensors
    """
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype

    ttnn_weights = {}

    for key, value in torch_weights.items():
        if "bias" in key:
            # Bias: expand to [1, out_features] using TTNN (no torch.unsqueeze)
            ttnn_weights[key] = tensor_1d_to_2d_ttnn(value, device, dtype=bias_dtype)
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


# Default export
SuffixEmbedding = SuffixEmbeddingTTNN

# ============================================================================
# pi0.5-specific additions (subclasses, overrides)
# ============================================================================

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.tt.ttnn_common import (
    create_sinusoidal_pos_embedding_ttnn,
    tensor_1d_to_2d_ttnn,
)


class Pi0_5SuffixEmbeddingTTNN(SuffixEmbeddingTTNN):
    """
    PI0.5 TTNN suffix.

    Expected weight keys (TTNN tensors):
      - action_in_proj.{weight,bias}
      - action_out_proj.{weight,bias}
      - time_mlp_in.{weight,bias}
      - time_mlp_out.{weight,bias}
    """

    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, "ttnn.Tensor"],
        device: "ttnn.Device",
    ):
        assert config.pi05, "Pi0_5SuffixEmbeddingTTNN requires config.pi05=True"
        self.config = config
        self.device = device
        self.weights = weights

        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Pre-compute attention mask pattern: pi0.5 has only action tokens,
        # so the pattern is [1, 0, 0, ..., 0] of length action_horizon.
        suffix_len = config.action_horizon
        pad_len = ((suffix_len + 31) // 32) * 32
        att = ttnn.zeros((1, pad_len), device=device, dtype=ttnn.bfloat16)
        att = ttnn.to_layout(att, ttnn.TILE_LAYOUT)
        ones = ttnn.ones((1, 1), device=device, dtype=ttnn.bfloat16)
        ones = ttnn.to_layout(ones, ttnn.TILE_LAYOUT)
        ones_padded = ttnn.pad(ones, [(0, 0), (0, pad_len - 1)], value=0.0)
        ttnn.deallocate(ones)
        ttnn.deallocate(att)
        self._att_mask_pattern = ones_padded
        self._att_mask_suffix_len = suffix_len

        self.indices = ttnn.arange(0, 512, 1, device=device, dtype=ttnn.float32)

    def embed_adarms_cond(self, timestep: "ttnn.Tensor") -> "ttnn.Tensor":
        """sincos(t) -> Linear -> silu -> Linear -> silu -> adarms_cond.

        The trailing silu matches openpi/lerobot pi05 reference; without it
        the scale/shift/gate modulations downstream flip sign of model outputs.
        """
        sincos = create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
            indices=self.indices,
        )
        x = ttnn.linear(
            sincos,
            self.weights["time_mlp_in.weight"],
            bias=self.weights.get("time_mlp_in.bias"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )
        x = ttnn.silu(x)
        x = ttnn.linear(
            x,
            self.weights["time_mlp_out.weight"],
            bias=self.weights.get("time_mlp_out.bias"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )
        return ttnn.silu(x)

    def embed_suffix(
        self,
        state: Optional["ttnn.Tensor"],
        noisy_actions: "ttnn.Tensor",
        timestep: "ttnn.Tensor",
    ) -> Tuple["ttnn.Tensor", Optional["ttnn.Tensor"], "ttnn.Tensor", "ttnn.Tensor"]:
        batch_size = noisy_actions.shape[0]

        adarms_cond = self.embed_adarms_cond(timestep)
        suffix_embs = self.embed_actions(noisy_actions)

        suffix_len = suffix_embs.shape[1]
        if batch_size == 1:
            suffix_att_masks = ttnn.slice(self._att_mask_pattern, [0, 0], [1, suffix_len])
        else:
            att_sliced = ttnn.slice(self._att_mask_pattern, [0, 0], [1, suffix_len])
            suffix_att_masks = ttnn.repeat(att_sliced, (batch_size, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return suffix_embs, None, suffix_att_masks, adarms_cond


def convert_pi0_5_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: "ttnn.Device",
    dtype: Optional["ttnn.DataType"] = None,
) -> Dict[str, "ttnn.Tensor"]:
    """
    Convert PI0.5 suffix weights from PyTorch to TTNN.

    Accepts only keys that pi0.5 actually uses (action_in_proj, action_out_proj,
    time_mlp_in, time_mlp_out). Other keys in the checkpoint are ignored.
    """
    keep_prefixes = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype

    out: Dict[str, "ttnn.Tensor"] = {}
    for key, value in torch_weights.items():
        if not any(key.startswith(p) for p in keep_prefixes):
            continue
        if "bias" in key:
            out[key] = tensor_1d_to_2d_ttnn(value, device, dtype=bias_dtype)
        else:
            out[key] = ttnn.from_torch(
                value.T.contiguous(),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    return out
