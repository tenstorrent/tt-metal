# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower - PyTorch Reference Implementation.

This module implements the SigLIP vision encoder that processes images
into feature embeddings for the VLM backbone.

SigLIP Architecture:
    - Patch embedding (conv2d to extract patches)
    - Positional embedding (learned)
    - Transformer encoder blocks
    - Multi-modal projector (linear to match language model dimension)
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import SigLIPConfig


# ============================================================================
# Patch Embedding
# ============================================================================


class PatchEmbedding:
    """
    Convert image patches to embeddings (PyTorch).

    Uses Conv2d with kernel_size = patch_size to extract non-overlapping patches.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize patch embedding.

        Args:
            config: SigLIP configuration
            weights: Dictionary with:
                - patch_embedding.weight: (hidden_size, channels, patch_size, patch_size)
                - patch_embedding.bias: (hidden_size,)
        """
        self.config = config
        # Handle both formats: vision_model.embeddings.patch_embedding (checkpoint) and patch_embedding (legacy)
        self.conv_weight = weights.get("patch_embedding.weight") or weights.get(
            "vision_model.embeddings.patch_embedding.weight"
        )
        self.conv_bias = weights.get("patch_embedding.bias") or weights.get(
            "vision_model.embeddings.patch_embedding.bias"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from images.

        Args:
            pixel_values: (batch_size, channels, height, width)

        Returns:
            (batch_size, num_patches, hidden_size)
        """
        # Apply convolution (ensure dtype compatibility)
        conv_weight = self.conv_weight.to(pixel_values.dtype)
        conv_bias = self.conv_bias.to(pixel_values.dtype) if self.conv_bias is not None else None
        x = F.conv2d(
            pixel_values,
            conv_weight,
            conv_bias,
            stride=self.config.patch_size,
        )

        # Reshape: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)

        return x


# ============================================================================
# SigLIP Attention
# ============================================================================


class SigLIPAttention:
    """
    SigLIP self-attention (PyTorch).

    Standard multi-head attention without rotary embeddings.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize attention.

        Args:
            config: SigLIP configuration
            weights: Attention weights
        """
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = weights["self_attn.q_proj.weight"]
        self.k_proj = weights["self_attn.k_proj.weight"]
        self.v_proj = weights["self_attn.v_proj.weight"]
        self.out_proj = weights["self_attn.out_proj.weight"]

        self.q_bias = weights.get("self_attn.q_proj.bias")
        self.k_bias = weights.get("self_attn.k_proj.bias")
        self.v_bias = weights.get("self_attn.v_proj.bias")
        self.out_bias = weights.get("self_attn.out_proj.bias")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections (ensure dtype compatibility)
        q_proj = self.q_proj.to(hidden_states.dtype)
        k_proj = self.k_proj.to(hidden_states.dtype)
        v_proj = self.v_proj.to(hidden_states.dtype)
        q_bias = self.q_bias.to(hidden_states.dtype) if self.q_bias is not None else None
        k_bias = self.k_bias.to(hidden_states.dtype) if self.k_bias is not None else None
        v_bias = self.v_bias.to(hidden_states.dtype) if self.v_bias is not None else None

        q = F.linear(hidden_states, q_proj, q_bias)
        k = F.linear(hidden_states, k_proj, k_bias)
        v = F.linear(hidden_states, v_proj, v_bias)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project (ensure dtype compatibility)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out_proj = self.out_proj.to(hidden_states.dtype)
        out_bias = self.out_bias.to(hidden_states.dtype) if self.out_bias is not None else None
        return F.linear(attn_output, out_proj, out_bias)


# ============================================================================
# SigLIP MLP
# ============================================================================


class SigLIPMLP:
    """
    SigLIP MLP with GELU activation (PyTorch).
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize MLP.

        Args:
            config: SigLIP configuration
            weights: MLP weights
        """
        self.fc1_weight = weights["mlp.fc1.weight"]
        self.fc1_bias = weights.get("mlp.fc1.bias")
        self.fc2_weight = weights["mlp.fc2.weight"]
        self.fc2_bias = weights.get("mlp.fc2.bias")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        # Ensure dtype compatibility
        fc1_weight = self.fc1_weight.to(hidden_states.dtype)
        fc1_bias = self.fc1_bias.to(hidden_states.dtype) if self.fc1_bias is not None else None
        fc2_weight = self.fc2_weight.to(hidden_states.dtype)
        fc2_bias = self.fc2_bias.to(hidden_states.dtype) if self.fc2_bias is not None else None

        x = F.linear(hidden_states, fc1_weight, fc1_bias)
        x = F.gelu(x, approximate="tanh")
        return F.linear(x, fc2_weight, fc2_bias)


# ============================================================================
# SigLIP Block
# ============================================================================


class SigLIPBlock:
    """
    Complete SigLIP transformer block (PyTorch).

    Architecture: Pre-LN
        x -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize block.

        Args:
            config: SigLIP configuration
            weights: Block weights
        """
        self.config = config

        self.ln1_weight = weights["layer_norm1.weight"]
        self.ln1_bias = weights.get("layer_norm1.bias")
        self.ln2_weight = weights["layer_norm2.weight"]
        self.ln2_bias = weights.get("layer_norm2.bias")

        self.attention = SigLIPAttention(config, weights)
        self.mlp = SigLIPMLP(config, weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        # Pre-attention norm (ensure dtype compatibility)
        ln1_weight = self.ln1_weight.to(hidden_states.dtype) if self.ln1_weight is not None else None
        ln1_bias = self.ln1_bias.to(hidden_states.dtype) if self.ln1_bias is not None else None
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            ln1_weight,
            ln1_bias,
            self.config.layer_norm_eps,
        )

        # Attention with residual
        hidden_states = hidden_states + self.attention.forward(normed)

        # Pre-MLP norm (ensure dtype compatibility)
        ln2_weight = self.ln2_weight.to(hidden_states.dtype) if self.ln2_weight is not None else None
        ln2_bias = self.ln2_bias.to(hidden_states.dtype) if self.ln2_bias is not None else None
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            ln2_weight,
            ln2_bias,
            self.config.layer_norm_eps,
        )

        # MLP with residual
        hidden_states = hidden_states + self.mlp.forward(normed)

        return hidden_states


# ============================================================================
# Full Vision Tower
# ============================================================================


class SigLIPVisionTower:
    """
    Complete SigLIP vision tower (PyTorch).

    Processes images into embeddings for the VLM backbone.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize vision tower.

        Args:
            config: SigLIP configuration
            weights: All vision tower weights
        """
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config, weights)

        # Position embedding (handle both formats)
        self.position_embedding = weights.get("position_embedding.weight") or weights.get(
            "vision_model.embeddings.position_embedding.weight"
        )

        # Encoder blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(SigLIPBlock(config, block_weights))

        # Final layer norm
        self.post_layernorm_weight = weights.get("post_layernorm.weight") or weights.get(
            "vision_model.post_layernorm.weight"
        )
        self.post_layernorm_bias = weights.get("post_layernorm.bias") or weights.get("vision_model.post_layernorm.bias")

    def _get_layer_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific layer."""
        # Handle both formats: vision_model.encoder.layers.X (checkpoint) and encoder.layers.X (legacy)
        prefixes = [f"vision_model.encoder.layers.{layer_idx}.", f"encoder.layers.{layer_idx}."]
        layer_weights = {}
        for prefix in prefixes:
            for key, value in weights.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    layer_weights[new_key] = value
        return layer_weights

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process images to embeddings.

        Args:
            pixel_values: (batch_size, channels, height, width)

        Returns:
            (batch_size, num_patches, hidden_size)
        """
        # Patch embedding
        hidden_states = self.patch_embed.forward(pixel_values)

        # Add position embeddings (with interpolation if needed)
        if self.position_embedding is not None:
            num_patches = hidden_states.shape[1]
            num_positions = self.position_embedding.shape[0]

            if num_patches != num_positions:
                # Interpolate position embeddings to match the number of patches
                pos_embed = self.position_embedding.unsqueeze(0).permute(0, 2, 1)  # (1, hidden_size, num_positions)

                # Calculate original grid size (assume square)
                orig_size = int(num_positions**0.5)
                new_size = int(num_patches**0.5)

                pos_embed = pos_embed.reshape(1, self.config.hidden_size, orig_size, orig_size)
                pos_embed = torch.nn.functional.interpolate(
                    pos_embed, size=(new_size, new_size), mode="bicubic", align_corners=False
                )
                pos_embed = pos_embed.reshape(1, self.config.hidden_size, -1).permute(
                    0, 2, 1
                )  # (1, num_patches, hidden_size)
                pos_embed = pos_embed.squeeze(0)  # (num_patches, hidden_size)
            else:
                pos_embed = self.position_embedding

            hidden_states = hidden_states + pos_embed

        # Encoder blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)

        # Final layer norm (ensure dtype compatibility)
        if self.post_layernorm_weight is not None:
            post_ln_weight = self.post_layernorm_weight.to(hidden_states.dtype)
            post_ln_bias = (
                self.post_layernorm_bias.to(hidden_states.dtype) if self.post_layernorm_bias is not None else None
            )
            hidden_states = F.layer_norm(
                hidden_states,
                (self.config.hidden_size,),
                post_ln_weight,
                post_ln_bias,
                self.config.layer_norm_eps,
            )

        return hidden_states


# ============================================================================
# Multi-modal Projector
# ============================================================================


class MultiModalProjector:
    """
    Projects vision features to language model dimension (PyTorch).
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize projector.

        Args:
            weights: Dictionary with linear.weight and optional linear.bias
        """
        self.weight = weights["linear.weight"]
        self.bias = weights.get("linear.bias")

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features.

        Args:
            vision_features: (batch_size, num_patches, vision_hidden_size)

        Returns:
            (batch_size, num_patches, language_hidden_size)
        """
        # Ensure dtype compatibility
        weight = self.weight.to(vision_features.dtype)
        bias = self.bias.to(vision_features.dtype) if self.bias is not None else None
        return F.linear(vision_features, weight, bias)
