# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Embedding modules - Token and position embeddings"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


@dataclass
class EmbeddingConfig:
    """Configuration for Embedding module"""

    vocab_size: int
    embedding_dim: int
    padding_idx: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    scale_embeddings: bool = False


class Embedding(torch.nn.Module):
    """
    Token embedding module.

    Converts token indices to dense vector representations.
    Supports optional embedding scaling as used in some models.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device

        # Embedding weight matrix
        self.weight = None

        # Scaling factor for embeddings
        self.scale = config.embedding_dim**0.5 if config.scale_embeddings else 1.0

    def setup_weight(self, weight: ttnn.Tensor):
        """Set pre-loaded embedding weight"""
        self.weight = weight

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            input_ids: Token indices of shape [batch_size, seq_len]

        Returns:
            embeddings: Token embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        # Embedding lookup
        embeddings = ttnn.embedding(input_ids, self.weight)

        # Apply scaling if configured
        if self.config.scale_embeddings:
            embeddings = embeddings * self.scale

        return embeddings


class PositionEmbedding(torch.nn.Module):
    """
    Learned position embeddings.

    Used in models that learn position representations rather than
    using fixed sinusoidal embeddings or rotary embeddings.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        embedding_dim: int,
        device: ttnn.Device,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim
        self.device = device

        # Position embedding weight matrix
        self.weight = None

    def setup_weight(self, weight: ttnn.Tensor):
        """Set pre-loaded position embedding weight"""
        self.weight = weight

    def forward(
        self,
        position_ids: Optional[ttnn.Tensor] = None,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> ttnn.Tensor:
        """
        Get position embeddings.

        Args:
            position_ids: Optional position indices
            seq_len: Sequence length (used if position_ids not provided)
            batch_size: Batch size (used if position_ids not provided)

        Returns:
            position_embeddings: Position embeddings
        """
        if position_ids is None:
            # Create default position IDs [0, 1, 2, ..., seq_len-1]
            position_ids = ttnn.arange(seq_len, device=self.device)
            position_ids = ttnn.broadcast_to(position_ids, [batch_size, seq_len])

        # Position embedding lookup
        position_embeddings = ttnn.embedding(position_ids, self.weight)

        return position_embeddings


class VisionEmbedding(torch.nn.Module):
    """
    Vision embedding module for multimodal models.

    Converts image patches to embeddings, commonly used in vision transformers.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        embedding_dim: int,
        add_class_token: bool = True,
        add_position_embeddings: bool = True,
        device: ttnn.Device = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.add_class_token = add_class_token
        self.add_position_embeddings = add_position_embeddings
        self.device = device

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch projection weight (conv2d)
        self.patch_embed_weight = None
        self.patch_embed_bias = None

        # Class token
        self.class_token = None

        # Position embeddings
        self.position_embeddings = None

    def setup_weights(
        self,
        patch_embed_weight: ttnn.Tensor,
        patch_embed_bias: Optional[ttnn.Tensor] = None,
        class_token: Optional[ttnn.Tensor] = None,
        position_embeddings: Optional[ttnn.Tensor] = None,
    ):
        """Set pre-loaded weights for vision embedding"""
        self.patch_embed_weight = patch_embed_weight
        self.patch_embed_bias = patch_embed_bias
        self.class_token = class_token
        self.position_embeddings = position_embeddings

    def forward(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            pixel_values: Images of shape [batch_size, num_channels, height, width]

        Returns:
            embeddings: Patch embeddings of shape [batch_size, num_patches + 1, embedding_dim]
                       (+1 is for class token if enabled)
        """
        batch_size = pixel_values.shape[0]

        # Extract patches using convolution
        # Using conv2d with kernel_size=patch_size and stride=patch_size
        patch_embeddings = ttnn.conv2d(
            pixel_values,
            self.patch_embed_weight,
            bias=self.patch_embed_bias,
            stride=self.patch_size,
            padding=0,
        )

        # Flatten patches: [batch_size, embedding_dim, h, w] -> [batch_size, num_patches, embedding_dim]
        patch_embeddings = ttnn.reshape(patch_embeddings, [batch_size, self.embedding_dim, -1])
        patch_embeddings = ttnn.transpose(patch_embeddings, 1, 2)

        # Add class token if configured
        if self.add_class_token and self.class_token is not None:
            # Expand class token for batch
            class_tokens = ttnn.broadcast_to(self.class_token, [batch_size, 1, self.embedding_dim])
            # Concatenate class token with patch embeddings
            patch_embeddings = ttnn.concat([class_tokens, patch_embeddings], dim=1)

        # Add position embeddings if configured
        if self.add_position_embeddings and self.position_embeddings is not None:
            patch_embeddings = patch_embeddings + self.position_embeddings

        return patch_embeddings
