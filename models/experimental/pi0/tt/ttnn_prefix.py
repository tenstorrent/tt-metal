# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - TTNN Implementation

This module handles embedding of images and language tokens to create the
prefix part of the sequence for transformer processing.

Components:
    - Image embedding via SigLIP vision tower
    - Language token embedding via Gemma embeddings
    - Concatenation of image and language embeddings with proper masking

Attention Pattern:
    - All prefix tokens can attend to each other (bidirectional)
    - Suffix tokens can attend to prefix (cross-attention)
"""

import math
from typing import List, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import PrefixConfig


class PrefixEmbeddingTTNN:
    """
    TTNN implementation of prefix embedding.

    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """

    def __init__(
        self,
        config: PrefixConfig,
        device: ttnn.Device,
        embed_image_fn=None,
        embed_language_fn=None,
    ):
        """
        Initialize prefix embedding with TTNN.

        Args:
            config: Prefix configuration
            device: TTNN device
            embed_image_fn: Function to embed images
            embed_language_fn: Function to embed language tokens
        """
        self.config = config
        self.device = device
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn

    def embed_images(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
    ) -> Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
        """
        Embed multiple images using TTNN.

        Args:
            images: List of PyTorch image tensors (vision tower handles TTNN conversion)
            img_masks: List of PyTorch mask tensors

        Returns:
            Tuple of (image_embeddings, expanded_masks) as TTNN tensors
        """
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")

        image_embs = []
        expanded_masks = []

        for img, mask in zip(images, img_masks):
            # embed_image_fn handles PyTorch->TTNN conversion internally
            img_emb = self.embed_image_fn(img)
            image_embs.append(img_emb)

            # Expand mask - convert from PyTorch if needed
            shape = img_emb.shape
            batch_size, num_tokens = shape[0], shape[1]

            if isinstance(mask, torch.Tensor):
                # Convert PyTorch mask to TTNN, reshape on device (no torch.unsqueeze)
                mask_ttnn = ttnn.from_torch(
                    mask.float(),  # (batch_size,)
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Reshape to 2D and convert to TILE on device
                mask_ttnn = ttnn.reshape(mask_ttnn, (batch_size, 1))
                mask_ttnn = ttnn.to_layout(mask_ttnn, ttnn.TILE_LAYOUT)
                # Expand on device using ttnn.repeat (no round-trip!)
                expanded_mask = ttnn.repeat(mask_ttnn, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                # Already TTNN - expand directly on device (no round-trip!)
                mask_reshaped = ttnn.reshape(mask, (batch_size, 1))
                expanded_mask = ttnn.repeat(mask_reshaped, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)

            expanded_masks.append(expanded_mask)

        return image_embs, expanded_masks

    def embed_language(
        self,
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Embed language tokens using TTNN.

        Args:
            lang_tokens: TTNN tensor of token IDs
            lang_masks: TTNN tensor of validity masks

        Returns:
            TTNN tensor of scaled embeddings
        """
        if self.embed_language_fn is None:
            raise RuntimeError("embed_language_fn not set")

        lang_emb = self.embed_language_fn(lang_tokens)

        # Scale by sqrt(hidden_dim) - use scalar multiply
        hidden_dim = lang_emb.shape[-1]
        scale = math.sqrt(hidden_dim)

        return ttnn.mul(lang_emb, scale)

    def embed_prefix(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Main embedding function for prefix (TTNN version).

        Args:
            images: List of TTNN image tensors
            img_masks: List of TTNN mask tensors
            lang_tokens: TTNN tensor of language tokens
            lang_masks: TTNN tensor of language masks

        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)
        """
        embs = []
        pad_masks = []
        num_tokens_list = []

        # Process images
        if images and self.embed_image_fn is not None:
            image_embs, img_pad_masks = self.embed_images(images, img_masks)
            for img_emb, img_pad_mask in zip(image_embs, img_pad_masks):
                embs.append(img_emb)
                pad_masks.append(img_pad_mask)
                num_tokens_list.append(img_emb.shape[1])

        # Process language
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens, lang_masks)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            num_tokens_list.append(lang_emb.shape[1])

        # Concatenate using TTNN
        prefix_embs = ttnn.concat(embs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        prefix_pad_masks = ttnn.concat(pad_masks, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Create attention mask (all zeros for bidirectional prefix attention)
        total_tokens = sum(num_tokens_list)
        batch_size = prefix_embs.shape[0]

        # Create zeros mask directly on device (no host transfer needed)
        prefix_att_masks = ttnn.zeros(
            (batch_size, total_tokens),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.ReadDeviceProfiler(
            self.device
        )  # Clear device profiler buffer, this helps resolve a issue when building profiler perf sheets

        return prefix_embs, prefix_pad_masks, prefix_att_masks


# Default export
PrefixEmbedding = PrefixEmbeddingTTNN
