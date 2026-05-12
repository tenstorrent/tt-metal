# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

        self.prefix_att_masks = ttnn.zeros(
            (1, 544),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

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

        image_embs: List[ttnn.Tensor] = []
        expanded_masks: List[ttnn.Tensor] = []

        # OPTIMIZATION: when multiple images share the same shape, run SigLIP
        # in a single bs=N pass instead of N sequential bs=1 calls. Halves
        # the kernel dispatch count for the vision tower.
        same_shape = (
            len(images) > 1
            and isinstance(images[0], ttnn.Tensor)
            and all(isinstance(im, ttnn.Tensor) and tuple(im.shape) == tuple(images[0].shape) for im in images)
        )
        if same_shape:
            # All inputs must be the same layout for concat
            imgs_tiled = [
                im if im.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(im, ttnn.TILE_LAYOUT) for im in images
            ]
            stacked = ttnn.concat(imgs_tiled, dim=0)  # (N, C, H, W)
            all_embs = self.embed_image_fn(stacked)  # (N, num_tokens, vlm_hidden)
            ttnn.deallocate(stacked)
            num_tokens = all_embs.shape[1]
            hidden = all_embs.shape[2]
            for i in range(len(images)):
                img_emb = ttnn.slice(all_embs, [i, 0, 0], [i + 1, num_tokens, hidden])
                if img_emb.layout != ttnn.TILE_LAYOUT:
                    img_emb = ttnn.to_layout(img_emb, ttnn.TILE_LAYOUT)
                image_embs.append(img_emb)
            ttnn.deallocate(all_embs)
        else:
            for img in images:
                emb = self.embed_image_fn(img)
                if emb.layout != ttnn.TILE_LAYOUT:
                    emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
                image_embs.append(emb)

        # Build expanded masks (mask handling unchanged — masks may be per-image)
        for img_emb, mask in zip(image_embs, img_masks):
            shape = img_emb.shape
            batch_size, num_tokens = shape[0], shape[1]

            if isinstance(mask, torch.Tensor):
                mask_ttnn = ttnn.from_torch(
                    mask.float(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                mask_ttnn = ttnn.reshape(mask_ttnn, (batch_size, 1))
                mask_ttnn = ttnn.to_layout(mask_ttnn, ttnn.TILE_LAYOUT)
                expanded_mask = ttnn.repeat(mask_ttnn, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
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

        # ttnn.embedding returns ROW_MAJOR; convert to TILE so downstream
        # concat with TILE image embeddings works for any token length.
        if lang_emb.layout != ttnn.TILE_LAYOUT:
            lang_emb = ttnn.to_layout(lang_emb, ttnn.TILE_LAYOUT)

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

        # Defensive: TTNN concat requires all inputs in the same layout.
        embs = [e if e.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(e, ttnn.TILE_LAYOUT) for e in embs]
        pad_masks = [m if m.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(m, ttnn.TILE_LAYOUT) for m in pad_masks]
        prefix_embs = ttnn.concat(embs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        prefix_pad_masks = ttnn.concat(pad_masks, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Create attention mask (all zeros for bidirectional prefix attention)
        total_tokens = sum(num_tokens_list)
        batch_size = prefix_embs.shape[0]

        # Create zeros mask directly on device (no host transfer needed)
        prefix_att_masks = self.prefix_att_masks

        return prefix_embs, prefix_pad_masks, prefix_att_masks


# Default export
PrefixEmbedding = PrefixEmbeddingTTNN
