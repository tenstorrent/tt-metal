# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - PyTorch Reference Implementation.

This module handles embedding of images and language tokens to create
the prefix part of the sequence for VLM transformer processing.

Components:
    - Vision tower: SigLIP for image encoding (via embed_image_fn callback)
    - Language embedding: Token embeddings (via embed_language_fn callback)
"""

import math
from typing import List, Tuple

import torch

from models.experimental.pi0.common.configs import PrefixConfig


def safe_cat(tensors: list, dim: int = -1) -> torch.Tensor:
    """Safely concatenate tensors with dtype handling."""
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    target_dtype = tensors[0].dtype
    converted = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in tensors]
    return torch.cat(converted, dim=dim)


class PrefixEmbedding:
    """
    PyTorch implementation of prefix embedding.

    Combines image and language embeddings for the VLM backbone.
    Uses callback functions for embedding operations.
    """

    def __init__(
        self,
        config: PrefixConfig,
        embed_image_fn=None,
        embed_language_fn=None,
    ):
        """
        Initialize prefix embedding.

        Args:
            config: Prefix configuration
            embed_image_fn: Function to embed images (from SigLIP)
            embed_language_fn: Function to embed language tokens (from Gemma)
        """
        self.config = config
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn

    def embed_images(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Embed multiple images.

        Args:
            images: List of image tensors (batch_size, channels, height, width)
            img_masks: List of boolean masks (batch_size,) indicating valid images

        Returns:
            Tuple of (image_embeddings, expanded_masks):
                - image_embeddings: List of (batch_size, num_tokens, hidden_dim)
                - expanded_masks: List of (batch_size, num_tokens)
        """
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")

        image_embs = []
        expanded_masks = []

        for img, mask in zip(images, img_masks):
            # Embed image through vision tower
            img_emb = self.embed_image_fn(img)
            image_embs.append(img_emb)

            # Expand mask to match token count
            batch_size, num_tokens = img_emb.shape[:2]
            expanded_mask = mask[:, None].expand(batch_size, num_tokens)
            expanded_masks.append(expanded_mask)

        return image_embs, expanded_masks

    def embed_language(
        self,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed language tokens.

        Args:
            lang_tokens: (batch_size, seq_len) token IDs
            lang_masks: (batch_size, seq_len) boolean validity mask

        Returns:
            (batch_size, seq_len, hidden_dim) scaled embeddings
        """
        if self.embed_language_fn is None:
            raise RuntimeError("embed_language_fn not set")

        lang_emb = self.embed_language_fn(lang_tokens)

        # Scale by sqrt(hidden_dim) as per standard practice
        hidden_dim = lang_emb.shape[-1]
        return lang_emb * math.sqrt(hidden_dim)

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main embedding function for prefix (images + language).

        Args:
            images: List of image tensors
            img_masks: List of image validity masks
            lang_tokens: Language token IDs
            lang_masks: Language token masks

        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks):
                - prefix_embs: (batch_size, prefix_len, hidden_dim)
                - prefix_pad_masks: (batch_size, prefix_len) padding mask
                - prefix_att_masks: (batch_size, prefix_len) attention pattern
                    where 0 = bidirectional attention (prefix behavior)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        if images and self.embed_image_fn is not None:
            image_embs, img_pad_masks = self.embed_images(images, img_masks)
            for img_emb, img_pad_mask in zip(image_embs, img_pad_masks):
                embs.append(img_emb)
                pad_masks.append(img_pad_mask)
                # Images use bidirectional attention (0)
                num_img_tokens = img_emb.shape[1]
                att_masks.extend([0] * num_img_tokens)

        # Process language
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens, lang_masks)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            # Language uses bidirectional attention within prefix (0)
            num_lang_tokens = lang_emb.shape[1]
            att_masks.extend([0] * num_lang_tokens)

        # Concatenate all embeddings
        prefix_embs = safe_cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)

        # Create attention mask tensor
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=device)
        prefix_att_masks = att_masks_tensor.unsqueeze(0).expand(batch_size, -1)

        return prefix_embs, prefix_pad_masks, prefix_att_masks
