# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module for TTNN PI0 implementation.

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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from .ttnn_common import safe_cat_torch


@dataclass
class PrefixConfig:
    """Configuration for prefix embedding."""
    vlm_width: int = 2048
    num_image_tokens: int = 256  # Tokens per image from SigLIP
    max_lang_tokens: int = 512


class PrefixEmbeddingTorch:
    """
    PyTorch implementation of prefix embedding.
    
    Combines image and language embeddings for the VLM backbone.
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
        prefix_embs = safe_cat_torch(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        
        # Create attention mask tensor
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=device)
        prefix_att_masks = att_masks_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks


class PrefixEmbeddingTTNN:
    """
    TTNN implementation of prefix embedding.
    
    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """
    
    def __init__(
        self,
        config: PrefixConfig,
        device: "ttnn.Device",
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
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn
    
    def embed_images(
        self,
        images: List["ttnn.Tensor"],
        img_masks: List["ttnn.Tensor"],
    ) -> Tuple[List["ttnn.Tensor"], List["ttnn.Tensor"]]:
        """
        Embed multiple images using TTNN.
        
        Args:
            images: List of TTNN image tensors
            img_masks: List of TTNN mask tensors
        
        Returns:
            Tuple of (image_embeddings, expanded_masks)
        """
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")
        
        image_embs = []
        expanded_masks = []
        
        for img, mask in zip(images, img_masks):
            img_emb = self.embed_image_fn(img)
            image_embs.append(img_emb)
            
            # Expand mask
            shape = img_emb.shape
            batch_size, num_tokens = shape[0], shape[1]
            mask_reshaped = ttnn.reshape(mask, (batch_size, 1))
            expanded_mask = ttnn.repeat(mask_reshaped, (1, num_tokens))
            expanded_masks.append(expanded_mask)
        
        return image_embs, expanded_masks
    
    def embed_language(
        self,
        lang_tokens: "ttnn.Tensor",
        lang_masks: "ttnn.Tensor",
    ) -> "ttnn.Tensor":
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
        
        # Scale by sqrt(hidden_dim)
        hidden_dim = lang_emb.shape[-1]
        scale = math.sqrt(hidden_dim)
        
        return ttnn.multiply(lang_emb, scale)
    
    def embed_prefix(
        self,
        images: List["ttnn.Tensor"],
        img_masks: List["ttnn.Tensor"],
        lang_tokens: "ttnn.Tensor",
        lang_masks: "ttnn.Tensor",
    ) -> Tuple["ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor"]:
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
        
        # Create zeros mask on host then transfer
        att_masks_torch = torch.zeros(batch_size, total_tokens, dtype=torch.bool)
        prefix_att_masks = ttnn.from_torch(
            att_masks_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks


class MockEmbeddingFunctions:
    """
    Mock embedding functions for testing without full model.
    
    Creates random embeddings with correct shapes.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_image_tokens: int = 256,
    ):
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
    
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Mock image embedding."""
        batch_size = image.shape[0]
        return torch.randn(
            batch_size,
            self.num_image_tokens,
            self.hidden_dim,
            device=image.device,
            dtype=image.dtype,
        )
    
    def embed_language(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mock language embedding."""
        batch_size, seq_len = tokens.shape
        return torch.randn(
            batch_size,
            seq_len,
            self.hidden_dim,
            device=tokens.device,
            dtype=torch.float32,
        )


# Default to PyTorch implementation
PrefixEmbedding = PrefixEmbeddingTorch

