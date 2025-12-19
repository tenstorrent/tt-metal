# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - PyTorch Reference Implementation.

This module handles embedding of images and language tokens to create
the prefix part of the sequence for VLM transformer processing.

Components:
    - Vision tower: SigLIP for image encoding
    - Multi-modal projector: Projects vision features to language dimension
    - Language embedding: Token embeddings for language
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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
    
    Embeds images + language tokens for the VLM backbone.
    """
    
    def __init__(
        self,
        config: PrefixConfig,
        vision_tower,  # SigLIPVisionTower instance
        projector,     # MultiModalProjector instance  
        embed_tokens: torch.Tensor,
    ):
        """
        Initialize prefix embedding.
        
        Args:
            config: Prefix configuration
            vision_tower: SigLIP vision tower for image encoding
            projector: Multi-modal projector for dimension matching
            embed_tokens: Token embedding weights
        """
        self.config = config
        self.vision_tower = vision_tower
        self.projector = projector
        self.embed_tokens = embed_tokens
    
    def embed_images(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed images through vision tower and projector.
        
        Args:
            images: List of image tensors per sample
            masks: List of boolean masks indicating valid images
        
        Returns:
            Tuple of (image_embeddings, image_masks)
        """
        batch_size = len(images)
        all_image_embs = []
        all_masks = []
        
        for batch_idx in range(batch_size):
            batch_images = images[batch_idx]
            batch_masks = masks[batch_idx]
            
            if batch_images.numel() > 0:
                # Process through vision tower
                vision_features = self.vision_tower.forward(batch_images)
                # Project to language dimension
                projected = self.projector.forward(vision_features)
                
                # Apply masks
                projected = projected * batch_masks.view(-1, 1, 1)
                
                all_image_embs.append(projected.reshape(-1, projected.shape[-1]))
                all_masks.append(batch_masks.view(-1, 1).expand(-1, projected.shape[1]).reshape(-1))
        
        if all_image_embs:
            image_embs = torch.stack([safe_cat([e], dim=0) for e in all_image_embs])
            image_masks = torch.stack([m for m in all_masks])
        else:
            image_embs = torch.zeros(batch_size, 0, self.config.vlm_hidden_size)
            image_masks = torch.zeros(batch_size, 0, dtype=torch.bool)
        
        return image_embs, image_masks
    
    def embed_language(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed language tokens.
        
        Args:
            tokens: Token IDs (batch_size, seq_len)
            masks: Boolean masks indicating valid tokens
        
        Returns:
            Tuple of (language_embeddings, language_masks)
        """
        # Get token embeddings
        embed_tokens = self.embed_tokens.to(tokens.device)
        embeddings = F.embedding(tokens, embed_tokens)
        
        # Scale by sqrt(hidden_dim) as in Gemma
        hidden_dim = embeddings.shape[-1]
        embeddings = embeddings * math.sqrt(hidden_dim)
        
        return embeddings, masks
    
    def embed_prefix(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create prefix embeddings from images and language.
        
        Args:
            images: List of image tensors per sample
            image_masks: List of boolean masks for valid images
            lang_tokens: Language token IDs
            lang_masks: Language token masks
        
        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed images
        image_embs, img_masks = self.embed_images(images, image_masks)
        if image_embs.numel() > 0:
            embs.append(image_embs)
            pad_masks.append(img_masks)
            # Images use bidirectional attention
            att_masks.append(torch.ones_like(img_masks, dtype=torch.bool))
        
        # Embed language
        lang_embs, lang_pad_masks = self.embed_language(lang_tokens, lang_masks)
        embs.append(lang_embs)
        pad_masks.append(lang_pad_masks)
        # Language uses bidirectional attention
        att_masks.append(torch.ones_like(lang_pad_masks, dtype=torch.bool))
        
        # Concatenate
        prefix_embs = safe_cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.cat(att_masks, dim=1)
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks

