# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - TTNN Implementation.

This module handles embedding of images and language tokens using
TTNN operations for efficient execution on Tenstorrent hardware.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import PrefixConfig


class TtPrefixEmbedding:
    """TTNN implementation of prefix embedding."""
    
    def __init__(
        self,
        config: PrefixConfig,
        vision_tower,  # TtSigLIPVisionTower instance
        projector,     # TtMultiModalProjector instance
        embed_tokens: ttnn.Tensor,
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        self.vision_tower = vision_tower
        self.projector = projector
        self.embed_tokens = embed_tokens
    
    def embed_images(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Embed images through vision tower and projector."""
        batch_size = len(images)
        all_image_embs = []
        all_masks = []
        
        for batch_idx in range(batch_size):
            batch_images = images[batch_idx]
            batch_masks = masks[batch_idx]
            
            if batch_images.numel() > 0:
                # Process through vision tower (returns TTNN)
                vision_features = self.vision_tower.forward(batch_images)
                # Project to language dimension
                projected = self.projector.forward(vision_features)
                
                # Convert masks and apply
                mask_ttnn = ttnn.from_torch(
                    batch_masks.view(-1, 1, 1).float(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                projected = ttnn.multiply(projected, mask_ttnn)
                
                # Convert to torch for reshaping
                projected_torch = ttnn.to_torch(projected)
                reshaped = projected_torch.reshape(-1, projected_torch.shape[-1])
                
                all_image_embs.append(reshaped)
                all_masks.append(batch_masks.view(-1, 1).expand(-1, projected_torch.shape[1]).reshape(-1))
        
        if all_image_embs:
            # Stack and convert back to TTNN
            stacked = torch.stack(all_image_embs)
            image_embs = ttnn.from_torch(
                stacked,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            
            mask_stacked = torch.stack(all_masks)
            image_masks = ttnn.from_torch(
                mask_stacked.float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            image_embs = ttnn.zeros(
                (batch_size, 0, self.config.vlm_hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            image_masks = ttnn.zeros(
                (batch_size, 0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        
        return image_embs, image_masks
    
    def embed_language(
        self,
        tokens: ttnn.Tensor,
        masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Embed language tokens using TTNN embedding."""
        # Use ttnn.embedding
        embeddings = ttnn.embedding(
            tokens,
            self.embed_tokens,
            layout=ttnn.TILE_LAYOUT,
        )
        
        # Scale by sqrt(hidden_dim)
        hidden_dim = embeddings.shape[-1]
        scale = ttnn.from_torch(
            torch.tensor(math.sqrt(hidden_dim), dtype=torch.bfloat16),
            device=self.device,
        )
        embeddings = ttnn.multiply(embeddings, scale)
        
        return embeddings, masks
    
    def embed_prefix(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Create prefix embeddings from images and language."""
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed images
        image_embs, img_masks = self.embed_images(images, image_masks)
        if image_embs.shape[1] > 0:
            embs.append(image_embs)
            pad_masks.append(img_masks)
            # Images use bidirectional attention
            att_masks.append(ttnn.ones_like(img_masks))
        
        # Embed language
        lang_embs, lang_pad_masks = self.embed_language(lang_tokens, lang_masks)
        embs.append(lang_embs)
        pad_masks.append(lang_pad_masks)
        att_masks.append(ttnn.ones_like(lang_pad_masks))
        
        # Concatenate
        if len(embs) > 1:
            prefix_embs = ttnn.concat(embs, dim=1)
            prefix_pad_masks = ttnn.concat(pad_masks, dim=1)
            prefix_att_masks = ttnn.concat(att_masks, dim=1)
        else:
            prefix_embs = embs[0]
            prefix_pad_masks = pad_masks[0]
            prefix_att_masks = att_masks[0]
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks

