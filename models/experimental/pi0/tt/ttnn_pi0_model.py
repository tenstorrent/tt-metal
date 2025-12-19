# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Model - TTNN Implementation.

This module provides the complete PI0 model using TTNN operations:
    - TtPrefixEmbedding: Images + language → embeddings
    - TtSuffixEmbedding: State + actions + timestep → embeddings
    - TtPaliGemmaBackbone: VLM + Action Expert transformers
    - TtDenoisingModule: Flow matching for action generation
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import (
    PI0ModelConfig,
    SuffixConfig,
    PrefixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_suffix import TtSuffixEmbedding, convert_suffix_weights_to_ttnn
from models.experimental.pi0.tt.ttnn_prefix import TtPrefixEmbedding
from models.experimental.pi0.tt.ttnn_paligemma import TtPaliGemmaBackbone
from models.experimental.pi0.tt.ttnn_denoise import TtDenoisingModule, TtKVCacheManager


class TtPI0Model:
    """
    Complete PI0 model implementation using TTNN.
    
    This class orchestrates all components for training and inference
    with hardware acceleration on Tenstorrent devices.
    """
    
    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
        device: ttnn.Device,
    ):
        """
        Initialize PI0 model with TTNN.
        
        Args:
            config: Model configuration
            weight_loader: Loaded weights
            device: TTNN device
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device
        
        # Initialize components
        self._init_backbone()
        self._init_suffix_embedding()
        self._init_prefix_embedding()
        self._init_denoising()
    
    def _init_backbone(self):
        """Initialize PaliGemma backbone."""
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = TtPaliGemmaBackbone(paligemma_config, weights, self.device)
    
    def _init_suffix_embedding(self):
        """Initialize suffix embedding module."""
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        ttnn_weights = convert_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = TtSuffixEmbedding(suffix_config, ttnn_weights, self.device)
    
    def _init_prefix_embedding(self):
        """Initialize prefix embedding module."""
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
        )
        self.prefix_embedding = TtPrefixEmbedding(
            prefix_config,
            self.backbone.vision_tower,
            self.backbone.mm_projector,
            self.backbone.vlm_embed_tokens,
            self.device,
        )
    
    def _init_denoising(self):
        """Initialize denoising module."""
        denoise_config = DenoiseConfig(
            num_steps=self.config.num_denoising_steps,
        )
        self.denoising = TtDenoisingModule(denoise_config, self._denoise_forward, self.device)
        
        # KV cache for inference
        self.kv_cache = TtKVCacheManager(
            num_layers=self.config.expert_config.depth,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.config.expert_config.num_kv_heads,
            head_dim=self.config.expert_config.head_dim,
            device=self.device,
        )
    
    def _denoise_forward(
        self,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
        kv_cache: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        state: Optional[ttnn.Tensor] = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """Forward pass for denoising (predicts velocity)."""
        # Embed suffix
        suffix_embs, _, _, _ = self.suffix_embedding.embed_suffix(
            state,
            noisy_actions,
            timestep,
        )
        
        # Forward through expert
        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            past_key_values=kv_cache,
        )
        
        # Project back to action dimension
        if not self.config.pi05:
            # Extract action tokens (skip state token)
            # Convert to torch for slicing, then back to TTNN
            expert_torch = ttnn.to_torch(expert_output)
            action_output_torch = expert_torch[:, 1:, :]
            action_output = ttnn.from_torch(
                action_output_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            action_output = expert_output
        
        velocity = self.suffix_embedding.project_output(action_output)
        
        return velocity
    
    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Embed images and language to form prefix."""
        return self.prefix_embedding.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
    
    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Embed state and actions to form suffix."""
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)
    
    def forward_training(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
        state: ttnn.Tensor,
        actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass for training (computes velocity for loss).
        
        Returns:
            Predicted velocity (TTNN tensor)
        """
        # Embed prefix
        prefix_embs, _, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Embed suffix
        suffix_embs, _, _, _ = self.embed_suffix(state, actions, timestep)
        
        # Forward through backbone
        _, expert_output = self.backbone.forward_shared_attention(
            prefix_embs,
            suffix_embs,
        )
        
        # Project to actions
        if not self.config.pi05:
            expert_torch = ttnn.to_torch(expert_output)
            action_output_torch = expert_torch[:, 1:, :]
            action_output = ttnn.from_torch(
                action_output_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            action_output = expert_output
        
        velocity = self.suffix_embedding.project_output(action_output)
        
        return velocity
    
    def forward_inference(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
        state: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass for inference (generates actions via denoising).
        
        Returns:
            Generated actions (TTNN tensor)
        """
        batch_size = state.shape[0]
        
        # Prefill: process prefix and cache KV
        prefix_embs, _, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        _, vlm_cache = self.backbone.forward_vlm(
            prefix_embs,
            use_cache=True,
        )
        
        # Denoise to generate actions
        actions = self.denoising.sample_actions(
            batch_size,
            prefix_kv_cache=vlm_cache,
            state=state,
        )
        
        return actions

