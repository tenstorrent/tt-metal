# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemma backbone wrapper for TTNN PI0 implementation.

This module combines vision, language, and action expert components:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)

The dual-expert architecture shares attention layers:
    - VLM and Expert compute separate Q, K, V
    - K, V are concatenated for shared attention
    - Outputs are split and processed through separate MLPs
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from .ttnn_gemma import (
    GemmaConfig,
    GemmaBlockTorch,
    GemmaBlockTTNN,
    rms_norm_torch,
    precompute_freqs_cis_torch,
)
from .ttnn_siglip import (
    SigLIPConfig,
    SigLIPVisionTowerTorch,
    SigLIPVisionTowerTTNN,
    MultiModalProjectorTorch,
    MultiModalProjectorTTNN,
)


@dataclass
class PaliGemmaConfig:
    """Configuration for PaliGemma backbone."""
    vlm_config: GemmaConfig = None
    expert_config: GemmaConfig = None
    siglip_config: SigLIPConfig = None
    vocab_size: int = 257152
    max_seq_len: int = 2048
    
    def __post_init__(self):
        if self.vlm_config is None:
            self.vlm_config = GemmaConfig.gemma_2b()
        if self.expert_config is None:
            self.expert_config = GemmaConfig.gemma_300m()
        if self.siglip_config is None:
            self.siglip_config = SigLIPConfig()


class PaliGemmaBackboneTorch:
    """
    PaliGemma backbone combining VLM and Action Expert (PyTorch).
    
    This implements the dual-expert transformer architecture where:
    - Prefix tokens (images + language) go through VLM backbone
    - Suffix tokens (state + actions) go through action expert
    - Both share attention layers (K, V concatenated)
    """
    
    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize PaliGemma backbone.
        
        Args:
            config: PaliGemma configuration
            weights: Categorized weights from weight_loader
        """
        self.config = config
        
        # VLM components
        self.vlm_embed_tokens = weights["vlm_language"].get("model.embed_tokens.weight")
        self.vlm_norm = weights["vlm_language"].get("model.norm.weight")
        
        # Expert components
        self.expert_norm = weights["action_expert"].get("model.norm.weight")
        
        # VLM blocks
        self.vlm_blocks = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_vlm_block_weights(weights["vlm_language"], i)
            self.vlm_blocks.append(GemmaBlockTorch(config.vlm_config, block_weights, i))
        
        # Expert blocks
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_expert_block_weights(weights["action_expert"], i)
            self.expert_blocks.append(GemmaBlockTorch(config.expert_config, block_weights, i))
        
        # Vision tower
        self.vision_tower = SigLIPVisionTowerTorch(config.siglip_config, weights["vlm_vision"])
        
        # Multi-modal projector
        self.mm_projector = MultiModalProjectorTorch(weights["vlm_projector"])
        
        # Precompute RoPE embeddings
        self.cos, self.sin = precompute_freqs_cis_torch(
            config.vlm_config.head_dim,
            config.max_seq_len,
            config.vlm_config.rope_base,
        )
    
    def _get_vlm_block_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract VLM block weights."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                block_weights[new_key] = value
        return block_weights
    
    def _get_expert_block_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract expert block weights."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                block_weights[new_key] = value
        return block_weights
    
    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Embed images through vision tower and projector.
        
        Args:
            pixel_values: (batch_size, channels, height, width)
        
        Returns:
            (batch_size, num_patches, vlm_width)
        """
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)
    
    def embed_language_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed language tokens.
        
        Args:
            token_ids: (batch_size, seq_len) token IDs
        
        Returns:
            (batch_size, seq_len, vlm_width)
        """
        return F.embedding(token_ids, self.vlm_embed_tokens)
    
    def forward_vlm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through VLM backbone.
        
        Args:
            hidden_states: Prefix embeddings (batch, prefix_len, vlm_width)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: Cached KV from previous forward
            use_cache: Whether to return updated cache
        
        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None
        
        for i, block in enumerate(self.vlm_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                self.cos,
                self.sin,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)
        
        # Final norm
        hidden_states = rms_norm_torch(
            hidden_states,
            self.vlm_norm,
            self.config.vlm_config.rms_norm_eps,
        )
        
        return hidden_states, new_cache
    
    def forward_expert(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through action expert.
        
        Args:
            hidden_states: Suffix embeddings (batch, suffix_len, expert_width)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: Cached KV from VLM prefix (for cross-attention)
            use_cache: Whether to return updated cache
        
        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None
        
        # Precompute RoPE for expert
        cos, sin = precompute_freqs_cis_torch(
            self.config.expert_config.head_dim,
            self.config.max_seq_len,
            self.config.expert_config.rope_base,
        )
        
        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                cos,
                sin,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)
        
        # Final norm
        hidden_states = rms_norm_torch(
            hidden_states,
            self.expert_norm,
            self.config.expert_config.rms_norm_eps,
        )
        
        return hidden_states, new_cache
    
    def forward_shared_attention(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        suffix_mask: Optional[torch.Tensor] = None,
        prefix_position_ids: Optional[torch.Tensor] = None,
        suffix_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with shared attention between VLM and Expert.
        
        In this architecture:
        1. VLM processes prefix, Expert processes suffix independently
        2. In attention layers, K and V are concatenated
        3. Both see the full sequence for attention
        4. MLPs remain separate
        
        For simplicity in this implementation, we process them separately
        and let the suffix attend to prefix via cross-attention mask.
        
        Args:
            prefix_embs: VLM prefix embeddings (batch, prefix_len, vlm_width)
            suffix_embs: Expert suffix embeddings (batch, suffix_len, expert_width)
            prefix_mask: Prefix attention mask
            suffix_mask: Suffix attention mask
            prefix_position_ids: Prefix positions
            suffix_position_ids: Suffix positions
        
        Returns:
            Tuple of (vlm_output, expert_output)
        """
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            prefix_position_ids,
            use_cache=True,
        )
        
        # Process suffix through expert, using VLM cache for cross-attention
        # Note: In full shared attention, we'd need custom attention implementation
        # For now, expert processes independently (suffix attends only to suffix)
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            suffix_position_ids,
            past_key_values=None,  # TODO: Implement cross-attention
            use_cache=False,
        )
        
        return vlm_output, expert_output


class PaliGemmaBackboneTTNN:
    """
    PaliGemma backbone using TTNN operations.
    """
    
    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: "ttnn.Device",
    ):
        """
        Initialize PaliGemma backbone with TTNN.
        
        Args:
            config: PaliGemma configuration
            weights: Categorized PyTorch weights
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        
        # Convert embedding to TTNN
        self.vlm_embed_tokens = ttnn.from_torch(
            weights["vlm_language"]["model.embed_tokens.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Embeddings use row major
            device=device,
        )
        
        # Convert norms
        self.vlm_norm = ttnn.from_torch(
            weights["vlm_language"]["model.norm.weight"].unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.expert_norm = ttnn.from_torch(
            weights["action_expert"]["model.norm.weight"].unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        # Initialize vision tower
        self.vision_tower = SigLIPVisionTowerTTNN(
            config.siglip_config,
            weights["vlm_vision"],
            device,
        )
        
        # Initialize projector
        self.mm_projector = MultiModalProjectorTTNN(weights["vlm_projector"], device)
        
        # Store torch weights for blocks (converted on demand)
        self.torch_weights = weights
        
        # Precompute RoPE on device
        cos, sin = precompute_freqs_cis_torch(
            config.vlm_config.head_dim,
            config.max_seq_len,
        )
        self.cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    def embed_image(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """
        Embed images through vision tower and projector (TTNN).
        
        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)
        
        Returns:
            TTNN tensor (batch_size, num_patches, vlm_width)
        """
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)
    
    def embed_language_tokens(self, token_ids: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Embed language tokens using TTNN.
        
        Args:
            token_ids: TTNN tensor of token IDs
        
        Returns:
            TTNN tensor of embeddings
        """
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)


# Default export
PaliGemmaBackbone = PaliGemmaBackboneTorch

