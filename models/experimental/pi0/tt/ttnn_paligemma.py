# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemma backbone wrapper - TTNN Implementation.

This module combines vision, language, and action expert components using TTNN:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import GemmaConfig, SigLIPConfig, PaliGemmaConfig
from models.experimental.pi0.reference.torch_gemma import precompute_freqs_cis
from models.experimental.pi0.tt.ttnn_gemma import TtGemmaBlock, rms_norm
from models.experimental.pi0.tt.ttnn_siglip import TtSigLIPVisionTower, TtMultiModalProjector


class TtPaliGemmaBackbone:
    """PaliGemma backbone using TTNN operations."""
    
    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: ttnn.Device,
    ):
        """
        Initialize PaliGemma backbone with TTNN.
        
        Args:
            config: PaliGemma configuration
            weights: Categorized PyTorch weights
            device: TTNN device
        """
        self.config = config
        self.device = device
        
        # Convert embedding to TTNN (handle tied embeddings)
        embed_weight = (weights["vlm_language"].get("model.embed_tokens.weight") or 
                       weights["vlm_language"].get("lm_head.weight"))
        if embed_weight is not None:
            self.vlm_embed_tokens = ttnn.from_torch(
                embed_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        else:
            self.vlm_embed_tokens = None
        
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
        self.vision_tower = TtSigLIPVisionTower(
            config.siglip_config,
            weights["vlm_vision"],
            device,
        )
        
        # Initialize projector
        self.mm_projector = TtMultiModalProjector(weights["vlm_projector"], device)
        
        # Store torch weights for blocks
        self.torch_weights = weights
        
        # Precompute RoPE on device
        cos, sin = precompute_freqs_cis(
            config.vlm_config.head_dim,
            config.max_seq_len,
        )
        self.cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Expert RoPE
        expert_cos, expert_sin = precompute_freqs_cis(
            config.expert_config.head_dim,
            config.max_seq_len,
        )
        self.expert_cos = ttnn.from_torch(expert_cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.expert_sin = ttnn.from_torch(expert_sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Initialize VLM transformer blocks
        self.vlm_blocks = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_block_weights_ttnn(weights["vlm_language"], i)
            self.vlm_blocks.append(TtGemmaBlock(config.vlm_config, block_weights, i, device))
        
        # Initialize Expert transformer blocks
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_block_weights_ttnn(weights["action_expert"], i)
            self.expert_blocks.append(TtGemmaBlock(config.expert_config, block_weights, i, device))
    
    def _get_block_weights_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:
        """Extract block weights and convert to TTNN."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                # Transpose weight matrices for TTNN linear
                if "weight" in new_key and "layernorm" not in new_key and "norm" not in new_key:
                    value = value.T.contiguous()
                
                block_weights[new_key] = ttnn.from_torch(
                    value if len(value.shape) > 1 else value.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
        return block_weights
    
    def embed_image(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """Embed images through vision tower and projector (TTNN)."""
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)
    
    def embed_language_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Embed language tokens using TTNN."""
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)
    
    def forward_vlm(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """Forward pass through VLM backbone using TTNN."""
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
        hidden_states = rms_norm(
            hidden_states,
            self.vlm_norm,
            self.config.vlm_config.rms_norm_eps,
        )
        
        return hidden_states, new_cache
    
    def forward_expert(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """Forward pass through action expert using TTNN."""
        new_cache = [] if use_cache else None
        
        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                self.expert_cos,
                self.expert_sin,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)
        
        # Final norm
        hidden_states = rms_norm(
            hidden_states,
            self.expert_norm,
            self.config.expert_config.rms_norm_eps,
        )
        
        return hidden_states, new_cache
    
    def forward_shared_attention(
        self,
        prefix_embs: ttnn.Tensor,
        suffix_embs: ttnn.Tensor,
        prefix_mask: Optional[ttnn.Tensor] = None,
        suffix_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass with shared attention between VLM and Expert."""
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            use_cache=True,
        )
        
        # Process suffix through expert
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            past_key_values=None,
            use_cache=False,
        )
        
        return vlm_output, expert_output

