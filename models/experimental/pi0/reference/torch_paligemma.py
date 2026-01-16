# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemma backbone wrapper - PyTorch Reference Implementation.

This module combines vision, language, and action expert components:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import PaliGemmaConfig
from models.experimental.pi0.reference.torch_gemma import GemmaBlock, rms_norm, precompute_freqs_cis
from models.experimental.pi0.reference.torch_siglip import SigLIPVisionTower, MultiModalProjector


class PaliGemmaBackbone:
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

        # VLM components (handle tied embeddings)
        embed_tokens = weights["vlm_language"].get("model.embed_tokens.weight")
        self.vlm_embed_tokens = (
            embed_tokens if embed_tokens is not None else weights["vlm_language"].get("lm_head.weight")
        )
        self.vlm_norm = weights["vlm_language"].get("model.norm.weight")

        # Expert components
        self.expert_norm = weights["action_expert"].get("model.norm.weight")

        # VLM blocks
        self.vlm_blocks = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_block_weights(weights["vlm_language"], i)
            self.vlm_blocks.append(GemmaBlock(config.vlm_config, block_weights, i))

        # Expert blocks
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_block_weights(weights["action_expert"], i)
            self.expert_blocks.append(GemmaBlock(config.expert_config, block_weights, i))

        # Vision tower
        self.vision_tower = SigLIPVisionTower(config.siglip_config, weights["vlm_vision"])

        # Multi-modal projector
        self.mm_projector = MultiModalProjector(weights["vlm_projector"])

        # Precompute RoPE embeddings
        self.cos, self.sin = precompute_freqs_cis(
            config.vlm_config.head_dim,
            config.max_seq_len,
            config.vlm_config.rope_base,
        )

    def _get_block_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract block weights."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                block_weights[new_key] = value
        return block_weights

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed images through vision tower and projector."""
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)

    def embed_language_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed language tokens."""
        return F.embedding(token_ids, self.vlm_embed_tokens)

    def forward_vlm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through VLM backbone."""
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through action expert."""
        new_cache = [] if use_cache else None

        # Precompute RoPE for expert
        cos, sin = precompute_freqs_cis(
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
        hidden_states = rms_norm(
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
        """Forward pass with shared attention between VLM and Expert."""
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            prefix_position_ids,
            use_cache=True,
        )

        # Process suffix through expert
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            suffix_position_ids,
            past_key_values=None,
            use_cache=False,
        )

        return vlm_output, expert_output
