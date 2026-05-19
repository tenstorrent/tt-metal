# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, rms_norm, precompute_freqs_cis
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower, MultiModalProjector
from models.experimental.pi0_5.reference.torch_siglip_hf import HFSigLIPVisionTower, use_hf_siglip


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

        # Vision tower — opt-in HF wrapper for openpi/upstream pi05_libero compat.
        if use_hf_siglip():
            self.vision_tower = HFSigLIPVisionTower(config.siglip_config, weights["vlm_vision"])
        else:
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
        """Embed images through vision tower and projector.

        When `PI0_SIGLIP_HF=1`, the vision tower runs the openpi mixed-bf16
        scheme and returns bf16 — mm_projector then runs in bf16 too (also
        matching openpi). We cast back to `pixel_values.dtype` before
        returning so the rest of our fp32-by-default pipeline stays unaware.
        """
        vision_features = self.vision_tower.forward(pixel_values)
        projected = self.mm_projector.forward(vision_features)
        return projected.to(pixel_values.dtype)

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


# ============================================================================
# pi0.5-specific additions (subclasses, overrides)
# ============================================================================

from typing import Dict, List, Optional, Tuple

import torch

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.reference.torch_gemma import precompute_freqs_cis

from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, ada_rms_norm_no_gate


class Pi0_5PaliGemmaBackbone(PaliGemmaBackbone):
    def __init__(self, config: PaliGemmaConfig, weights: Dict[str, torch.Tensor]):
        # The parent builds plain GemmaBlocks for the expert and reads
        # `input_layernorm.weight` / `post_attention_layernorm.weight` /
        # `model.norm.weight` from `weights["action_expert"]`. PI0.5 uses
        # adaRMS everywhere in the expert and those tensors don't exist in
        # the checkpoint. Inject zero placeholders so super().__init__ runs,
        # then discard the parent's expert artifacts.
        ae = weights["action_expert"]
        expert_w = config.expert_config.width
        placeholders = {}
        for i in range(config.expert_config.depth):
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                key = f"model.layers.{i}.{name}"
                if key not in ae:
                    placeholders[key] = torch.zeros(expert_w)
        if "model.norm.weight" not in ae:
            placeholders["model.norm.weight"] = torch.zeros(expert_w)
        ae.update(placeholders)

        super().__init__(config, weights)

        # Replace expert blocks with adaRMS variants.
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_block_weights(weights["action_expert"], i)
            self.expert_blocks.append(AdaRMSGemmaBlock(config.expert_config, block_weights, i))

        # Final expert norm: adaRMS, not plain.
        self.expert_norm_mod_weight = ae["model.norm.dense.weight"]
        self.expert_norm_mod_bias = ae.get("model.norm.dense.bias")

    def forward_expert(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        new_cache = [] if use_cache else None

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
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        hidden_states = ada_rms_norm_no_gate(
            hidden_states,
            adarms_cond,
            self.expert_norm_mod_weight,
            self.expert_norm_mod_bias,
            self.config.expert_config.rms_norm_eps,
        )
        return hidden_states, new_cache
