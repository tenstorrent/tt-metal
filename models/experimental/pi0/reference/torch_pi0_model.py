# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Model - PyTorch Reference Implementation.

This module provides the complete PI0 model that orchestrates all components:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers
    - DenoisingModule: Flow matching for action generation
"""

from typing import List, Optional, Tuple

import torch

from models.experimental.pi0.common.configs import (
    PI0ModelConfig,
    SuffixConfig,
    PrefixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.reference.torch_suffix import SuffixEmbedding
from models.experimental.pi0.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0.reference.torch_paligemma import PaliGemmaBackbone
from models.experimental.pi0.reference.torch_denoise import DenoisingModule, KVCacheManager


# openpi uses this specific constant (not torch.finfo(dtype).min) for the
# masked positions of the additive attention mask — finfo.min overflows bfloat16
# and produces NaN on cast, whereas -2.3819763e38 survives the round-trip.
_ATT_MASK_NEG_INF = -2.3819763e38


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Lerobot/openpi `make_att_2d_masks` reference: build a 2D attention mask
    where each query attends to keys whose cumulative `att_masks` value is ≤ its
    own and both query and key are non-padded.
    """
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _bool_mask_to_additive(mask_2d: torch.Tensor) -> torch.Tensor:
    """Convert a (B, L, L) bool attention mask to a (B, 1, L, L) additive mask
    with 0 for attend / large negative for skip. Uses openpi's specific value
    (-2.3819763e38) rather than finfo.min, because finfo.min overflows bfloat16
    and produces NaN on cast.
    """
    additive = torch.zeros_like(mask_2d, dtype=torch.float32)
    additive.masked_fill_(~mask_2d, _ATT_MASK_NEG_INF)
    return additive.unsqueeze(1)


class PI0Model:
    """
    Complete PI0 model implementation (PyTorch).

    This class orchestrates all components for inference.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
    ):
        """
        Initialize PI0 model.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
        """
        self.config = config
        self.weight_loader = weight_loader

        # Initialize components (order matters: backbone before prefix)
        self._init_suffix_embedding()
        self._init_backbone()
        self._init_prefix_embedding()
        self._init_denoising()

    def _init_suffix_embedding(self):
        """Initialize suffix embedding module."""
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        self.suffix_embedding = SuffixEmbedding(suffix_config, pi0_weights)

    def _init_prefix_embedding(self):
        """Initialize prefix embedding module (after backbone)."""
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
        )
        # Create prefix embedding with backbone's embedding functions
        self.prefix_embedding = PrefixEmbedding(
            prefix_config,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def _init_backbone(self):
        """Initialize PaliGemma backbone."""
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackbone(paligemma_config, weights)

    def _init_denoising(self):
        """Initialize denoising module."""
        denoise_config = DenoiseConfig(
            num_steps=self.config.num_denoising_steps,
        )
        self.denoising = DenoisingModule(denoise_config, self._denoise_forward)

        # KV cache for inference
        self.kv_cache = KVCacheManager(
            num_layers=self.config.expert_config.depth,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.config.expert_config.num_kv_heads,
            head_dim=self.config.expert_config.head_dim,
        )

    def _denoise_forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        state: Optional[torch.Tensor] = None,
        prefix_offset: int = 0,
        prefix_pad_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for denoising (predicts velocity).

        The `prefix_offset` and `prefix_pad_masks` kwargs are threaded in from
        `forward_inference` via the denoising module's `**forward_kwargs`
        pathway. They carry the state needed to match lerobot/openpi's RoPE +
        attention-mask semantics for the expert cross-attention.
        """
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.suffix_embedding.embed_suffix(
            state,
            noisy_actions,
            timestep,
        )

        bsz, suffix_len = suffix_embs.shape[0], suffix_embs.shape[1]
        device = suffix_embs.device

        # Suffix RoPE positions: start at the actual non-pad prefix length so
        # suffix tokens are rotated at the correct global positions (matches
        # lerobot's `prefix_offset + cumsum(suffix_mask) - 1`).
        position_ids = (
            torch.arange(
                prefix_offset,
                prefix_offset + suffix_len,
                device=device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        # Build the additive attention mask for cross-attn over cached prefix.
        # Required so suffix doesn't attend to padded prefix positions in the
        # KV cache.
        if prefix_pad_masks is not None:
            full_pad = torch.cat([prefix_pad_masks, suffix_pad_masks.bool()], dim=1)
            # Prefix block is fully bidirectional (att=0 everywhere); suffix
            # carries its own causal-style att pattern.
            prefix_att_zeros = torch.zeros(prefix_pad_masks.shape, dtype=torch.bool, device=device)
            full_att = torch.cat([prefix_att_zeros, suffix_att_masks.bool()], dim=1)
            full_2d = _make_att_2d_masks(full_pad, full_att)
            # Only the suffix rows (queries), all columns (prefix+suffix keys).
            suffix_2d = full_2d[:, -suffix_len:, :]
            additive_suffix_mask = _bool_mask_to_additive(suffix_2d).to(suffix_embs.dtype)
        else:
            additive_suffix_mask = None

        # Forward through expert (pass adarms_cond for Pi0.5)
        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            attention_mask=additive_suffix_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

        # Project back to action dimension (skip state token if PI0)
        if not self.config.pi05:
            action_output = expert_output[:, 1:, :]
        else:
            action_output = expert_output

        velocity = self.suffix_embedding.project_output(action_output)

        return velocity

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language to form prefix."""
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed state and actions to form suffix."""
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def forward_inference(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference (generates actions via denoising).

        Returns:
            Generated actions (batch_size, action_horizon, action_dim)
        """
        batch_size = state.shape[0]
        device = state.device

        # Prefill: process prefix and cache KV
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Build additive 4D attention mask (lerobot/openpi make_att_2d_masks
        # → masked_fill_ → unsqueeze(1)). Required so padded prefix positions
        # (right-padded lang tokens) do not contribute to attention.
        prefix_att_2d = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        additive_mask = _bool_mask_to_additive(prefix_att_2d).to(prefix_embs.dtype)

        # Position ids: cumsum(pad_mask) - 1, clamped at 0 for any leading pads.
        position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        position_ids.clamp_(min=0)

        # Non-pad prefix length (assumes batch is homogeneous — true for Pi0.5
        # libero where batch_size == 1).
        prefix_offset = int(prefix_pad_masks.long().sum(dim=1)[0].item())

        _, vlm_cache = self.backbone.forward_vlm(
            prefix_embs,
            attention_mask=additive_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # Denoise to generate actions. prefix_offset / prefix_pad_masks flow
        # through DenoisingModule.sample_actions -> denoise_step via
        # **forward_kwargs and land in _denoise_forward.
        actions = self.denoising.sample_actions(
            batch_size,
            prefix_kv_cache=vlm_cache,
            device=device,
            state=state,
            prefix_offset=prefix_offset,
            prefix_pad_masks=prefix_pad_masks,
        )

        return actions

    # Alias for compatibility with ttnn_pi0_reference API
    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for forward_inference (sample_actions API)."""
        return self.forward_inference(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
