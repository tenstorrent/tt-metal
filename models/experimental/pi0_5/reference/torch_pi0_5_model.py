# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 model (PyTorch reference).

Inference flow:
    1. Embed prefix (images + language; state is encoded inside lang_tokens).
    2. forward_vlm to fill prefix KV cache.
    3. Denoising loop:
         - Embed suffix: action_in_proj(noisy_actions), plus
           adarms_cond = time_mlp(swish(time_mlp(sincos(t)))).
         - forward_expert with adarms_cond and prefix_kv_cache.
         - velocity = action_out_proj(expert_output).
         - x_t <- x_t + dt * velocity.
"""

from typing import List, Optional, Tuple

import torch

from models.experimental.pi0.common.configs import (
    DenoiseConfig,
    PaliGemmaConfig,
    PrefixConfig,
    SuffixConfig,
)
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader as PI0WeightLoader
from models.experimental.pi0.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0.reference.torch_denoise import DenoisingModule, KVCacheManager

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone


class Pi0_5Model:
    """PI0.5 reference model (inference)."""

    def __init__(self, config: Pi0_5ModelConfig, weight_loader: PI0WeightLoader):
        assert config.pi05, "Pi0_5Model requires config.pi05=True"
        self.config = config
        self.weight_loader = weight_loader

        self._init_suffix_embedding()
        self._init_backbone()
        self._init_prefix_embedding()
        self._init_denoising()

    def _init_suffix_embedding(self):
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=True,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        self.suffix_embedding = Pi0_5SuffixEmbedding(suffix_config, pi0_weights)

    def _init_backbone(self):
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = Pi0_5PaliGemmaBackbone(paligemma_config, weights)

    def _init_prefix_embedding(self):
        prefix_config = PrefixConfig(vlm_hidden_size=self.config.vlm_config.width)
        self.prefix_embedding = PrefixEmbedding(
            prefix_config,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def _init_denoising(self):
        self.denoising = DenoisingModule(
            DenoiseConfig(num_steps=self.config.num_denoising_steps),
            self._denoise_forward,
        )
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
        **kwargs,
    ) -> torch.Tensor:
        suffix_embs, _, _, adarms_cond = self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)
        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            adarms_cond=adarms_cond,
            past_key_values=kv_cache,
        )
        # pi0.5: no state token to skip; entire expert output is action tokens.
        return self.suffix_embedding.project_output(expert_output)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def forward_inference(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        prefix_embs, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        _, vlm_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        return self.denoising.sample_actions(
            batch_size,
            prefix_kv_cache=vlm_cache,
            device=device,
            state=state,
        )

    sample_actions = forward_inference
