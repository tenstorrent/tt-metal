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

import os
from typing import List, Optional, Tuple

import torch

from models.experimental.pi0_5.common.configs import (
    DenoiseConfig,
    PaliGemmaConfig,
    PrefixConfig,
    SuffixConfig,
)


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Big-vision-style 2D attention mask from pad_masks + att_masks.
    Mirrors openpi.models_pytorch.pi0_pytorch.make_att_2d_masks (line 52).
    """
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def _build_prefix_mask_and_pos(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    embs_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (position_ids, attention_mask_4d_additive) ready for GemmaBlock.
    `attention_mask_4d_additive`: 0.0 where allowed, very-negative where blocked.
    """
    pos = torch.cumsum(pad_masks.to(torch.int64), dim=1) - 1
    mask_2d = _make_att_2d_masks(pad_masks, att_masks)  # (B, S, S) bool
    # additive form: 0 where True, large-negative where False, broadcast to heads.
    neg_inf = torch.finfo(embs_dtype).min if embs_dtype.is_floating_point else -1e9
    additive = torch.zeros_like(mask_2d, dtype=embs_dtype)
    additive.masked_fill_(~mask_2d, neg_inf)
    return pos, additive.unsqueeze(1)  # (B, 1, S, S)


def _use_bf16_vlm() -> bool:
    """`PI0_BF16_VLM=1` -> cast Gemma activations to bf16 throughout forward_vlm
    / forward_expert / projection. Required for upstream openpi pi05 checkpoint
    compat: it was trained in bf16 and our fp32 inference accumulates drift
    that's small per-layer but compounds to cos-sim ~0.85 final actions.
    Off by default to preserve the lerobot-finetune fp32 path.
    """
    return os.environ.get("PI0_BF16_VLM", "").strip().lower() in ("1", "true", "yes", "on")


from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader as PI0WeightLoader
from models.experimental.pi0_5.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0_5.reference.torch_denoise import DenoisingModule, KVCacheManager

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
            DenoiseConfig(
                num_steps=self.config.num_denoising_steps,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
            ),
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
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.suffix_embedding.embed_suffix(
            state, noisy_actions, timestep
        )
        if _use_bf16_vlm():
            suffix_embs = suffix_embs.to(torch.bfloat16)
            adarms_cond = adarms_cond.to(torch.bfloat16) if adarms_cond is not None else None

        # Build cross-attention mask + position_ids for suffix→(prefix+suffix),
        # mirroring openpi.pi0_pytorch.denoise_step (line 422). Without this,
        # suffix tokens attend uniformly to padding prefix slots which corrupts
        # the cached prefix V signal.
        prefix_pad_masks = kwargs.get("prefix_pad_masks")
        attention_mask_4d = None
        position_ids = None
        if prefix_pad_masks is not None and suffix_pad_masks is not None and suffix_att_masks is not None:
            B = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            suffix_len = suffix_pad_masks.shape[1]
            # suffix→prefix: each suffix row can attend to any non-pad prefix token
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(B, suffix_len, prefix_len)
            # suffix→suffix: standard big_vision att 2d
            suffix_att_2d = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # (B, suffix_len, prefix_len + suffix_len)
            neg_inf = torch.finfo(suffix_embs.dtype).min
            additive = torch.zeros_like(full_att_2d, dtype=suffix_embs.dtype)
            additive.masked_fill_(~full_att_2d, neg_inf)
            attention_mask_4d = additive.unsqueeze(1)  # (B, 1, suffix_len, prefix_len + suffix_len)
            prefix_offsets = prefix_pad_masks.to(torch.int64).sum(dim=-1, keepdim=True)
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int64), dim=1) - 1

        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            adarms_cond=adarms_cond,
            past_key_values=kv_cache,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
        )
        # pi0.5: no state token to skip; entire expert output is action tokens.
        if _use_bf16_vlm():
            expert_output = expert_output.to(torch.float32)
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if _use_bf16_vlm():
            prefix_embs = prefix_embs.to(torch.bfloat16)

        # Match openpi's pi0_pytorch.sample_actions: build the cumsum-based
        # position_ids and a 4D additive attention mask from pad+att masks so
        # padding tokens don't get sequential RoPE positions or full
        # attention. Without this, prefix K diverges massively from openpi
        # at padding positions (rotary applied at wrong indices).
        position_ids, attention_mask_4d = _build_prefix_mask_and_pos(
            prefix_pad_masks, prefix_att_masks, prefix_embs.dtype
        )

        _, vlm_cache = self.backbone.forward_vlm(
            prefix_embs,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            use_cache=True,
        )

        return self.denoising.sample_actions(
            batch_size,
            prefix_kv_cache=vlm_cache,
            device=device,
            state=state,
            prefix_pad_masks=prefix_pad_masks,
        )

    sample_actions = forward_inference
