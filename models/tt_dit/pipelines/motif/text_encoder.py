# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers

from ...encoders.clip.encoder_pair import CLIPTokenizerEncoderPair
from ...encoders.t5.encoder_pair import T5TokenizerEncoderPair
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ..events import PipelineEventCallback, SectionEnd, SectionStart, null_callback

if TYPE_CHECKING:
    from collections.abc import Sequence


class TextEncoder:
    def __init__(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        enable_t5: bool,
        use_torch_clip_encoder: bool,
        use_torch_t5_encoder: bool,
    ) -> None:
        device = ccl_manager.mesh_device

        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        # suppress noisy "load report warning"
        verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()

        self._clip_l = CLIPTokenizerEncoderPair(
            "openai/clip-vit-large-patch14",
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            sequence_length=None,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_clip_encoder,
        )

        self._clip_g = CLIPTokenizerEncoderPair(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            sequence_length=None,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_clip_encoder,
        )

        self._t5 = T5TokenizerEncoderPair(
            "google/flan-t5-xxl",
            zero_masking=True,
            sequence_length=256,
            empty_sequence_length=None,
            embedding_dim=4096,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_t5_encoder,
            enabled=enable_t5,
            use_attention_mask=True,
        )

        transformers.logging.set_verbosity(verbosity)

    @torch.no_grad()
    def encode(
        self,
        prompts: tuple[Sequence[str], Sequence[str], Sequence[str]],
        *,
        num_images_per_prompt: int,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        on_event(SectionStart("clip_encoding"))
        clip_l, pooled_clip_l = self._clip_l.encode(
            prompts=prompts[0], num_images_per_prompt=num_images_per_prompt, enable_tracing=traced
        )
        clip_g, pooled_clip_g = self._clip_g.encode(
            prompts=prompts[1], num_images_per_prompt=num_images_per_prompt, enable_tracing=traced
        )
        on_event(SectionEnd("clip_encoding"))

        on_event(SectionStart("t5_encoding"))
        t5 = self._t5.encode(prompts=prompts[2], num_images_per_prompt=num_images_per_prompt, enable_tracing=traced)
        on_event(SectionEnd("t5_encoding"))

        clip = torch.cat([clip_l, clip_g], dim=-1)
        clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))

        embeds = torch.cat([clip, t5], dim=-2)
        pooled_embeds = torch.cat([pooled_clip_l, pooled_clip_g], dim=-1)

        return embeds, pooled_embeds

    def encode_cfg(
        self,
        pos_prompts: tuple[Sequence[str], Sequence[str], Sequence[str]],
        neg_prompts: tuple[Sequence[str | None], Sequence[str | None], Sequence[str | None]],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_context, pos_pooled = self.encode(
            pos_prompts,
            num_images_per_prompt=num_images_per_prompt,
            traced=traced,
            on_event=on_event,
        )

        if not cfg_enabled:
            return pos_context, pos_pooled, pos_context, pos_pooled

        neg_prompts_1, neg_prompts_2, neg_prompts_3 = neg_prompts
        no_neg_prompt = [x is None for x in neg_prompts_1]
        neg_prompts_1 = [x if x is not None else "" for x in neg_prompts_1]
        neg_prompts_2 = [x if x is not None else "" for x in neg_prompts_2]
        neg_prompts_3 = [x if x is not None else "" for x in neg_prompts_3]

        neg_context, neg_pooled = self.encode(
            (neg_prompts_1, neg_prompts_2, neg_prompts_3),
            num_images_per_prompt=num_images_per_prompt,
            traced=traced,
            on_event=on_event,
        )

        early_context = torch.cat([neg_context, pos_context], dim=0)
        early_pooled = torch.cat([neg_pooled, pos_pooled], dim=0)

        masked_neg_context = neg_context.clone()
        masked_neg_pooled = neg_pooled.clone()
        for i, no_neg in enumerate(no_neg_prompt):
            if no_neg:
                masked_neg_context[i] = 0
                masked_neg_pooled[i] = 0

        late_context = torch.cat([masked_neg_context, pos_context], dim=0)
        late_pooled = torch.cat([masked_neg_pooled, pos_pooled], dim=0)

        return early_context, early_pooled, late_context, late_pooled
