# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import T5EncoderModel, T5TokenizerFast

from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback

if TYPE_CHECKING:
    from collections.abc import Sequence


class TextEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        force_zeros_for_empty_prompt: bool,
        max_sequence_length: int = 256,
    ) -> None:
        self.text_encoder = T5EncoderModel.from_pretrained(
            checkpoint_name, subfolder="text_encoder", torch_dtype=torch.float32
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._force_zeros_for_empty_prompt = force_zeros_for_empty_prompt
        self._max_sequence_length = max_sequence_length

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_videos_per_prompt: int = 1,
        disable_attention_mask: bool = False,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Encode both cfg branches and return them in (uncond, cond) order."""
        on_event(SectionStart("t5_encoding"))
        cond_embeds, cond_mask = self._get_t5_prompt_embeds(
            prompt=list(prompts),
            num_videos_per_prompt=num_videos_per_prompt,
            disable_attention_mask=disable_attention_mask,
        )
        uncond_embeds, uncond_mask = self._get_t5_prompt_embeds(
            prompt=list(negative_prompts),
            num_videos_per_prompt=num_videos_per_prompt,
            disable_attention_mask=disable_attention_mask,
        )
        on_event(SectionEnd("t5_encoding"))

        return (uncond_embeds, cond_embeds), (uncond_mask, cond_mask)

    def _get_t5_prompt_embeds(
        self,
        prompt: list[str],
        *,
        num_videos_per_prompt: int,
        disable_attention_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_sequence_length = self._max_sequence_length
        device = "cpu"
        dtype = self.text_encoder.dtype

        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        if disable_attention_mask:
            prompt_attention_mask.fill_(value=True)

        # The original Mochi implementation zeros out empty negative prompts but this can lead to
        # overflow when placing the entire pipeline under the autocast context, hence the option.
        if self._force_zeros_for_empty_prompt and (prompt == "" or prompt[-1] == ""):
            text_input_ids = torch.zeros_like(text_input_ids, device=device)
            prompt_attention_mask = torch.zeros_like(prompt_attention_mask, dtype=torch.bool, device=device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask
