# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn
from models.tt_dit.encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback

if TYPE_CHECKING:
    from collections.abc import Sequence


PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX = 34


class TextEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
    ) -> None:
        self._encoder = Qwen25VlTokenizerEncoderPair(
            checkpoint_name,
            tokenizer_subfolder="tokenizer",
            encoder_subfolder="text_encoder",
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch,
            is_fsdp=True,  # Best configuration for wh t3k and galaxy
        )

    def encoder_loaded(self) -> bool:
        return self._encoder.encoder_loaded()

    def reload_encoder_weights(self) -> None:
        self._encoder.reload_encoder_weights()

    def deallocate_encoder_weights(self) -> None:
        self._encoder.deallocate_encoder_weights()

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        all_prompts = [*negative_prompts, *prompts] if cfg_enabled else list(prompts)
        all_prompts = [PROMPT_TEMPLATE.format(e) for e in all_prompts]

        on_event(SectionStart("qwen_encoding"))
        embeds, mask = self._encoder.encode(
            all_prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=512 + PROMPT_DROP_IDX,
            enable_tracing=traced,
        )
        on_event(SectionEnd("qwen_encoding"))

        embeds[torch.logical_not(mask)] = 0.0

        return embeds[:, PROMPT_DROP_IDX:], mask[:, PROMPT_DROP_IDX:]
