# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen25vl import Qwen25VlCheckpoint, Qwen25VlEncoder
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX = 34
SEQUENCE_LENGTH = 512


class TextEncoder:
    """QwenImage's Qwen2.5-VL text encoder wrapper with PyTorch fallback."""

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
    ) -> None:
        self._device = device
        self._parallel_config = parallel_config
        self._tokenizer = transformers.Qwen2Tokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")

        self._torch_encoder: transformers.Qwen2_5_VLForConditionalGeneration | None = None
        self._checkpoint: Qwen25VlCheckpoint | None = None
        self._encoder: Qwen25VlEncoder | None = None
        self._tracer: Tracer | None = None

        if use_torch:
            self._torch_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_name, subfolder="text_encoder"
            )
            self._torch_encoder.eval()
        else:
            self._checkpoint = Qwen25VlCheckpoint(checkpoint_name, subfolder="text_encoder")
            self._encoder = self._checkpoint.build(
                device=device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._encoder.forward, device=device, clone_prep_inputs=False)

    def encoder_loaded(self) -> bool:
        return self._encoder is None or self._encoder.is_loaded()

    def reload_encoder_weights(self) -> None:
        """Reload encoder weights to device after deallocation."""
        if self._encoder is None or self._encoder.is_loaded():
            return
        assert self._checkpoint is not None

        logger.info("reloading encoder weights to device...")
        self._checkpoint.load_weights(self._encoder, device=self._device, parallel_config=self._parallel_config)
        ttnn.synchronize_device(self._device)

    def deallocate_encoder_weights(self) -> None:
        """Deallocate encoder weights from device."""
        if self._encoder is None or not self._encoder.is_loaded():
            return

        self._encoder.deallocate_weights()
        ttnn.synchronize_device(self._device)

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
        embeds, mask = self._encode(all_prompts, traced=traced)
        on_event(SectionEnd("qwen_encoding"))

        embeds = embeds.repeat_interleave(num_images_per_prompt, dim=0)
        mask = mask.repeat_interleave(num_images_per_prompt, dim=0)

        embeds[mask == 0] = 0.0

        return embeds[:, PROMPT_DROP_IDX:], mask[:, PROMPT_DROP_IDX:]

    def _encode(self, prompts: Sequence[str], *, traced: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the last hidden state of every prompt and the tokenizer's attention mask."""
        tokens, mask = self._tokenize(prompts, sequence_length=SEQUENCE_LENGTH + PROMPT_DROP_IDX)

        if self._torch_encoder is not None:
            output = self._torch_encoder.forward(
                tokens.to(device=self._torch_encoder.device),
                attention_mask=mask.to(device=self._torch_encoder.device),
                output_hidden_states=True,
            )
            return output.hidden_states[-1].to("cpu"), mask

        assert self._encoder is not None
        assert self._tracer is not None

        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(mask, device=self._device)
        forward = self._tracer if traced else self._encoder.forward
        tt_embeds = forward(tt_tokens, mask=tt_mask, skip_final_linear=True)

        # The tracer reuses its output tensor on every call, so read back before the next one.
        return tensor.to_torch(tt_embeds), mask

    def _tokenize(self, prompts: Sequence[str], *, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(
            list(prompts),
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )
        tokens = tokenized.input_ids
        mask = tokenized.attention_mask

        untruncated_tokens = self._tokenizer(list(prompts), return_tensors="pt", padding="longest").input_ids
        if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
            logger.warning("input text was truncated")

        return tokens, mask
