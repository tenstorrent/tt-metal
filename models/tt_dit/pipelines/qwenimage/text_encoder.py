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
from models.tt_dit.utils.mesh import reshape_for_factor
from models.tt_dit.utils.padding import torch_pad
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager


PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX = 34


class TextEncoder:
    """QwenImage's Qwen2.5-VL text encoder wrapper with PyTorch fallback.

    Each prompt is encoded on its own, padded to the smallest of ``sequence_length_buckets`` it
    fits, counted after the template prefix is dropped. Each CFG pass is padded to the longest
    length among its prompts, and both passes to a common length when they are batched together.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        sequence_length_buckets: Sequence[int],
        use_torch: bool,
    ) -> None:
        self._device = device
        self._parallel_config = parallel_config
        self._tokenizer = transformers.Qwen2Tokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")

        for bucket in sequence_length_buckets:
            if bucket <= 0 or bucket % 32 != 0:
                msg = f"bucket {bucket} must be a positive multiple of 32"
                raise ValueError(msg)
        self._sequence_length_buckets = sorted(set(sequence_length_buckets))

        self._torch_encoder: transformers.Qwen2_5_VLForConditionalGeneration | None = None
        self._encoder: Qwen25VlEncoder | None = None
        # One tracer per encoder input length, created on first use.
        self._tracers: dict[int, Tracer] = {}

        if use_torch:
            self._torch_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_name, subfolder="text_encoder"
            )
            self._torch_encoder.eval()
        else:
            logger.info("loading text encoder weights to device...")
            checkpoint = Qwen25VlCheckpoint(checkpoint_name, subfolder="text_encoder")
            with self._reshape():
                self._encoder = checkpoint.build(device=device, parallel_config=parallel_config, ccl_manager=ccl_manager)
            ttnn.synchronize_device(device)

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        batch_passes: bool,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> list[torch.Tensor]:
        """Returns the prompt embeddings of each CFG pass, the negative pass first.

        With ``batch_passes`` the passes are padded to a common length and batched into one entry.
        Padded positions are zero.
        """
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        if not cfg_enabled:
            cfg_passes = [prompts]
        elif batch_passes:
            cfg_passes = [[*negative_prompts, *prompts]]
        else:
            cfg_passes = [negative_prompts, prompts]

        on_event(SectionStart("qwen_encoding"))
        outputs = []
        with self._reshape():
            for cfg_pass in cfg_passes:
                embeds = [self._encode_prompt(prompt, traced=traced) for prompt in cfg_pass]
                length = max(e.shape[1] for e in embeds)
                embeds = torch.cat([torch_pad(e, length - e.shape[1], dim=1) for e in embeds])
                outputs.append(embeds.repeat_interleave(num_images_per_prompt, dim=0))
        on_event(SectionEnd("qwen_encoding"))

        return outputs

    def _encode_prompt(self, prompt: str, *, traced: bool) -> torch.Tensor:
        """Returns the last hidden state of ``prompt`` past the template prefix, at its bucket length."""
        max_length = self._sequence_length_buckets[-1] + PROMPT_DROP_IDX
        tokens, mask = self._tokenize(PROMPT_TEMPLATE.format(prompt), sequence_length=max_length)

        length = self._bucket(int(mask.sum()) - PROMPT_DROP_IDX) + PROMPT_DROP_IDX
        tokens, mask = tokens[:, :length], mask[:, :length]

        if self._torch_encoder is not None:
            output = self._torch_encoder.forward(
                tokens.to(device=self._torch_encoder.device),
                attention_mask=mask.to(device=self._torch_encoder.device),
                output_hidden_states=True,
            )
            embeds = output.hidden_states[-1].to("cpu")
        else:
            embeds = self._encode_tokens(tokens, mask, traced=traced)

        embeds[mask == 0] = 0.0
        return embeds[:, PROMPT_DROP_IDX:]

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, *, traced: bool) -> torch.Tensor:
        assert self._encoder is not None

        forward = self._encoder.forward
        if traced:
            length = tokens.shape[1]
            if length not in self._tracers:
                self._tracers[length] = Tracer(self._encoder.forward, device=self._device, clone_prep_inputs=False)
            forward = self._tracers[length]

        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(mask, device=self._device)
        tt_embeds = forward(tt_tokens, mask=tt_mask, skip_final_linear=True)

        # The tracer reuses its output tensor on every call, so read back before the next one.
        return tensor.to_torch(tt_embeds)

    def _reshape(self) -> AbstractContextManager[None]:
        """Reshapes the device so that the tensor parallel axis spans exactly the tensor parallel factor."""
        return reshape_for_factor(self._device, self._parallel_config.tensor_parallel)

    def _bucket(self, token_count: int) -> int:
        return min(bucket for bucket in self._sequence_length_buckets if bucket >= token_count)

    def _tokenize(self, prompt: str, *, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )

        if len(self._tokenizer(prompt).input_ids) > sequence_length:
            logger.warning("input text was truncated")

        return tokenized.input_ids, tokenized.attention_mask
