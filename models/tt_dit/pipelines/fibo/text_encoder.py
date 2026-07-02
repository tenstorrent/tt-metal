# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers

import ttnn
from models.tt_dit.encoders.smollm3 import SmolLm3Checkpoint
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


_BOT_TOKEN_ID = 128000


class TextEncoder:
    """FIBO's SmolLM3 text encoder wrapper with PyTorch fallback."""

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
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")

        if use_torch:
            self._torch_encoder = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint_name,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )
            self._torch_encoder.eval()
            self._encoder = None
            self._tracer = None
        else:
            self._torch_encoder = None
            self._encoder = SmolLm3Checkpoint(checkpoint_name).build(
                device=device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._encoder.forward, device=device, prep_run=False)

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        max_sequence_length: int,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        all_prompts = [*negative_prompts, *prompts] if cfg_enabled else list(prompts)

        tokens, mask = self._tokenize(all_prompts, max_sequence_length=max_sequence_length)

        on_event(SectionStart("smollm3_encoding"))
        if self._torch_encoder is not None:
            outputs = self._torch_encoder.forward(
                input_ids=tokens,
                attention_mask=mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states)
        else:
            tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32)
            tt_mask = tensor.from_torch(mask, device=self._device)
            tt_hidden_states = self._tracer(
                tt_tokens,
                mask=tt_mask,
                skip_final_linear=True,
                output_hidden_states=True,
                traced=traced,
            )
            hidden_states = [tensor.to_torch(h) for h in tt_hidden_states]
        on_event(SectionEnd("smollm3_encoding"))

        mask_inv = ~mask.unsqueeze(-1).bool()
        for h in hidden_states:
            h.masked_fill_(mask_inv, 0)

        hidden_states = [h.repeat_interleave(num_images_per_prompt, dim=0) for h in hidden_states]
        mask = mask.repeat_interleave(num_images_per_prompt, dim=0)

        # FIBO uses concat(last_layer, second_to_last_layer) along the channel axis as the
        # transformer's encoder_hidden_states input.
        embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)

        return embeds, hidden_states, mask

    def _tokenize(self, prompts: Sequence[str], *, max_sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(
            list(prompts),
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        empty_rows = torch.tensor([p == "" for p in prompts], dtype=torch.bool)
        input_ids[empty_rows, 0] = _BOT_TOKEN_ID
        attention_mask[empty_rows, 0] = 1
        attention_mask[empty_rows, 1:] = 0

        return input_ids, attention_mask
