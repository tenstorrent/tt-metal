# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers
from loguru import logger

import ttnn

from ...encoders.mistral3.model_mistral3 import Mistral3Encoder
from ...encoders.transformer import RopeConfig
from ...layers.module import Module
from ...utils import cache, tensor
from .system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from transformers import PreTrainedTokenizerBase

    from ...parallel.config import EncoderParallelConfig
    from ...parallel.manager import CCLManager


class PromptEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch_encoder: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._tokenizer = transformers.LlamaTokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer")
        assert isinstance(self._tokenizer, transformers.LlamaTokenizerFast)

        if use_torch_encoder:
            self._encoder = _load_torch_encoder(checkpoint_name)
            return

        self._encoder = Mistral3Encoder(
            vocab_size=131072,
            head_size=128,
            embed_size=5120,
            ff_size=32768,
            num_layers=40,
            num_heads=32,
            num_kv_heads=8,
            norm_eps=1e-05,
            attn_qkv_bias=False,
            attn_out_bias=False,
            rope_config=RopeConfig(theta=1000000000),
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        def get_torch_state_dict() -> dict[str, torch.Tensor]:
            return Mistral3Encoder.convert_state(_load_torch_encoder(checkpoint_name).state_dict())

        cache.load_model(
            self._encoder,
            model_name="flux2",
            subfolder="text_encoder",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            dtype="bf16",
            get_torch_state_dict=get_torch_state_dict,
        )

    def encode(
        self, prompts: Sequence[str], *, num_images_per_prompt: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_prompt_embeds(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=sequence_length,
            output_from_layers=[10, 20, 30],
            tokenizer=self._tokenizer,
            encoder=self._encoder,
            device=self._device,
        )

    def upsample(self, prompts: Sequence[str], *, max_length: int, temperature: float) -> list[str]:
        return _upsample_prompts(
            prompts,
            max_length=max_length,
            temperature=temperature,
            tokenizer=self._tokenizer,
            encoder=self._encoder,
            device=self._device,
        )


def _load_torch_encoder(checkpoint_name: str) -> transformers.Mistral3ForConditionalGeneration:
    model = transformers.Mistral3ForConditionalGeneration.from_pretrained(checkpoint_name, subfolder="text_encoder")
    model.eval()
    return model


def _get_prompt_embeds(
    prompts: Sequence[str],
    *,
    encoder: Module | torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    num_images_per_prompt: int,
    output_from_layers: Sequence[int],
    device: ttnn.MeshDevice | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    conversation = _format_input(prompts, system_message=SYSTEM_MESSAGE)

    tokenizer_out = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_dict=True,
    )
    tokens = tokenizer_out.input_ids
    mask = tokenizer_out.attention_mask

    untruncated_out = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        padding="longest",
        return_dict=True,
    )
    untruncated_tokens = untruncated_out.input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("input text was truncated")

    if isinstance(encoder, Module):
        assert device is not None

        tt_tokens = tensor.from_torch(tokens, device=device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(mask, device=device)

        tt_hidden_states = encoder.forward(
            tt_tokens,
            mask=tt_mask,
            skip_final_linear=True,
            output_hidden_states=True,
        )
        tt_prompt_embeds = ttnn.concat([tt_hidden_states[k] for k in output_from_layers], dim=-1)

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        prompt_embeds.masked_fill_(~mask.unsqueeze(-1).bool(), 0.0)
    else:
        tokens = tokens.to(device=encoder.device)

        with torch.no_grad():
            output = encoder.forward(
                tokens,
                mask=mask,
                output_hidden_states=True,
            )
        prompt_embeds = torch.concat([output.hidden_states[k] for k in output_from_layers], dim=-1).to("cpu")

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    mask = mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, mask


def _upsample_prompts(
    prompts: Sequence[str],
    *,
    encoder: Module | torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    temperature: float,
    device: ttnn.MeshDevice | None,
) -> list[str]:
    conversation = _format_input(prompts, system_message=SYSTEM_MESSAGE_UPSAMPLING_T2I)

    tokenizer_out = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        add_generation_prompt=True,
    )
    tokens = tokenizer_out.input_ids
    mask = tokenizer_out.attention_mask

    if isinstance(encoder, Module):
        assert device is not None

        tt_tokens = tensor.from_torch(tokens, device=device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(mask, device=device)

        tt_output = encoder.generate(
            tt_tokens,
            mask=tt_mask,
            eos_tokens=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature,
        )

        output_tokens = ttnn.to_torch(ttnn.get_device_tensors(tt_output.tokens)[0])
    else:
        tokens = tokens.to(device=encoder.device)

        with torch.no_grad():
            output_tokens = encoder.generate(
                tokens,
                attention_mask=mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
            )

    input_length = tokens.shape[1]
    generated_tokens = output_tokens[:, input_length:]

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _format_input(prompts: Sequence[str], *, system_message: str) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in prompts
    ]
