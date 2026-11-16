# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from loguru import logger
from transformers import PreTrainedTokenizerBase

import ttnn
from models.tt_dit.encoders.mistral3.model_mistral3 import Mistral3Encoder
from models.tt_dit.encoders.qwen3.model_qwen3 import Qwen3Encoder
from models.tt_dit.encoders.transformer import WEIGHT_CACHE_DTYPE
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.flux2.system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I
from models.tt_dit.utils import cache, tensor


class PromptEncoder:
    """The FLUX.2 prompt encoder.

    Mistral 3 for FLUX.2-dev, Qwen3 for FLUX.2 Klein (diffusers
    ``Flux2KleinPipeline._get_qwen3_prompt_embeds``), chosen by the checkpoint's text encoder.
    """

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
        self._parallel_config = parallel_config
        self._checkpoint_name = checkpoint_name

        hf_config = transformers.AutoConfig.from_pretrained(checkpoint_name, subfolder="text_encoder")
        self._is_qwen3 = hf_config.model_type == "qwen3"

        if self._is_qwen3:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
            self._output_from_layers = [9, 18, 27]
            self._encoder_cls = Qwen3Encoder
            self._load_torch_encoder = _load_torch_qwen3_encoder
        else:
            self._tokenizer = transformers.LlamaTokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer")
            assert isinstance(self._tokenizer, transformers.LlamaTokenizerFast)
            self._output_from_layers = [10, 20, 30]
            self._encoder_cls = Mistral3Encoder
            self._load_torch_encoder = _load_torch_mistral3_encoder

        if use_torch_encoder:
            self._encoder = self._load_torch_encoder(checkpoint_name)
        else:
            self._encoder = self._encoder_cls(
                self._encoder_cls.config_from_hf(hf_config),
                device=device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )

    def load_weights(self) -> None:
        if not isinstance(self._encoder, Module):
            return

        def get_torch_state_dict() -> dict[str, torch.Tensor]:
            return self._encoder_cls.convert_state(self._load_torch_encoder(self._checkpoint_name).state_dict())

        cache.load_model(
            self._encoder,
            model_name=Path(self._checkpoint_name).name,
            subfolder="text_encoder",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            mesh_device=self._device,
            dtype=WEIGHT_CACHE_DTYPE,
            get_torch_state_dict=get_torch_state_dict,
        )

    def encode(
        self,
        prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        sequence_length: int,
        traced: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenize = _tokenize_qwen3 if self._is_qwen3 else _tokenize_mistral3
        tokens, mask = tokenize(prompts, tokenizer=self._tokenizer, sequence_length=sequence_length)

        return _get_prompt_embeds(
            tokens,
            mask,
            num_images_per_prompt=num_images_per_prompt,
            output_from_layers=self._output_from_layers,
            # Zeroing the padding positions is this port's choice for FLUX.2-dev; diffusers zeroes them for neither
            # model, and Klein follows it.
            zero_padding=not self._is_qwen3,
            encoder=self._encoder,
            device=self._device,
            traced=traced,
        )

    def upsample(
        self, prompts: Sequence[str], *, max_length: int, temperature: float, traced: bool = False
    ) -> list[str]:
        if self._is_qwen3:
            msg = "prompt upsampling is only supported for Mistral 3"
            raise ValueError(msg)

        return _upsample_prompts(
            prompts,
            max_length=max_length,
            temperature=temperature,
            tokenizer=self._tokenizer,
            encoder=self._encoder,
            device=self._device,
            traced=traced,
        )


def _load_torch_mistral3_encoder(checkpoint_name: str) -> transformers.Mistral3ForConditionalGeneration:
    model = transformers.Mistral3ForConditionalGeneration.from_pretrained(checkpoint_name, subfolder="text_encoder")
    model.eval()
    return model


def _load_torch_qwen3_encoder(checkpoint_name: str) -> transformers.Qwen3ForCausalLM:
    model = transformers.Qwen3ForCausalLM.from_pretrained(
        checkpoint_name, subfolder="text_encoder", dtype=torch.bfloat16
    )
    model.eval()
    return model


def _tokenize_mistral3(
    prompts: Sequence[str], *, tokenizer: PreTrainedTokenizerBase, sequence_length: int
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

    return tokens, mask


def _tokenize_qwen3(
    prompts: Sequence[str], *, tokenizer: PreTrainedTokenizerBase, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    tokenizer_out = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True, max_length=sequence_length
    )

    if any(len(ids) > sequence_length for ids in tokenizer(texts).input_ids):
        logger.warning("input text was truncated")

    return tokenizer_out.input_ids, tokenizer_out.attention_mask


def _get_prompt_embeds(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    encoder: Module | torch.nn.Module,
    num_images_per_prompt: int,
    output_from_layers: Sequence[int],
    zero_padding: bool,
    device: ttnn.MeshDevice | None,
    traced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(encoder, Module):
        assert device is not None

        tt_tokens = tensor.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_mask = tensor.from_torch(mask, device=device)

        # only the requested layers' states, in layer order
        tt_hidden_states = encoder.forward(
            tt_tokens,
            mask=tt_mask,
            skip_final_linear=True,
            output_hidden_states=output_from_layers,
            traced=traced,
        )
        tt_prompt_embeds = ttnn.concat(tt_hidden_states, dim=-1)

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=encoder.device)

        with torch.no_grad():
            output = encoder.forward(
                tokens,
                attention_mask=mask,
                output_hidden_states=True,
            )
        prompt_embeds = torch.concat([output.hidden_states[k] for k in output_from_layers], dim=-1).to("cpu")

    if zero_padding:
        prompt_embeds.masked_fill_(~mask.unsqueeze(-1).bool(), 0.0)

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
    traced: bool = False,
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

        output_tokens = encoder.generate(
            tokens,
            mask=mask,
            eos_tokens=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature,
            traced=traced,
        ).tokens
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
