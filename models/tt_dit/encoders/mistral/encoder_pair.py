# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import AutoProcessor

import ttnn

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .model_mistral import MistralTextEncoder

if TYPE_CHECKING:
    pass


HIDDEN_STATE_LAYERS_HF = (10, 20, 30)
HIDDEN_STATE_LAYERS_INTERNAL = tuple(k - 1 for k in HIDDEN_STATE_LAYERS_HF)
NUM_ENCODER_LAYERS = max(HIDDEN_STATE_LAYERS_HF)

SYSTEM_MESSAGE = (
    "You are an image generation model. You will be given a prompt and you will "
    "generate an image based on that prompt."
)

MISTRAL_SEQUENCE_LENGTH = 512


class MistralTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool = False,
        is_fsdp: bool = False,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._checkpoint = checkpoint
        self._subfolder = subfolder
        self._use_torch = use_torch
        self._is_fsdp = is_fsdp
        self._encoder_loaded = True

        self._tokenizer = AutoProcessor.from_pretrained(checkpoint, subfolder=tokenizer_subfolder)
        self._encoder = self._load_encoder(checkpoint, subfolder, use_torch=use_torch)

    def _load_encoder(self, checkpoint: str, subfolder: str, *, use_torch: bool) -> MistralTextEncoder:
        from transformers import Mistral3ForConditionalGeneration

        logger.info("loading Mistral3 from checkpoint...")
        torch_model = Mistral3ForConditionalGeneration.from_pretrained(
            checkpoint,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        )
        torch_model.eval()

        if use_torch:
            return torch_model

        text_config = torch_model.config.text_config

        model = MistralTextEncoder(
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_hidden_layers=NUM_ENCODER_LAYERS,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            rope_theta=getattr(text_config, "rope_theta", None) or text_config.rope_parameters["rope_theta"],
            hidden_state_layers=HIDDEN_STATE_LAYERS_INTERNAL,
            device=self._device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
            is_fsdp=self._is_fsdp,
        )

        torch_text_model = torch_model.model.language_model
        torch_state_dict = torch_text_model.state_dict()

        keys_to_remove = [
            k for k in torch_state_dict if k.startswith("layers.") and int(k.split(".")[1]) >= NUM_ENCODER_LAYERS
        ]
        for k in keys_to_remove:
            del torch_state_dict[k]

        del torch_model
        del torch_text_model

        self._encoder_state_dict = torch_state_dict

        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: torch_state_dict,
            model_name=checkpoint,
            subfolder="mistral_text_encoder",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            is_fsdp=self._is_fsdp,
        )

        return model

    def encoder_loaded(self) -> bool:
        return self._use_torch or self._encoder.is_loaded()

    def reload_encoder_weights(self) -> None:
        if self._use_torch or self._encoder_loaded:
            return

        logger.info("reloading Mistral3 encoder weights to device...")
        cache.load_model(
            tt_model=self._encoder,
            get_torch_state_dict=lambda: self._encoder_state_dict,
            model_name=self._checkpoint,
            subfolder="mistral_text_encoder",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            is_fsdp=self._is_fsdp,
        )
        self._encoder_loaded = True
        ttnn.synchronize_device(self._device)

    def deallocate_encoder_weights(self) -> None:
        if self._use_torch or not self._encoder_loaded:
            return

        self._encoder.deallocate_weights()
        self._encoder_loaded = False
        ttnn.synchronize_device(self._device)

    def encode(self, prompts: list[str]) -> torch.Tensor:
        return _get_mistral_prompt_embeds(
            prompts=prompts,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            mesh_device=self._device,
        )


def _format_input(prompts: list[str], system_message: str) -> list[list[dict]]:
    messages = []
    for prompt in prompts:
        messages.append(
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )
    return messages


def _get_mistral_prompt_embeds(
    *,
    prompts: list[str],
    tokenizer: AutoProcessor,
    text_encoder: MistralTextEncoder,
    mesh_device: ttnn.MeshDevice,
) -> torch.Tensor:
    messages = _format_input(prompts, system_message=SYSTEM_MESSAGE)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MISTRAL_SEQUENCE_LENGTH,
    )

    tokens = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if isinstance(text_encoder, MistralTextEncoder):
        assert mesh_device is not None

        cos, sin = text_encoder.create_rope_tensors(tokens.shape[0], tokens.shape[1], attention_mask)

        tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
        tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device)
        tt_cos = tensor.from_torch(cos, device=mesh_device)
        tt_sin = tensor.from_torch(sin, device=mesh_device)

        tt_hidden_states = text_encoder.forward(
            tt_tokens, attention_mask=tt_attention_mask, pos_embeds=(tt_cos, tt_sin)
        )

        hidden_states_torch = []
        for hs in tt_hidden_states:
            hidden_states_torch.append(ttnn.to_torch(ttnn.get_device_tensors(hs)[0]).to(torch.bfloat16))

        out = torch.stack(hidden_states_torch, dim=1)
    else:
        device = next(text_encoder.parameters()).device
        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
        out = torch.stack(
            [output.hidden_states[k] for k in HIDDEN_STATE_LAYERS_HF],
            dim=1,
        )

    out = out.to(dtype=torch.bfloat16, device="cpu")
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds
