# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from transformers import PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from ...encoders.qwen25vl.model_qwen25vl import Qwen25VlTextEncoder
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


class Qwen25VlTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        tokenizer_subfolder: str | None = None,
        encoder_subfolder: str | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
        is_fsdp: bool = False,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._checkpoint = checkpoint
        self._encoder_subfolder = encoder_subfolder
        self._use_torch = use_torch
        self._encoder_loaded = True
        self._is_fsdp = is_fsdp

        if tokenizer_subfolder is not None:
            self._tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint, subfolder=tokenizer_subfolder)
        else:
            self._tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint)
        self._encoder = self._load_encoder(checkpoint, encoder_subfolder, use_torch=use_torch)

    def _load_encoder(
        self, checkpoint: str, subfolder: str | None, *, use_torch: bool
    ) -> Qwen2_5_VLForConditionalGeneration | Qwen25VlTextEncoder:
        # Only pass subfolder if it's not None
        if subfolder is not None:
            torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, subfolder=subfolder)
        else:
            torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)

        if use_torch:
            return torch_model

        model = Qwen25VlTextEncoder(
            vocab_size=torch_model.config.vocab_size,
            hidden_size=torch_model.config.hidden_size,
            intermediate_size=torch_model.config.intermediate_size,
            hidden_act=torch_model.config.hidden_act,
            num_hidden_layers=torch_model.config.num_hidden_layers,
            num_attention_heads=torch_model.config.num_attention_heads,
            num_key_value_heads=torch_model.config.num_key_value_heads,
            rms_norm_eps=torch_model.config.rms_norm_eps,
            rope_theta=torch_model.config.rope_theta,
            mrope_section=torch_model.config.rope_scaling["mrope_section"],
            device=self._device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
            is_fsdp=self._is_fsdp,
        )

        torch_text_model = torch_model.model.language_model

        # Get the state dict before deleting the torch model
        torch_state_dict = torch_text_model.state_dict()

        # Delete the torch model to free up memory before loading weights to device
        del torch_model
        del torch_text_model

        # Store state dict for potential reloading
        self._encoder_state_dict = torch_state_dict

        if not cache.initialize_from_cache(
            tt_model=model,
            torch_state_dict=torch_state_dict,
            model_name=checkpoint,
            subfolder=subfolder if subfolder is not None else "",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            dtype="bf16",
            is_fsdp=self._is_fsdp,
        ):
            logger.info("loading encoder from torch state...")
            model.load_torch_state_dict(torch_state_dict)

        return model

    def encoder_loaded(self) -> bool:
        return self._use_torch or self._encoder.is_loaded()

    def reload_encoder_weights(self) -> None:
        """Reload encoder weights to device after deallocation."""
        if self._use_torch or self._encoder_loaded:
            return

        logger.info("reloading encoder weights to device...")
        if not cache.initialize_from_cache(
            tt_model=self._encoder,
            torch_state_dict=self._encoder_state_dict,
            model_name=self._checkpoint,
            subfolder=self._encoder_subfolder if self._encoder_subfolder is not None else "",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            dtype="bf16",
            is_fsdp=self._is_fsdp,
        ):
            self._encoder.load_torch_state_dict(self._encoder_state_dict)

        self._encoder_loaded = True
        ttnn.synchronize_device(self._device)

    def deallocate_encoder_weights(self) -> None:
        """Deallocate encoder weights from device."""
        if self._use_torch or not self._encoder_loaded:
            return

        self._encoder.deallocate_weights()
        self._encoder_loaded = False
        ttnn.synchronize_device(self._device)

    def encode(
        self, prompts: Sequence[str], *, num_images_per_prompt: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_qwen_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            sequence_length=sequence_length,
            mesh_device=self._device,
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L188
def _get_qwen_prompt_embeds(
    prompts: Sequence[str],
    text_encoder: Qwen25VlTextEncoder | Qwen2_5_VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    mesh_device: ttnn.MeshDevice | None,
    sequence_length: int,
    num_images_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    )

    tokens = tokenizer_out.input_ids
    attention_mask = tokenizer_out.attention_mask

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("input text was truncated")

    if isinstance(text_encoder, Qwen25VlTextEncoder):
        assert mesh_device is not None

        cos, sin = text_encoder.create_rope_tensors(tokens.shape[0], tokens.shape[1], attention_mask)

        tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
        tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device)
        tt_cos = tensor.from_torch(cos, device=mesh_device)
        tt_sin = tensor.from_torch(sin, device=mesh_device)

        tt_hidden_states = text_encoder.forward(
            tt_tokens, attention_mask=tt_attention_mask, pos_embeds=(tt_cos, tt_sin)
        )
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)

        with torch.no_grad():
            output = text_encoder.forward(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        prompt_embeds = output.hidden_states[-1].to("cpu")

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, attention_mask
