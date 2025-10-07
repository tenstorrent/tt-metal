# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5EncoderModel

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from .model_t5 import T5Config, T5Encoder

if TYPE_CHECKING:
    from collections.abc import Iterable


class T5TokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        sequence_length: int | None,
        empty_sequence_length: int | None,
        embedding_dim: int | None,
        zero_masking: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_attention_mask: bool,
        use_torch: bool,
        enabled: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self._sequence_length = sequence_length if sequence_length is not None else self._tokenizer.model_max_length
        self._empty_sequence_length = (
            empty_sequence_length if empty_sequence_length is not None else self._sequence_length
        )
        self._embedding_dim = embedding_dim
        self._zero_masking = zero_masking
        self._use_attention_mask = use_attention_mask

        self._encoder = self._load_encoder(checkpoint, use_torch=use_torch) if enabled else None

    def _load_encoder(self, checkpoint: str, *, use_torch: bool) -> T5Encoder | T5EncoderModel:
        torch_model = T5EncoderModel.from_pretrained(checkpoint)

        if use_torch:
            return torch_model

        config = T5Config(
            vocab_size=torch_model.config.vocab_size,
            embed_dim=torch_model.config.d_model,
            ff_dim=torch_model.config.d_ff,
            kv_dim=torch_model.config.d_kv,
            num_heads=torch_model.config.num_heads,
            num_hidden_layers=torch_model.config.num_layers,
            max_prompt_length=self._sequence_length,
            layer_norm_eps=torch_model.config.layer_norm_epsilon,
            relative_attention_num_buckets=torch_model.config.relative_attention_num_buckets,
            relative_attention_max_distance=torch_model.config.relative_attention_max_distance,
        )

        model = T5Encoder(
            config=config,
            mesh_device=self._device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
        )

        if cache.cache_dir_is_set():
            cache_path = cache.get_and_create_cache_path(
                model_name=checkpoint,
                subfolder="",
                parallel_config=self._parallel_config,
                mesh_shape=self._device.shape,
                dtype="bf16",
            )
            if cache.cache_dict_exists(cache_path):
                logger.info("loading T5 encoder from cache...")
                model.from_cached_state_dict(cache.load_cache_dict(cache_path))
            else:
                logger.info("loading T5 encoder from torch state...")
                model.load_state_dict(torch_model.state_dict())
                logger.info("saving T5 encoder to cache...")
                cache.save_cache_dict(model.to_cached_state_dict(cache_path), cache_path)
        else:
            logger.info("loading T5 encoder from torch state...")
            model.load_state_dict(torch_model.state_dict())

        return model

    def encode(self, prompts: Iterable[str], *, num_images_per_prompt: int) -> torch.Tensor:
        return _get_t5_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            sequence_length=self._sequence_length,
            empty_sequence_length=self._empty_sequence_length,
            embedding_dim=self._embedding_dim,
            zero_masking=self._zero_masking,
            use_attention_mask=self._use_attention_mask,
            mesh_device=self._device,
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    *,
    prompts: Iterable[str],
    text_encoder: T5Encoder | T5EncoderModel | None,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    empty_sequence_length: int,
    num_images_per_prompt: int,
    mesh_device: ttnn.MeshDevice | None,
    embedding_dim: int | None,
    zero_masking: bool,
    use_attention_mask: bool,
) -> torch.Tensor:
    prompts = list(prompts)

    if text_encoder is None:
        assert embedding_dim is not None
        return torch.zeros([len(prompts) * num_images_per_prompt, empty_sequence_length, embedding_dim])

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    )

    tokens = tokenizer_out.input_ids
    attention_mask = tokenizer_out.attention_mask if use_attention_mask else None

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("T5 input text was truncated")

    if isinstance(text_encoder, T5Encoder):
        assert mesh_device is not None

        tt_tokens = ttnn.from_torch(
            tokens,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            device=mesh_device,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        tt_attention_mask = (
            ttnn.from_torch(
                attention_mask[:, None, None, :],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
            if attention_mask is not None
            else None
        )
        tt_hidden_states = text_encoder(prompt=tt_tokens, attention_mask=tt_attention_mask, device=mesh_device)
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, attention_mask=attention_mask)
        prompt_embeds = output.last_hidden_state.to("cpu")

    if zero_masking:
        prompt_embeds = prompt_embeds * (tokens != tokenizer.pad_token_id).unsqueeze(-1)

    if embedding_dim is not None:
        assert prompt_embeds.shape[-1] == embedding_dim

    return prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
