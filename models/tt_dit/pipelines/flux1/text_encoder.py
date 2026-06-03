# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import ttnn
from models.tt_dit.encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from models.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import cache
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


T5_SEQUENCE_LENGTH = 512


class TextEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        enable_t5: bool,
        joint_attention_dim: int,
        use_torch_clip: bool,
        use_torch_t5: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._joint_attention_dim = joint_attention_dim

        logger.info("loading torch text encoders...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._t5_tokenizer = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_2")
        torch_text_encoder_1 = CLIPTextModel.from_pretrained(checkpoint_name, subfolder="text_encoder")
        torch_text_encoder_1.eval()
        torch_t5_text_encoder = (
            T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_2") if enable_t5 else None
        )

        model_name = os.path.basename(checkpoint_name)

        if use_torch_clip:
            self._text_encoder_1 = torch_text_encoder_1
            self._clip_tracer = None
        else:
            logger.info("creating TT-NN CLIP text encoder...")
            clip_config_1 = CLIPConfig(
                vocab_size=torch_text_encoder_1.config.vocab_size,
                embed_dim=torch_text_encoder_1.config.hidden_size,
                ff_dim=torch_text_encoder_1.config.intermediate_size,
                num_heads=torch_text_encoder_1.config.num_attention_heads,
                num_hidden_layers=torch_text_encoder_1.config.num_hidden_layers,
                max_prompt_length=77,
                layer_norm_eps=torch_text_encoder_1.config.layer_norm_eps,
                attention_dropout=torch_text_encoder_1.config.attention_dropout,
                hidden_act=torch_text_encoder_1.config.hidden_act,
            )
            self._text_encoder_1 = CLIPEncoder(
                config=clip_config_1,
                mesh_device=device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                eos_token_id=2,
            )
            self._text_encoder_1.load_torch_state_dict(torch_text_encoder_1.state_dict())
            self._clip_tracer = Tracer(self._text_encoder_1.forward, device=device, prep_run=False)

        if enable_t5:
            if use_torch_t5:
                self._t5_text_encoder = torch_t5_text_encoder
                self._t5_tracer = None
            else:
                logger.info("creating TT-NN T5 text encoder...")
                t5_config = T5Config(
                    vocab_size=torch_t5_text_encoder.config.vocab_size,
                    embed_dim=torch_t5_text_encoder.config.d_model,
                    ff_dim=torch_t5_text_encoder.config.d_ff,
                    kv_dim=torch_t5_text_encoder.config.d_kv,
                    num_heads=torch_t5_text_encoder.config.num_heads,
                    num_hidden_layers=torch_t5_text_encoder.config.num_layers,
                    max_prompt_length=T5_SEQUENCE_LENGTH,
                    layer_norm_eps=torch_t5_text_encoder.config.layer_norm_epsilon,
                    relative_attention_num_buckets=torch_t5_text_encoder.config.relative_attention_num_buckets,
                    relative_attention_max_distance=torch_t5_text_encoder.config.relative_attention_max_distance,
                )
                self._t5_text_encoder = T5Encoder(
                    config=t5_config,
                    mesh_device=device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                )
                cache.load_model(
                    self._t5_text_encoder,
                    get_torch_state_dict=torch_t5_text_encoder.state_dict,
                    model_name=model_name,
                    subfolder="t5_text_encoder",
                    parallel_config=parallel_config,
                    mesh_shape=tuple(device.shape),
                )
                self._t5_tracer = Tracer(self._t5_text_encoder.forward, device=device, prep_run=False)
        else:
            self._t5_text_encoder = None
            self._t5_tracer = None

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: tuple[Sequence[str], Sequence[str]],
        neg_prompts: tuple[Sequence[str], Sequence[str]] | None,
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        clip_skip: int = 0,
        traced: bool = False,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, pooled_prompt_embeds = self._encode(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            traced=traced,
            on_event=on_event,
        )

        if not cfg_enabled:
            return prompt_embeds, pooled_prompt_embeds

        assert neg_prompts is not None
        negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode(
            neg_prompts,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            traced=traced,
            on_event=on_event,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        return prompt_embeds, pooled_prompt_embeds

    def _encode(
        self,
        prompts: tuple[Sequence[str], Sequence[str]],
        *,
        num_images_per_prompt: int,
        clip_skip: int,
        traced: bool,
        on_event: PipelineEventCallback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompts_1, prompts_2 = prompts
        tokenizer_max_length = self._tokenizer_1.model_max_length

        on_event(SectionStart("clip_encoding"))
        _, pooled_prompt_embeds = _get_clip_prompt_embeds(
            prompts=prompts_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._text_encoder_1,
            tracer=self._clip_tracer if traced else None,
            sequence_length=tokenizer_max_length,
            mesh_device=self._device,
            clip_skip=clip_skip,
        )
        on_event(SectionEnd("clip_encoding"))

        on_event(SectionStart("t5_encoding"))
        prompt_embeds = _get_t5_prompt_embeds(
            prompts=prompts_2,
            text_encoder=self._t5_text_encoder,
            tracer=self._t5_tracer if traced else None,
            tokenizer=self._t5_tokenizer,
            sequence_length=T5_SEQUENCE_LENGTH,
            empty_sequence_length=T5_SEQUENCE_LENGTH,
            num_images_per_prompt=num_images_per_prompt,
            mesh_device=self._device,
            embedding_dim=self._joint_attention_dim,
        )
        on_event(SectionEnd("t5_encoding"))

        return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    prompts: Sequence[str],
    text_encoder: CLIPEncoder | CLIPTextModel,
    tracer: Tracer | None = None,
    tokenizer: CLIPTokenizer,
    sequence_length: int,
    num_images_per_prompt: int,
    clip_skip: int = 0,
    mesh_device: ttnn.MeshDevice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    ).input_ids

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("CLIP input text was truncated")

    if isinstance(text_encoder, CLIPEncoder):
        assert mesh_device is not None

        tt_tokens = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )

        tt_prompt_embeds, tt_normalized = (tracer or text_encoder.forward)(
            prompt_tokenized=tt_tokens,
            skip_pooling=True,
        )
        tt_pooled_prompt_embeds = text_encoder.pooled_output(
            prompt_tokenized=tt_tokens,
            normalized_final_state=tt_normalized,
        )
        tt_prompt_embeds = tt_prompt_embeds[-(clip_skip + 2)]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, output_hidden_states=True)
        prompt_embeds = output.hidden_states[-(clip_skip + 2)].to("cpu")
        pooled_prompt_embeds = output.pooler_output.to("cpu")

    # In diffusers v0.35.1 `pooled_prompt_embeds` is repeated along the wrong dimension in
    # `StableDiffusion3Pipeline`, effectively mixing up the prompts.
    pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    *,
    prompts: Sequence[str],
    text_encoder: T5Encoder | T5EncoderModel | None,
    tracer: Tracer | None = None,
    tokenizer: T5TokenizerFast,
    sequence_length: int,
    empty_sequence_length: int,
    num_images_per_prompt: int,
    mesh_device: ttnn.MeshDevice | None = None,
    embedding_dim: int,
) -> torch.Tensor:
    if text_encoder is None:
        return torch.zeros([len(prompts) * num_images_per_prompt, empty_sequence_length, embedding_dim])

    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    ).input_ids

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
        tt_hidden_states = (tracer or text_encoder.forward)(prompt=tt_tokens)
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens)
        prompt_embeds = output.last_hidden_state.to("cpu")

    return prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
