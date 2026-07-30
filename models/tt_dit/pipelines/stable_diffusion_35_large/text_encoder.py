# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import ttnn
from models.tt_dit.encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from models.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tracing import Tracer


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
        max_t5_sequence_length: int = 256,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._joint_attention_dim = joint_attention_dim
        self._max_t5_sequence_length = max_t5_sequence_length

        logger.info("loading torch text encoders...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer_2")
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_3")
        torch_text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(checkpoint_name, subfolder="text_encoder")
        torch_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(checkpoint_name, subfolder="text_encoder_2")
        torch_text_encoder_3 = (
            T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_3") if enable_t5 else None
        )

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
            projection_dim=torch_text_encoder_1.config.projection_dim,
        )
        clip_config_2 = CLIPConfig(
            vocab_size=torch_text_encoder_2.config.vocab_size,
            embed_dim=torch_text_encoder_2.config.hidden_size,
            ff_dim=torch_text_encoder_2.config.intermediate_size,
            num_heads=torch_text_encoder_2.config.num_attention_heads,
            num_hidden_layers=torch_text_encoder_2.config.num_hidden_layers,
            max_prompt_length=77,
            layer_norm_eps=torch_text_encoder_2.config.layer_norm_eps,
            attention_dropout=torch_text_encoder_2.config.attention_dropout,
            hidden_act=torch_text_encoder_2.config.hidden_act,
            projection_dim=torch_text_encoder_2.config.projection_dim,
        )

        logger.info("creating TT-NN CLIP text encoders...")
        self._text_encoder_1 = CLIPEncoder(
            config=clip_config_1,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            eos_token_id=2,
        )
        self._text_encoder_1.load_torch_state_dict(torch_text_encoder_1.state_dict())

        self._text_encoder_2 = CLIPEncoder(
            config=clip_config_2,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            eos_token_id=2,
        )
        self._text_encoder_2.load_torch_state_dict(torch_text_encoder_2.state_dict())

        if enable_t5:
            t5_config = T5Config(
                vocab_size=torch_text_encoder_3.config.vocab_size,
                embed_dim=torch_text_encoder_3.config.d_model,
                ff_dim=torch_text_encoder_3.config.d_ff,
                kv_dim=torch_text_encoder_3.config.d_kv,
                num_heads=torch_text_encoder_3.config.num_heads,
                num_hidden_layers=torch_text_encoder_3.config.num_layers,
                max_prompt_length=256,
                layer_norm_eps=torch_text_encoder_3.config.layer_norm_epsilon,
                relative_attention_num_buckets=torch_text_encoder_3.config.relative_attention_num_buckets,
                relative_attention_max_distance=torch_text_encoder_3.config.relative_attention_max_distance,
            )
            logger.info("creating TT-NN T5 text encoder...")
            self._text_encoder_3 = T5Encoder(
                config=t5_config,
                mesh_device=device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            self._text_encoder_3.load_torch_state_dict(torch_text_encoder_3.state_dict())
            self._t5_tracer = Tracer(self._text_encoder_3.forward, device=device, prep_run=False)
        else:
            self._text_encoder_3 = None
            self._t5_tracer = None

        self._clip_tracer_1 = Tracer(self._text_encoder_1.forward, device=device, prep_run=False)
        self._clip_tracer_2 = Tracer(self._text_encoder_2.forward, device=device, prep_run=False)

    @property
    def t5_enabled(self) -> bool:
        return self._text_encoder_3 is not None

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: tuple[Sequence[str], Sequence[str], Sequence[str]],
        neg_prompts: tuple[Sequence[str], Sequence[str], Sequence[str]] | None,
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        clip_skip: int | None = None,
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
        prompts: tuple[Sequence[str], Sequence[str], Sequence[str]],
        *,
        num_images_per_prompt: int,
        clip_skip: int | None,
        traced: bool,
        on_event: PipelineEventCallback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompts_1, prompts_2, prompts_3 = prompts
        tokenizer_max_length = self._tokenizer_1.model_max_length

        on_event(SectionStart("clip_encoding"))
        prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=prompts_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._text_encoder_1,
            tracer=self._clip_tracer_1 if traced else None,
            tokenizer_max_length=tokenizer_max_length,
            ttnn_device=self._device,
            clip_skip=clip_skip,
        )
        prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=prompts_2,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_2,
            text_encoder=self._text_encoder_2,
            tracer=self._clip_tracer_2 if traced else None,
            tokenizer_max_length=tokenizer_max_length,
            ttnn_device=self._device,
            clip_skip=clip_skip,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        on_event(SectionEnd("clip_encoding"))

        on_event(SectionStart("t5_encoding"))
        t5_prompt_embed = _get_t5_prompt_embeds(
            device=self._device,
            prompt=prompts_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=self._max_t5_sequence_length,
            tokenizer=self._tokenizer_3,
            text_encoder=self._text_encoder_3,
            tracer=self._t5_tracer if traced else None,
            tokenizer_max_length=tokenizer_max_length,
            joint_attention_dim=self._joint_attention_dim,
        )
        on_event(SectionEnd("t5_encoding"))

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    clip_skip: int | None = None,
    ttnn_device: ttnn.MeshDevice,
    num_images_per_prompt: int,
    prompt: Sequence[str],
    text_encoder: CLIPEncoder,
    tracer: Tracer | None = None,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )

    tt_text_input_ids = tensor.from_torch(text_input_ids, dtype=ttnn.uint32, device=ttnn_device)

    encoder_output, normalized_output = (tracer or text_encoder.forward)(
        prompt_tokenized=tt_text_input_ids,
        skip_pooling=True,
    )
    pooled_output = text_encoder.pooled_output(
        prompt_tokenized=tt_text_input_ids,
        normalized_final_state=normalized_output,
    )

    if clip_skip is None:
        sequence_embeddings = encoder_output[-2]
    else:
        layer_index = -(clip_skip + 2)
        if abs(layer_index) > len(encoder_output):
            layer_index = -2
        sequence_embeddings = encoder_output[layer_index]

    prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(sequence_embeddings)[0])
    pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(pooled_output)[0])

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: Sequence[str],
    *,
    device: ttnn.MeshDevice,
    joint_attention_dim: int,
    max_sequence_length: int,
    num_images_per_prompt: int,
    text_encoder: T5Encoder | None,
    tracer: Tracer | None = None,
    tokenizer_max_length: int,
    tokenizer: T5TokenizerFast,
) -> torch.Tensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None:
        return torch.zeros(
            (batch_size * num_images_per_prompt, tokenizer_max_length, joint_attention_dim),
            dtype=torch.bfloat16,
        )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    tt_text_input_ids = tensor.from_torch(text_input_ids, device=device)
    hidden_states = (tracer or text_encoder.forward)(prompt=tt_text_input_ids)
    tt_prompt_embeds = hidden_states[-1]
    tt_prompt_embeds = ttnn.get_device_tensors(tt_prompt_embeds)[0]
    prompt_embeds = ttnn.to_torch(tt_prompt_embeds)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
