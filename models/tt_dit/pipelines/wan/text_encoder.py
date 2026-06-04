# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import os
from typing import TYPE_CHECKING

import ftfy
import regex as re
from transformers import AutoTokenizer, UMT5EncoderModel

import ttnn
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import cache

if TYPE_CHECKING:
    from collections.abc import Sequence


def _basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text: str) -> str:
    return _whitespace_clean(_basic_clean(text))


class TextEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        encoder_parallel_config: EncoderParallelConfig,
        dit_parallel_config: DiTParallelConfig,
        max_sequence_length: int = 512,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._encoder_parallel_config = encoder_parallel_config
        self._dit_parallel_config = dit_parallel_config
        self._checkpoint_name = checkpoint_name
        self._max_sequence_length = max_sequence_length

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer", trust_remote_code=True)
        self._torch_text_encoder = UMT5EncoderModel.from_pretrained(
            checkpoint_name, subfolder="text_encoder", trust_remote_code=True
        )

        umt5_config = UMT5Config(
            vocab_size=self._torch_text_encoder.config.vocab_size,
            embed_dim=self._torch_text_encoder.config.d_model,
            ff_dim=self._torch_text_encoder.config.d_ff,
            kv_dim=self._torch_text_encoder.config.d_kv,
            num_heads=self._torch_text_encoder.config.num_heads,
            num_hidden_layers=self._torch_text_encoder.config.num_layers,
            max_prompt_length=512,  # TODO: Consider removing
            layer_norm_eps=self._torch_text_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=self._torch_text_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=self._torch_text_encoder.config.relative_attention_max_distance,
        )

        self._tt_encoder = UMT5Encoder(
            config=umt5_config,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=encoder_parallel_config,
        )

    def prepare(self) -> None:
        cache.load_model(
            self._tt_encoder,
            model_name=os.path.basename(self._checkpoint_name),
            subfolder="text_encoder",
            parallel_config=self._encoder_parallel_config,
            mesh_shape=tuple(self._device.shape),
            get_torch_state_dict=lambda: self._torch_text_encoder.state_dict(),
        )

    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        cfg_enabled: bool,
        num_videos_per_prompt: int = 1,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[object, object]:
        on_event(SectionStart("t5_encoding"))

        prompts_list = list(prompts)
        batch_size = len(prompts_list)

        all_input_prompts: list[str] = list(prompts_list)
        pos_prompt_end_idx = batch_size * num_videos_per_prompt
        neg_prompt_end_idx = pos_prompt_end_idx

        if cfg_enabled:
            negative_prompts_list = list(negative_prompts)
            assert batch_size == len(
                negative_prompts_list
            ), f"`negative_prompt` has batch size {len(negative_prompts_list)}, but `prompt` has batch size {batch_size}."
            all_input_prompts += negative_prompts_list
            neg_prompt_end_idx = pos_prompt_end_idx + batch_size * num_videos_per_prompt

        # Pad batch list of prompts to ensure proper sharding on batch dimension.
        total_prompts = len(all_input_prompts)
        num_devices = self._device.shape[1 - self._dit_parallel_config.tensor_parallel.mesh_axis]
        all_input_prompts += [" "] * ((num_devices - (total_prompts % num_devices)) % num_devices)

        all_prompt_embeds = self._encode(
            all_input_prompts,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        prompt_embeds = all_prompt_embeds[:, :pos_prompt_end_idx]
        negative_prompt_embeds = all_prompt_embeds[:, pos_prompt_end_idx:neg_prompt_end_idx] if cfg_enabled else None

        on_event(SectionEnd("t5_encoding"))
        return prompt_embeds, negative_prompt_embeds

    def _encode(
        self,
        prompts: list[str],
        *,
        num_videos_per_prompt: int,
    ) -> object:
        prompts = [prompt_clean(u) for u in prompts]
        batch_size = len(prompts)

        # NOTE: while the reference impl does not pad to max_sequence_length, for some reason this
        # seems to be necessary for correctness in this pipeline. TODO: investigate
        text_inputs = self._tokenizer(
            prompts,
            padding="max_length",
            max_length=self._max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        # Shard on batch dimension. On non TP axis
        dims = [None, None]
        DP_axis = 1 - self._dit_parallel_config.tensor_parallel.mesh_axis
        dims[DP_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(self._device, mesh_shape=tuple(self._device.shape), dims=dims)
        tt_prompt = ttnn.from_torch(
            text_input_ids,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            mesh_mapper=mesh_mapper,
        )
        tt_mask = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            mesh_mapper=mesh_mapper,
        )

        prompt_embeds = self._tt_encoder(tt_prompt, attention_mask=tt_mask)[-1]

        # use the mask to zero out the padding tokens.
        prompt_embeds = prompt_embeds * ttnn.unsqueeze(tt_mask, -1)

        prompt_embeds = self._ccl_manager.all_gather(prompt_embeds, dim=0, mesh_axis=DP_axis, use_hyperparams=True)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = ttnn.repeat(prompt_embeds, (1, num_videos_per_prompt, 1))
        return ttnn.view(prompt_embeds, (1, batch_size * num_videos_per_prompt, seq_len, -1))
