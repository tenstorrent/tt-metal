# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from transformers import AutoTokenizer, CLIPTextModel, PreTrainedTokenizerBase, T5EncoderModel

from .encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from .encoders.t5.model_t5 import T5Config, T5Encoder
from .parallel.config import EncoderParallelConfig, ParallelFactor
from .parallel.manager import CCLManager
from .utils import cache

if TYPE_CHECKING:
    from collections.abc import Iterable


class TextEncoder:
    T5_SEQUENCE_LENGTH = 256
    T5_EMPTY_SEQUENCE_LENGTH = 256
    T5_EMBEDDING_DIM = 4096
    CLIP_L_CHECKPOINT = "openai/clip-vit-large-patch14"
    CLIP_G_CHECKPOINT = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    T5_CHECKPOINT = "google/flan-t5-xxl"

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        enable_t5: bool,
        use_torch_clip_encoder: bool,
        use_torch_t5_encoder: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        # tokenizers

        self._tokenizer_clip_l = AutoTokenizer.from_pretrained(self.CLIP_L_CHECKPOINT)
        self._tokenizer_clip_g = AutoTokenizer.from_pretrained(self.CLIP_G_CHECKPOINT)
        self._tokenizer_t5 = AutoTokenizer.from_pretrained(self.T5_CHECKPOINT)

        # encoders

        self._encoder_clip_l = self._load_clip_encoder(self.CLIP_L_CHECKPOINT, use_torch=use_torch_clip_encoder)
        self._encoder_clip_g = self._load_clip_encoder(self.CLIP_G_CHECKPOINT, use_torch=use_torch_clip_encoder)
        self._encoder_t5 = (
            self._load_t5_encoder(self.T5_CHECKPOINT, use_torch=use_torch_t5_encoder) if enable_t5 else None
        )

    def _load_clip_encoder(self, checkpoint: str, *, use_torch: bool) -> CLIPEncoder | CLIPTextModel:
        torch_model = CLIPTextModel.from_pretrained(checkpoint)

        if use_torch:
            return torch_model

        logger.info("creating CLIP encoder on device...")

        config = CLIPConfig(
            vocab_size=torch_model.config.vocab_size,
            embed_dim=torch_model.config.hidden_size,
            ff_dim=torch_model.config.intermediate_size,
            num_heads=torch_model.config.num_attention_heads,
            num_hidden_layers=torch_model.config.num_hidden_layers,
            max_prompt_length=77,
            layer_norm_eps=torch_model.config.layer_norm_eps,
            attention_dropout=torch_model.config.attention_dropout,
            hidden_act=torch_model.config.hidden_act,
        )

        model = CLIPEncoder(
            config=config,
            mesh_device=self._device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
            eos_token_id=2,  # default EOS token ID for CLIP
        )

        model.load_state_dict(torch_model.state_dict())

        return model

    def _load_t5_encoder(self, checkpoint: str, *, use_torch: bool) -> T5Encoder | T5EncoderModel:
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
            max_prompt_length=self.T5_SEQUENCE_LENGTH,
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

    def encode(
        self,
        prompt_1: Iterable[str],
        prompt_2: Iterable[str],
        prompt_3: Iterable[str],
        *,
        num_images_per_prompt: int,
        clip_skip: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_l, pooled_clip_l = _get_clip_prompt_embeds(
            prompts=prompt_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_clip_l,
            text_encoder=self._encoder_clip_l,
            sequence_length=self._tokenizer_clip_l.model_max_length,
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            mesh_device=self._device,
        )
        clip_g, pooled_clip_g = _get_clip_prompt_embeds(
            prompts=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_clip_g,
            text_encoder=self._encoder_clip_g,
            sequence_length=self._tokenizer_clip_g.model_max_length,
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            mesh_device=self._device,
        )
        t5 = _get_t5_prompt_embeds(
            prompts=prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_t5,
            text_encoder=self._encoder_t5,
            sequence_length=self.T5_SEQUENCE_LENGTH,
            empty_sequence_length=self.T5_EMPTY_SEQUENCE_LENGTH,
            embedding_dim=self.T5_EMBEDDING_DIM,
            zero_masking=True,
            mesh_device=self._device,
        )

        clip = torch.cat([clip_l, clip_g], dim=-1)
        clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))

        embeds = torch.cat([clip, t5], dim=-2)
        pooled_embeds = torch.cat([pooled_clip_l, pooled_clip_g], dim=-1)

        return embeds, pooled_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    prompts: Iterable[str],
    text_encoder: CLIPEncoder | CLIPTextModel,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    num_images_per_prompt: int,
    skip_norm: bool,
    # clip_skip: Can only be nonzero if skip_norm is true. A value of zero means no clip skip. This
    # is different from the Stable Diffusion implementation, where a value of zero means skipping
    # one layer. See also the comment here:
    # https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L416
    true_clip_skip: int,
    mesh_device: ttnn.MeshDevice | None,
    zero_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not skip_norm:
        assert true_clip_skip == 0

    prompts = list(prompts)

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

        tt_prompt_embeds, tt_pooled_prompt_embeds, tt_normalized = text_encoder(
            prompt_tokenized=tt_tokens,
            mesh_device=mesh_device,
        )
        tt_prompt_embeds = tt_prompt_embeds[-(true_clip_skip + 1)] if skip_norm else tt_normalized

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, output_hidden_states=True)
        prompt_embeds = output.hidden_states[-(true_clip_skip + 1)] if skip_norm else output.last_hidden_states
        prompt_embeds = prompt_embeds.to("cpu")
        pooled_prompt_embeds = output.pooler_output.to("cpu")

    def masking_wo_first_eos(token: torch.Tensor, eos: int) -> torch.Tensor:
        idx = (token != eos).sum(dim=1)
        mask = token != eos
        arange = torch.arange(mask.size(0))
        mask[arange, idx] = True
        return mask.unsqueeze(-1)  # B x L x 1

    if zero_masking:
        prompt_embeds = prompt_embeds * masking_wo_first_eos(tokens, tokenizer.eos_token_id)

    # In diffusers v0.35.1 `pooled_prompt_embeds` is repeated along the wrong dimension in
    # `StableDiffusion3Pipeline`, effectively mixing up the prompts.
    pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, pooled_prompt_embeds


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
) -> torch.Tensor:
    prompts = list(prompts)

    if text_encoder is None:
        assert embedding_dim is not None
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
        tt_hidden_states = text_encoder(prompt=tt_tokens, device=mesh_device)
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens)
        prompt_embeds = output.last_hidden_state.to("cpu")

    if zero_masking:
        prompt_embeds = prompt_embeds * (tokens != tokenizer.pad_token_id).unsqueeze(-1)

    if embedding_dim is not None:
        assert prompt_embeds.shape[-1] == embedding_dim

    return prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    ccl_manager = CCLManager(mesh_device=device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=device.shape[1], mesh_axis=1))

    encoder = TextEncoder(
        device=device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        enable_t5=True,
        use_torch_clip_encoder=True,
        use_torch_t5_encoder=True,
    )

    prompts = ["hello"]
    embeds, pooled = encoder.encode(prompts, prompts, prompts, num_images_per_prompt=1)


if __name__ == "__main__":
    main()
