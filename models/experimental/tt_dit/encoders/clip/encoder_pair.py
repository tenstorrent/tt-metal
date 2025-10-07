# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from transformers import AutoTokenizer, CLIPTextModel

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from .model_clip import CLIPConfig, CLIPEncoder

if TYPE_CHECKING:
    from collections.abc import Iterable


class CLIPTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        sequence_length: int | None,
        skip_norm: bool,
        true_clip_skip: int,
        zero_masking: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._encoder = self._load_encoder(checkpoint, use_torch=use_torch)

        self._sequence_length = sequence_length if sequence_length is not None else self._tokenizer.model_max_length
        self._skip_norm = skip_norm
        self._true_clip_skip = true_clip_skip
        self._zero_masking = zero_masking

    def _load_encoder(self, checkpoint: str, *, use_torch: bool) -> CLIPEncoder | CLIPTextModel:
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

    def encode(
        self, prompts: Iterable[str], *, num_images_per_prompt: int, true_clip_skip: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_clip_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            sequence_length=self._sequence_length,
            skip_norm=self._skip_norm,
            true_clip_skip=true_clip_skip if true_clip_skip is not None else self._true_clip_skip,
            zero_masking=self._zero_masking,
            mesh_device=self._device,
        )


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
            return_normalized_state=True,
        )
        tt_prompt_embeds = tt_prompt_embeds[-(true_clip_skip + 1)] if skip_norm else tt_normalized

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, output_hidden_states=True)
        prompt_embeds = output.hidden_states[-(true_clip_skip + 1)] if skip_norm else output.last_hidden_state
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
