# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from transformers import PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

if TYPE_CHECKING:
    from collections.abc import Iterable


class Qwen25VlTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        max_batch_size: int,
        max_sequence_length: int,
        device: ttnn.MeshDevice,
        use_torch: bool,
    ) -> None:
        self._device = device

        self._tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint)
        self._torch_encoder, self._encoder = self._load_encoder(
            checkpoint,
            torch_only=use_torch,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
        )

    def _load_encoder(
        self,
        checkpoint: str,
        *,
        torch_only: bool,
        max_batch_size: int,
        max_sequence_length: int,
    ) -> tuple[Qwen2_5_VLForConditionalGeneration, Transformer | None]:
        if torch_only:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint), None

        logger.info("creating encoder on device...")

        os.environ["HF_MODEL"] = checkpoint
        model_args = ModelArgs(
            self._device,
            max_batch_size=max_batch_size,
            # optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
            max_seq_len=max_sequence_length,
            cache_hf=True,
        )
        state_dict = model_args.load_state_dict()
        torch_model = model_args.cached_hf_model
        assert isinstance(torch_model, Qwen2_5_VLForConditionalGeneration)

        dtype = ttnn.bfloat16

        return torch_model, Transformer(
            args=model_args,
            mesh_device=self._device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
        )

    def encode(self, prompts: Iterable[str], *, num_images_per_prompt: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_qwen_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            torch_text_encoder=self._torch_encoder,
            max_sequence_length=512,
            mesh_device=self._device,
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L188
def _get_qwen_prompt_embeds(
    prompts: Iterable[str],
    text_encoder: Transformer | None,
    torch_text_encoder: Qwen2_5_VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    mesh_device: ttnn.MeshDevice | None,
    max_sequence_length: int,
    num_images_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34

    prompts = [template.format(e) for e in prompts]

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=max_sequence_length + drop_idx,
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

    if text_encoder is not None:
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
        tt_output = text_encoder(prompt=tt_tokens, attention_mask=tt_attention_mask, device=mesh_device)
        tt_hidden_states = tt_output[-1]

        hidden_states = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_states)[0])
    else:
        tokens = tokens.to(device=torch_text_encoder.device)

        with torch.no_grad():
            output = torch_text_encoder.forward(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = output.hidden_states[-1].to("cpu")

    split_hidden_states = _extract_masked_hidden(hidden_states, attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, encoder_attention_mask


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)
