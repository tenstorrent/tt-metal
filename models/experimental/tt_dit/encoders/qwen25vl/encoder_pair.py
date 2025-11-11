# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger
from models.demos.qwen25_vl.tt.common import multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from transformers import PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        self._torch_encoder, self._encoder, self._model_args = self._load_encoder(
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
    ) -> tuple[Qwen2_5_VLForConditionalGeneration, Transformer | None, ModelArgs | None]:
        if torch_only:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint), None, None

        logger.info("creating encoder on device...")

        os.environ["HF_MODEL"] = checkpoint
        model_args = ModelArgs(
            self._device,
            instruct=True,
            max_batch_size=max_batch_size,
            optimizations=lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
            max_seq_len=max_sequence_length,
            cache_hf=True,
        )
        state_dict = model_args.load_state_dict()
        torch_model = model_args.cached_hf_model
        assert isinstance(torch_model, Qwen2_5_VLForConditionalGeneration)

        dtype = ttnn.bfloat8_b

        model = Transformer(
            args=model_args,
            mesh_device=self._device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
        )

        return torch_model, model, model_args

    def encode(
        self, prompts: Sequence[str], *, num_images_per_prompt: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_qwen_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            model_args=self._model_args,
            torch_text_encoder=self._torch_encoder,
            sequence_length=sequence_length,
            mesh_device=self._device,
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L188
def _get_qwen_prompt_embeds(
    prompts: Sequence[str],
    text_encoder: Transformer | None,
    model_args: ModelArgs | None,
    torch_text_encoder: Qwen2_5_VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    mesh_device: ttnn.MeshDevice | None,
    sequence_length: int,
    num_images_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34

    prompts = [template.format(e) for e in prompts]

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=sequence_length + drop_idx,
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
        assert len(prompts) == 1, "only batch size 1 is supported by the transformer model in prefill mode"

        assert mesh_device is not None
        assert model_args is not None

        pad_token_id = tokenizer.pad_token_id

        input_embeds = torch_text_encoder.model.language_model.embed_tokens(tokens)
        pad_embedding = torch_text_encoder.model.language_model.embed_tokens(torch.tensor(pad_token_id))

        input_prefill_pt, _decoding_pos, _prefill_lens = preprocess_inputs_prefill(
            input_embeds,
            model_args,
            attention_mask,
            pad_embedding=pad_embedding,
        )

        rope = multimodal_rope_from_hf(
            tokenizer_out, input_embeds, torch_text_encoder, model_args, pad_token_id=pad_token_id
        )

        prefill_input, rot_mats_prefill, page_table_tt, _ = text_encoder.prepare_inputs_prefill(
            input_prefill_pt, rot_mats=rope
        )

        tt_hidden_states = text_encoder.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_prefill,
            page_table=page_table_tt,
        )
        tt_hidden_states = text_encoder.norm(tt_hidden_states, mode="prefill")

        hidden_states = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_states)[0])
        hidden_states = hidden_states[:, :, : tokens.shape[1], :].squeeze(1)
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
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(sequence_length - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(sequence_length - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, encoder_attention_mask


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)
