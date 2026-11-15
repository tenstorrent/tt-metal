# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import SUBFOLDER
from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .packing import MAX_TEXT_LENGTH, padded_text_length, text_ids

ROPE_CACHE_SIZE = 8


class Flux3KleinPromptEncoder:
    def __init__(self, encoder: Qwen3VlTextEncoder, snapshot: str | os.PathLike, device: ttnn.MeshDevice) -> None:
        from transformers import AutoProcessor

        self._encoder = encoder
        self._device = device
        self._processor = AutoProcessor.from_pretrained(str(Path(snapshot) / SUBFOLDER))
        self._processor.tokenizer.padding_side = "right"
        # cos / sin on device by (rows, real tokens): right padding makes them a function of those two
        self._rope_cache: OrderedDict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = OrderedDict()

    def tokenize(self, prompt: str, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Token ids and attention mask padded to `length` rows (default: the reference's multiple of 80), and the
        number of rows the DiT attends: the real tokens rounded up to a multiple of 80."""
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self._processor.tokenizer
        max_length = length or MAX_TEXT_LENGTH
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1]
        valid = min(padded_text_length(real), max_length)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length or valid)
        return inputs["input_ids"], inputs["attention_mask"], valid

    def encode(self, prompt: str, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """`[1, L, 20480]` bf16 text tokens on device (replicated), their RoPE ids and the attended length (see
        `tokenize`)."""
        input_ids, attention_mask, valid = self.tokenize(prompt, length)

        tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = tensor.from_torch(attention_mask.bool(), device=self._device)
        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=self._rope(attention_mask))
        tokens = ttnn.concat(taps, dim=-1)
        for t in (*taps, tt_ids, tt_mask):
            ttnn.deallocate(t)

        return tokens, text_ids(tokens.shape[-2]), valid

    def _rope(self, attention_mask: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (int(attention_mask.shape[-1]), int(attention_mask.sum()))
        if key in self._rope_cache:
            self._rope_cache.move_to_end(key)
            return self._rope_cache[key]
        cos, sin = self._encoder.create_rope_tensors(1, key[0], attention_mask)
        pair = self._rope_cache[key] = (
            bf16_tensor(cos.to(torch.bfloat16), device=self._device),
            bf16_tensor(sin.to(torch.bfloat16), device=self._device),
        )
        if len(self._rope_cache) > ROPE_CACHE_SIZE:
            for t in self._rope_cache.popitem(last=False)[1]:
                ttnn.deallocate(t)
        return pair

    def close(self) -> None:
        for pair in self._rope_cache.values():
            for t in pair:
                ttnn.deallocate(t)
        self._rope_cache.clear()
