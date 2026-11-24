# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import SUBFOLDER
from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ...parallel.manager import CCLManager
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

    def encode(
        self, prompt: str, length: int | None = None, tokens: tuple[torch.Tensor, torch.Tensor, int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """`[1, L, 20480]` bf16 text tokens on device (replicated), their RoPE ids and the attended length (see
        `tokenize`; `tokens` = its result when already tokenized)."""
        input_ids, attention_mask, valid = tokens or self.tokenize(prompt, length)

        tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = tensor.from_torch(attention_mask.bool(), device=self._device)
        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=self._rope(attention_mask))
        tokens = ttnn.concat(taps, dim=-1)
        for t in (*taps, tt_ids, tt_mask):
            ttnn.deallocate(t)

        return tokens, text_ids(tokens.shape[-2]), valid

    def encode_rows(
        self,
        tokens: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
        ropes: Sequence[tuple[torch.Tensor, torch.Tensor]],
        sp_axis: int,
        ccl: CCLManager,
    ) -> list[tuple[ttnn.Tensor, torch.Tensor, int]]:
        """`encode` of up to one prompt per device row of `sp_axis` (`tokenize` results of one length, `rope_rows` of each):
        each row runs the single-prompt programs on its own prompt (a short batch repeats the last), then the rows' tokens
        are gathered and each prompt's returned replicated, as `encode` returns them."""
        rows = self._device.shape[sp_axis]
        if not 0 < len(tokens) <= rows or len({ids.shape for ids, _, _ in tokens}) != 1:
            raise ValueError(f"encode_rows takes 1 to {rows} prompts of one length")
        order = [*range(len(tokens)), *[len(tokens) - 1] * (rows - len(tokens))]

        def stacked(parts: Sequence[torch.Tensor], **kwargs) -> ttnn.Tensor:
            return tensor.from_torch(
                torch.cat([parts[i] for i in order]), device=self._device, mesh_axes=[sp_axis, ...], **kwargs
            )

        tt_ids = stacked([t[0] for t in tokens], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_mask = stacked([t[1].bool() for t in tokens])
        pos_embeds = tuple(stacked([rope[i] for rope in ropes]) for i in range(2))
        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=pos_embeds)
        local = ttnn.concat(taps, dim=-1)
        for t in (*taps, tt_ids, tt_mask, *pos_embeds):
            ttnn.deallocate(t)
        length, width = int(local.shape[-2]), int(local.shape[-1])
        gathered = ccl.all_gather(local, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
        ttnn.deallocate(local)
        out = [
            (ttnn.slice(gathered, [0, i * length, 0], [1, (i + 1) * length, width]), text_ids(length), valid)
            for i, (_, _, valid) in enumerate(tokens)
        ]
        ttnn.deallocate(gathered)
        return out

    def rope_rows(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One prompt's bf16 cos / sin on the host (what `_rope` uploads)."""
        cos, sin = self._encoder.create_rope_tensors(1, int(attention_mask.shape[-1]), attention_mask)
        return cos.to(torch.bfloat16), sin.to(torch.bfloat16)

    def _rope(self, attention_mask: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (int(attention_mask.shape[-1]), int(attention_mask.sum()))
        if key in self._rope_cache:
            self._rope_cache.move_to_end(key)
            return self._rope_cache[key]
        pair = self._rope_cache[key] = tuple(bf16_tensor(x, device=self._device) for x in self.rope_rows(attention_mask))
        if len(self._rope_cache) > ROPE_CACHE_SIZE:
            for t in self._rope_cache.popitem(last=False)[1]:
                ttnn.deallocate(t)
        return pair

    def close(self) -> None:
        for pair in self._rope_cache.values():
            for t in pair:
                ttnn.deallocate(t)
        self._rope_cache.clear()
