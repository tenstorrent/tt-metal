# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import SUBFOLDER, flux3_klein_rope_theta
from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .packing import MAX_TEXT_LENGTH, padded_text_length, text_ids


class Flux3KleinPromptEncoder:
    def __init__(
        self, encoder: Qwen3VlTextEncoder, snapshot: str | os.PathLike, device: ttnn.MeshDevice, *, text_config: dict
    ) -> None:
        from transformers import AutoProcessor

        self._encoder = encoder
        self._device = device
        self._processor = AutoProcessor.from_pretrained(str(Path(snapshot) / SUBFOLDER))
        self._processor.tokenizer.padding_side = "right"
        self._head_dim = int(text_config["head_dim"])
        self._rope_theta = flux3_klein_rope_theta(text_config)

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
        """`[L, 20480]` bf16 text tokens, their RoPE ids and the attended length (see `tokenize`)."""
        input_ids, attention_mask, valid = self.tokenize(prompt, length)

        # HF's positions for right padding; pad rows are real DiT inputs up to the multiple of 80
        positions = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1).reshape(-1)
        inv_freq = 1.0 / (
            self._rope_theta ** (torch.arange(0, self._head_dim, 2, dtype=torch.int64).float() / self._head_dim)
        )
        freqs = positions.float()[:, None] * inv_freq[None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)

        tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = tensor.from_torch(attention_mask.bool(), device=self._device)
        pos_embeds = (
            bf16_tensor(emb.cos().to(torch.bfloat16)[None, None], device=self._device),
            bf16_tensor(emb.sin().to(torch.bfloat16)[None, None], device=self._device),
        )
        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=pos_embeds)
        tokens = ttnn.concat(taps, dim=-1)
        for t in (*taps, tt_ids, tt_mask, *pos_embeds):
            ttnn.deallocate(t)

        rows = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0])[0]
        ttnn.deallocate(tokens)
        return rows, text_ids(rows.shape[0]), valid
