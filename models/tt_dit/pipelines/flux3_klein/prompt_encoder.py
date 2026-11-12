# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import SUBFOLDER, flux3_klein_rope_theta
from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, interleaved_rope_permutation
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .packing import MAX_TEXT_LENGTH, padded_text_length, text_ids

ENCODER_ROW_STEP = 128


def encoder_rows(valid: int, bucket: int) -> int:
    """Rows the encoder runs on: the attended rows rounded up to ENCODER_ROW_STEP, at most the bucket."""
    return min(bucket, -(-valid // ENCODER_ROW_STEP) * ENCODER_ROW_STEP)


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
        self._pos_cache: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

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
        `tokenize`). The encoder runs on `encoder_rows` rows; the rest of the L rows is zero."""
        input_ids, attention_mask, valid = self.tokenize(prompt, length)
        bucket_rows = int(attention_mask.shape[-1])
        rows = encoder_rows(valid, bucket_rows)
        tokens = self._encode(input_ids[:, :rows], attention_mask[:, :rows], bucket_rows)
        return tokens, text_ids(tokens.shape[-2]), valid

    def warmup(self, bucket: int) -> None:
        """Compile the encoder at every row count a prompt of this text bucket runs on: bucket / 128 encoder runs
        (8, 16, 32 for the 1024, 2048, 4096 buckets), each a new shape unless a smaller bucket compiled it."""
        for rows in (*range(ENCODER_ROW_STEP, bucket, ENCODER_ROW_STEP), bucket):
            attention_mask = torch.zeros((1, rows), dtype=torch.long)
            attention_mask[:, : max(1, rows - ENCODER_ROW_STEP + 1)] = 1
            ttnn.deallocate(self._encode(torch.zeros_like(attention_mask), attention_mask, bucket))

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, bucket_rows: int) -> ttnn.Tensor:
        real = int(attention_mask.sum())
        cache_key = (int(attention_mask.shape[-1]), real)
        cached = self._pos_cache.get(cache_key)

        positions = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1).reshape(-1)
        inv_freq = 1.0 / (
            self._rope_theta ** (torch.arange(0, self._head_dim, 2, dtype=torch.int64).float() / self._head_dim)
        )
        freqs = positions.float()[:, None] * inv_freq[None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., interleaved_rope_permutation(emb.shape[-1])]

        tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = tensor.from_torch(attention_mask.bool(), device=self._device)
        if cached is not None:
            pos_embeds = cached
        else:
            pos_embeds = (
                bf16_tensor(emb.cos().to(torch.bfloat16)[None, None], device=self._device),
                bf16_tensor(emb.sin().to(torch.bfloat16)[None, None], device=self._device),
            )
            if len(self._pos_cache) >= 8:
                _, old_pair = self._pos_cache.pop(next(iter(self._pos_cache)))
                for t in old_pair:
                    ttnn.deallocate(t)
            self._pos_cache[cache_key] = pos_embeds
        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=pos_embeds)
        tokens = ttnn.concat(taps, dim=-1)
        for t in (*taps, tt_ids, tt_mask):
            ttnn.deallocate(t)
        if tokens.shape[-2] < bucket_rows:
            padded = ttnn.pad(tokens, [(0, 0), (0, bucket_rows - tokens.shape[-2]), (0, 0)], value=0.0)
            ttnn.deallocate(tokens)
            tokens = padded
        return tokens
