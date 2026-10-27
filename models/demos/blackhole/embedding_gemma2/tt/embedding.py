# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TextEmbedding (HF `EmbeddingGemma2TextScaledWordEmbedding`): `embed_tokens(ids) * sqrt(hidden)`.

The sqrt(512) scale is folded into the table at load time. HF computes bf16(w) * 22.6274 and the
device would round the product to bf16 anyway, so folding gives the same single rounding.

Contract
    input   input_ids [B, S] uint32 ROW_MAJOR (device). S is the physical (bucketed) length.
    output  inputs_embeds [B, 1, S, hidden] bf16 TILE, DRAM interleaved.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import TextModelConfig
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight


class TextEmbedding(LightweightModule):
    def __init__(self, config: TextModelConfig, weight: ttnn.Tensor):
        super().__init__()
        self.config = config
        self.weight = weight

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], config: TextModelConfig, device, *, cache_dir: Path | None = None
    ) -> "TextEmbedding":
        w = state_dict["embed_tokens.weight"]
        if tuple(w.shape) != (config.vocab_size, config.hidden_size):
            raise ValueError(f"embed_tokens: unexpected shape {tuple(w.shape)}")
        scaled = (w.float() * torch.tensor(config.embed_scale, dtype=torch.float32)).to(torch.bfloat16)
        weight = as_weight(
            scaled, device, cache_dir=cache_dir, name="embed_tokens.scaled.weight", layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return cls(config, weight)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        B, S = input_ids.shape[0], input_ids.shape[1]
        x = ttnn.embedding(
            input_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(x, (B, 1, S, self.config.hidden_size))
