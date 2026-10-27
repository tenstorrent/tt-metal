# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE tables (HF `EmbeddingGemma2RotaryEmbedding`, rope_type "default", full rotation).

One cos/sin pair per layer type, built once on the host at setup:
    inv_freq = theta ** -(arange(0, d, 2) / d);   emb = cat(pos * inv_freq, pos * inv_freq)
    cos/sin  [1, 1, max_seq_len, head_dim] bf16 TILE (HF rotate_half layout)

`ttnn.experimental.rotary_embedding(x, cos, sin)` applies x*cos + rotate_half(x)*sin and reads
rows 0..S-1 of the table for an input with S rows, so one table serves every bucket.
Positions are 0..S-1 for every row of the batch, as in HF (`position_ids = arange(S)`).
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import FULL, SLIDING, TextModelConfig


def rope_cos_sin(theta: float, head_dim: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class RotaryTables(LightweightModule):
    def __init__(self, config: TextModelConfig, tables: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]):
        super().__init__()
        self.config = config
        self.tables = tables

    @classmethod
    def build(cls, config: TextModelConfig, device) -> "RotaryTables":
        tables = {}
        for lt in (SLIDING, FULL):
            hd = config.head_dim_for(lt)
            cos, sin = rope_cos_sin(config.rope_theta[lt], hd, config.max_seq_len)
            tables[lt] = tuple(
                ttnn.from_torch(
                    t.reshape(1, 1, config.max_seq_len, hd),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for t in (cos, sin)
            )
        return cls(config, tables)

    def forward(self, x: ttnn.Tensor, layer_type: str, memory_config=None) -> ttnn.Tensor:
        """x [B, H, S, head_dim] -> rotated, same shape. S <= max_seq_len."""
        cos, sin = self.tables[layer_type]
        return ttnn.experimental.rotary_embedding(x, cos, sin, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
