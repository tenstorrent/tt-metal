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

import os

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


# Buckets up to this length also get an L1-resident copy of their cos/sin rows at prepare(S)
# (E5a: the rotary op reads its tables from L1 instead of DRAM). All buckets <= 512 together hold
# 3 MB (about 23 KB per L1 bank).
ROPE_L1_MAX_SEQ = int(os.environ.get("EG2_ROPE_L1_MAX_SEQ", "512"))


class RotaryTables(LightweightModule):
    def __init__(self, config: TextModelConfig, tables: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]], device=None):
        super().__init__()
        self.config = config
        self.tables = tables
        self.device = device
        self.l1_tables: dict[tuple[str, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def prepare(self, seq_len: int) -> None:
        """Setup-time: L1 copies of rows 0..seq_len-1 for short buckets (host upload)."""
        if seq_len > ROPE_L1_MAX_SEQ or self.device is None:
            return
        for lt in (SLIDING, FULL):
            if (lt, seq_len) in self.l1_tables:
                continue
            hd = self.config.head_dim_for(lt)
            cos, sin = rope_cos_sin(self.config.rope_theta[lt], hd, seq_len)
            self.l1_tables[(lt, seq_len)] = tuple(
                ttnn.from_torch(
                    t.reshape(1, 1, seq_len, hd),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for t in (cos, sin)
            )

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
        return cls(config, tables, device)

    def forward(self, x: ttnn.Tensor, layer_type: str, memory_config=None) -> ttnn.Tensor:
        """x [B, H, S, head_dim] -> rotated, same shape. S <= max_seq_len."""
        cos, sin = self.l1_tables.get((layer_type, x.shape[2])) or self.tables[layer_type]
        return ttnn.experimental.rotary_embedding(x, cos, sin, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
