# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial RoPE tables (first 64 of 256 head dims, theta 1e7, neox ``rotate_half`` layout).

For text-only inputs HF's interleaved mRoPE sees three identical position streams, so the
interleave is a no-op and cos/sin equal plain 1D RoPE: ``emb = cat(pos * inv_freq, pos * inv_freq)``.
Tables are computed once at setup and live on device; ``forward`` only slices them.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class RotaryConfig:
    rotary_dim: int
    theta: float
    max_seq_len: int
    mesh_device: object


class PplxRotary(LightweightModule):
    def __init__(self, rotary_dim: int, theta: float, max_seq_len: int, mesh_device):
        super().__init__()
        self.config = RotaryConfig(rotary_dim=rotary_dim, theta=theta, max_seq_len=max_seq_len, mesh_device=mesh_device)
        self._build_tables()

    @classmethod
    def from_config(cls, config: RotaryConfig) -> "PplxRotary":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = config
        instance._build_tables()
        return instance

    def _build_tables(self) -> None:
        import torch

        c = self.config
        inv_freq = 1.0 / (c.theta ** (torch.arange(0, c.rotary_dim, 2, dtype=torch.int64).float() / c.rotary_dim))
        freqs = torch.arange(c.max_seq_len, dtype=torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_table, self.sin_table = [
            ttnn.from_torch(
                t.contiguous(),
                device=c.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in (emb.cos(), emb.sin())
        ]

    def forward(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos, sin for absolute positions ``start .. start+length-1``; each [1, length, 64] TILE."""
        if start < 0 or start + length > self.config.max_seq_len:
            raise ValueError(f"Positions {start}..{start + length} exceed the RoPE table ({self.config.max_seq_len})")
        return tuple(
            ttnn.to_layout(
                ttnn.reshape(t[start : start + length, :], [1, length, self.config.rotary_dim]), ttnn.TILE_LAYOUT
            )
            for t in (self.cos_table, self.sin_table)
        )
