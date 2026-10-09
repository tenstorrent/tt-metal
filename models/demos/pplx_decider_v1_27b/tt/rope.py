# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial RoPE tables (first 64 of 256 head dims, theta 1e7) for one fused full-width rotary op.

HF rotates the first ``rotary_dim`` dims of each head in the neox ``rotate_half`` layout: dim j
pairs with dim j + rotary_dim/2. For text-only inputs HF's interleaved mRoPE sees three identical
position streams, so cos/sin equal plain 1D RoPE.

The TT path runs ``ttnn.experimental.rotary_embedding_llama`` over the whole 256-dim head instead
of slice -> rotate 64 dims -> concat. That op rotates adjacent pairs (2i, 2i+1). The weight adapter
therefore permutes the q/k head dims with ``rope_head_permutation`` (pair (j, j+32) -> (2j, 2j+1),
the rest unchanged) and these tables carry cos=1, sin=0 on the 192 pass-through dims. q and k get
the same permutation, so q.k and the attention output are unchanged; v is not permuted.

Tables are computed once at setup and live on device; ``forward`` only slices them.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


def rope_head_permutation(head_dim: int, rotary_dim: int) -> list[int]:
    """``new[p] = old[perm[p]]``: neox pairs (j, j + rotary_dim/2) become adjacent (2j, 2j+1)."""
    half = rotary_dim // 2
    perm = [0] * rotary_dim
    for j in range(half):
        perm[2 * j], perm[2 * j + 1] = j, j + half
    return perm + list(range(rotary_dim, head_dim))


@dataclass
class RotaryConfig:
    rotary_dim: int
    theta: float
    max_seq_len: int
    mesh_device: object
    head_dim: int


class PplxRotary(LightweightModule):
    def __init__(self, rotary_dim: int, theta: float, max_seq_len: int, mesh_device, *, head_dim: int):
        super().__init__()
        self.config = RotaryConfig(
            rotary_dim=rotary_dim, theta=theta, max_seq_len=max_seq_len, mesh_device=mesh_device, head_dim=head_dim
        )
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
        # Interleaved pairs share one frequency; pass-through dims rotate by 0 (cos 1, sin 0).
        angles = freqs.repeat_interleave(2, dim=-1)
        pad = c.head_dim - c.rotary_dim
        cos = torch.nn.functional.pad(angles.cos(), (0, pad), value=1.0)
        sin = torch.nn.functional.pad(angles.sin(), (0, pad), value=0.0)
        # TILE tables [1, 1, max_seq_len, head_dim]: a chunk is one tile-row slice, no per-call tilize.
        self.cos_table, self.sin_table = [
            ttnn.from_torch(
                t.reshape(1, 1, c.max_seq_len, c.head_dim).contiguous(),
                device=c.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in (cos, sin)
        ]

    def forward(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos, sin for absolute positions ``start .. start+length-1``; each [1, 1, length, head_dim] TILE."""
        if start < 0 or start + length > self.config.max_seq_len:
            raise ValueError(f"Positions {start}..{start + length} exceed the RoPE table ({self.config.max_seq_len})")
        if start % 32:
            raise ValueError(f"RoPE chunk start {start} must be tile aligned")
        return tuple(t[:, :, start : start + length, :] for t in (self.cos_table, self.sin_table))
