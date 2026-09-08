# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3: RoPE for the DFlash drafter -- full-rotary, half-split, theta 1e7.

Differences from the target's ``tt/rope.py``, which is why this is a separate module rather
than a reuse:

* **Full rotary.** The target rotates 64 of 256 dims (``partial_rotary_factor`` 0.25); the
  drafter rotates all 128. So the tables are a different width and there is no unrotated
  passthrough half to splice back.
* **theta 1e7**, against the target's own value.
* **Asymmetric application.** This is the part worth reading twice. Queries are only the
  ``block`` positions while keys are ``concat(context, block)``, so the reference feeds a
  ``ctx_len + block``-long position range and then slices:

      q_embed = q * cos[..., -q_len:, :] + rotate_half(q) * sin[..., -q_len:, :]
      k_embed = k * cos              + rotate_half(k) * sin

  i.e. **q takes only the last ``block`` rows of the table, k takes all of them.** Feeding q
  the leading rows instead is a silent, plausible-looking bug: shapes match, PCC stays high
  at ctx=0, and it degrades as context grows.

Implementation note: rope is hand-rolled from ttnn primitives rather than using
``ttnn.experimental.rotary_embedding_llama`` (which the target's vision tower uses). That op
wants tile-aligned sequence lengths and the drafter's query is 16 rows -- half a tile.
head_dim 128 splits into two 64-wide halves, both tile-aligned, so the slice/negate/concat
form has no alignment constraint. Adopting the fused op is a later perf step, gated on
padding the block to 32 rows.

The cos/sin table is built on host **once at construction** and lives on device thereafter;
every forward slices it on device. Nothing in the inference path touches host.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig


def build_cos_sin(head_dim: int, theta: float, max_position: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``[max_position, head_dim]``, half-split duplicated.

    Matches ``Qwen3RotaryEmbedding`` for ``rope_type="default"``: fp32 throughout and
    ``attention_scaling == 1.0``, so no extra scale is applied.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [max_position, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # duplicated -> pairs with half-split rotate_half
    return emb.cos(), emb.sin()


class DFlashRoPE:
    """Device-resident cos/sin table plus the half-split rotation."""

    def __init__(
        self,
        mesh_device,
        cfg: DFlashDrafterConfig,
        max_position: int,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.max_position = max_position

        cos, sin = build_cos_sin(cfg.head_dim, cfg.rope_theta, max_position)
        self.cos = self._upload(cos, dtype)
        self.sin = self._upload(sin, dtype)

    def _upload(self, t: torch.Tensor, dtype) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.reshape(1, 1, self.max_position, self.head_dim).to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def tables_for(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos/sin rows for absolute positions ``[start, start + length)``, sliced on device."""
        end = start + length
        assert end <= self.max_position, f"position {end} exceeds table ({self.max_position})"
        begin = (0, 0, start, 0)
        stop = (1, 1, end, self.head_dim)
        step = (1, 1, 1, 1)
        return (
            ttnn.slice(self.cos, begin, stop, step, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.slice(self.sin, begin, stop, step, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

    def apply(self, x, cos, sin):
        """``x * cos + rotate_half(x) * sin`` for ``x`` of ``[1, heads, seq, head_dim]``.

        ``cos``/``sin`` are ``[1, 1, seq, head_dim]`` and broadcast over the head axis.
        """
        half = self.head_dim // 2
        seq = x.shape[-2]
        heads = x.shape[-3]
        mc = ttnn.DRAM_MEMORY_CONFIG

        # rotate_half: cat(-x2, x1) over the last dim. Both halves are tile-aligned (64 = 2
        # tiles), so no padding is involved.
        x1 = ttnn.slice(x, (0, 0, 0, 0), (1, heads, seq, half), (1, 1, 1, 1), memory_config=mc)
        x2 = ttnn.slice(x, (0, 0, 0, half), (1, heads, seq, self.head_dim), (1, 1, 1, 1), memory_config=mc)
        neg_x2 = ttnn.neg(x2, memory_config=mc)
        ttnn.deallocate(x2)
        rotated = ttnn.concat([neg_x2, x1], dim=-1, memory_config=mc)
        ttnn.deallocate(neg_x2)
        ttnn.deallocate(x1)

        x_cos = ttnn.mul(x, cos, memory_config=mc)
        rot_sin = ttnn.mul(rotated, sin, memory_config=mc)
        ttnn.deallocate(rotated)
        out = ttnn.add(x_cos, rot_sin, memory_config=mc)
        ttnn.deallocate(x_cos)
        ttnn.deallocate(rot_sin)
        return out
