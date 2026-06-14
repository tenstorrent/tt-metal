# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE for MiniMax-M3 text attention (TP=32, replicated cos/sin).

Matches ``reference.functional.build_rope_cos_sin`` + ``rope_forward``:

  * PARTIAL rope: ``rotary_dim = 64`` of ``head_dim = 128``. Only the first 64
    channels of each head are rotated; channels [64:128] pass through.
  * theta = 5e6, half-split GPT-NeoX ``rotate_half`` (``cat(-x2, x1)``).
  * cos/sin tables are built HOST-side (``build_cos_sin``) — the only torch is
    in table construction (outside the forward path), replicated to the mesh.

The on-device forward uses only ttnn elementwise ops (slice / mul / neg /
concat / add) — NO torch, NO ``rotary_embedding_llama`` (whose interleaved
trans-mat convention differs from M3's half-split). This keeps the math an
exact port of the reference and passes the no-torch-in-forward lint guard.

q layout: ``[B, H, S, head_dim]``, k layout: ``[B, n_kv, S, head_dim]``.
cos/sin are passed as device tensors shaped ``[B, 1, S, rotary_dim]`` so they
broadcast across heads.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimaxai_minimax_m3.tt import model_config as mc


def build_cos_sin(seq_len: int, rotary_dim: int = 64, theta: float = 5e6, position_ids=None):
    """Build partial-rope cos/sin tables HOST-side (mirrors the reference).

    Returns ``(cos, sin)`` each of torch shape ``[B, S, rotary_dim]``. This is
    table construction (NOT the forward path); converting to device tensors is
    done by :func:`cos_sin_to_mesh`.
    """
    if position_ids is None:
        position_ids = torch.arange(seq_len).unsqueeze(0)
    position_ids = position_ids.float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :]
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rotary_dim]
    return emb.cos(), emb.sin()


def cos_sin_to_mesh(cos: torch.Tensor, sin: torch.Tensor, mesh, dtype=mc.ACT_DTYPE):
    """Push host cos/sin [B,S,rotary_dim] to the mesh as [B,1,S,rotary_dim].

    Replicated across the mesh (rope tables are small / shared) and unsqueezed
    at dim 1 so they broadcast over the head axis of [B,H,S,head_dim] q/k.
    """
    cos = cos.unsqueeze(1)  # [B, 1, S, rotary_dim]
    sin = sin.unsqueeze(1)
    to = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mc.replicate_mapper(mesh)
    )
    return to(cos), to(sin)


class RotaryEmbedding(LightweightModule):
    def __init__(self, mesh_device, cos: ttnn.Tensor, sin: ttnn.Tensor, rotary_dim: int = 64):
        """
        Args:
            mesh_device: open bh_galaxy mesh.
            cos, sin: device tensors ``[B, 1, S, rotary_dim]`` (from
                :func:`cos_sin_to_mesh`).
            rotary_dim: partial rotary dim (64).
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.cos = cos
        self.sin = sin
        self.rotary_dim = rotary_dim

    def _rotate_half(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Half-split rotate: cat(-x2, x1) over the last dim."""
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _apply(self, t: ttnn.Tensor) -> ttnn.Tensor:
        rd = self.rotary_dim
        head_dim = t.shape[-1]
        t_rot = t[..., :rd]
        embed = ttnn.add(ttnn.mul(t_rot, self.cos), ttnn.mul(self._rotate_half(t_rot), self.sin))
        if head_dim > rd:
            t_pass = t[..., rd:]
            return ttnn.concat([embed, t_pass], dim=-1)
        return embed

    def forward(self, q: ttnn.Tensor, k: ttnn.Tensor):
        """Apply partial rope to q and k. Returns ``(q_embed, k_embed)``."""
        return self._apply(q), self._apply(k)
