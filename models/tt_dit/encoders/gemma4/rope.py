# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 dual-flavor rotary embedding (sliding `default` + full `proportional`).

Mirrors transformers' ``Gemma4TextRotaryEmbedding`` math but pre-computes the cos/sin
tables on host for the full ``max_position_embeddings`` range. Tables are stored fp32
on host; ``get_cos_sin(layer_type, position_ids)`` slices the active rows and uploads
as bf16 ttnn tensors.

Half-dim convention: cos/sin have shape ``(1, 1, seq, head_dim // 2)``. Attention
splits ``x`` into ``(x1, x2)`` half-halves and applies::

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

This matches the existing tt_dit Gemma 3 / Wan conventions.

Heads up on memory: at ``max_position_embeddings=262144`` and head dims (256, 512),
the two flavors together cache ~768 MB of fp32 host tensors (sliding 256 MB +
proportional 512 MB).
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module


class Gemma4RotaryEmbedding(Module):
    """Pre-computed dual-flavor RoPE.

    Constructed once per model with the layer-type → rope-params mapping. Holds
    fp32 cos/sin tables per layer type. ``get_cos_sin`` returns ttnn bf16 slices.
    """

    def __init__(
        self,
        *,
        max_position_embeddings: int,
        sliding_head_dim: int,
        sliding_rope_theta: float,
        full_head_dim: int,
        full_rope_theta: float,
        full_partial_rotary_factor: float,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.max_position_embeddings = max_position_embeddings

        self._tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._head_dims: dict[str, int] = {}

        # sliding_attention: default RoPE — every freq slot in [0, head_dim/2) is rotated.
        self._head_dims["sliding_attention"] = sliding_head_dim
        sliding_inv_freq = self._default_inv_freq(sliding_head_dim, sliding_rope_theta)
        self._tables["sliding_attention"] = self._build_cos_sin(sliding_inv_freq, max_position_embeddings)

        # full_attention: proportional partial-rotary RoPE — only the first
        # `int(partial_rotary_factor * head_dim/2)` slots are rotated; the rest are
        # filled with zero inv_freq so cos = 1 and sin = 0 (identity).
        self._head_dims["full_attention"] = full_head_dim
        full_inv_freq = self._proportional_inv_freq(full_head_dim, full_rope_theta, full_partial_rotary_factor)
        self._tables["full_attention"] = self._build_cos_sin(full_inv_freq, max_position_embeddings)

    @staticmethod
    def _default_inv_freq(head_dim: int, base: float) -> torch.Tensor:
        """Standard RoPE inv_freq: 1 / base^(2i/head_dim) for i in [0, head_dim/2)."""
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))

    @staticmethod
    def _proportional_inv_freq(head_dim: int, base: float, partial_rotary_factor: float) -> torch.Tensor:
        """Proportional RoPE: rotate only the first ``int(partial * head_dim/2)`` slots."""
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            return torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)], dim=0)
        return inv_freq_rotated

    @staticmethod
    def _build_cos_sin(inv_freq: torch.Tensor, max_seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Outer product → cos/sin of shape ``(max_seq, head_dim/2)`` in fp32."""
        positions = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        return freqs.cos(), freqs.sin()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Gemma4RotaryEmbedding is used via get_cos_sin(layer_type, position_ids).")

    def get_cos_sin(
        self,
        layer_type: str,
        position_ids: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Slice the cos/sin table at ``position_ids`` and upload bf16 to the mesh.

        Args:
            layer_type: ``"sliding_attention"`` or ``"full_attention"``.
            position_ids: ``(B, seq)`` long tensor of absolute positions.

        Returns:
            Tuple of ttnn tensors, each shape ``(B, 1, seq, head_dim/2)``.
        """
        cos_table, sin_table = self._tables[layer_type]
        cos = cos_table[position_ids]  # (B, seq, head_dim/2)
        sin = sin_table[position_ids]
        # Expand to (B, 1, seq, head_dim/2) — second axis for broadcast over heads.
        cos = cos.unsqueeze(1).to(torch.bfloat16)
        sin = sin.unsqueeze(1).to(torch.bfloat16)
        tt_cos = ttnn.from_torch(cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return tt_cos, tt_sin
