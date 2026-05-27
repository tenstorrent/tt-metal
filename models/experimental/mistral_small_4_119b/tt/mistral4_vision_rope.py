# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
2D rotary positional embedding for Pixtral vision patches.

For an image divided into a grid of patches (H_patches × W_patches), each
patch's position embedding splits the head dim in half:
    - first half rotates by the row index
    - second half rotates by the column index

The cos/sin tables are precomputed once on the host and uploaded as
[P × P, head_dim] device tensors (P = VISION_MAX_PATCHES_PER_SIDE).
Per-patch lookup is a single ``ttnn.embedding`` keyed on the flat 2D
position index  (row * P + col)  — fully on device, no host round trip.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    VISION_HEAD_DIM,
    VISION_MAX_PATCHES_PER_SIDE,
    VISION_ROPE_THETA,
)


def _build_2d_rope_table(
    head_dim: int = VISION_HEAD_DIM,
    max_patches_per_side: int = VISION_MAX_PATCHES_PER_SIDE,
    theta: float = VISION_ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin tables of shape [P*P, head_dim] indexed by flat patch id (h*P+w).
    Mirrors HF ``Mistral4RotaryEmbedding`` 2D variant for Pixtral patches.
    """
    base_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [head_dim/2]
    h_idx = torch.arange(max_patches_per_side)
    w_idx = torch.arange(max_patches_per_side)
    freqs_h = torch.outer(h_idx, base_freqs[::2])  # [P, head_dim/4]
    freqs_w = torch.outer(w_idx, base_freqs[1::2])  # [P, head_dim/4]

    inv_freq = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
            freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
        ],
        dim=-1,
    ).reshape(
        -1, head_dim // 2
    )  # [P*P, head_dim/2]

    full_freqs = torch.cat([inv_freq, inv_freq], dim=-1)  # [P*P, head_dim]
    return full_freqs.cos(), full_freqs.sin()


class TtPixtralRoPE2D:
    """Precomputed 2D RoPE table + on-device per-patch lookup."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        head_dim: int = VISION_HEAD_DIM,
        max_patches_per_side: int = VISION_MAX_PATCHES_PER_SIDE,
        theta: float = VISION_ROPE_THETA,
    ):
        self.mesh_device = mesh_device
        self.head_dim = head_dim
        self.max_patches_per_side = max_patches_per_side

        cos_t, sin_t = _build_2d_rope_table(head_dim, max_patches_per_side, theta)
        # cos/sin are [P*P, head_dim]; ttnn.embedding expects the embedding table
        # in ROW_MAJOR layout with a 2D shape — keep as-is.
        self.cos_table = ttnn.as_tensor(
            cos_t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_table = ttnn.as_tensor(
            sin_t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def lookup_device(self, ids_tt: ttnn.Tensor, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Look up cos/sin from a pre-uploaded uint32 position-id tensor on device.

        Returns (cos, sin) each of shape [1, 1, seq_len, head_dim] in TILE layout.
        Does not deallocate ``ids_tt`` (caller owns the buffer for trace replay).
        """
        cos = ttnn.embedding(
            ids_tt,
            self.cos_table,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            ids_tt,
            self.sin_table,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.reshape(cos, [1, 1, seq_len, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, seq_len, self.head_dim])
        return cos, sin

    def lookup(self, position_ids: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Look up cos/sin for ``position_ids`` (CPU long tensor of length seq_len) on device.

        Returns (cos, sin) each of shape [1, 1, seq_len, head_dim] in TILE layout,
        ready to broadcast against [batch, n_heads, seq_len, head_dim] q/k tensors.
        """
        ids_tt = ttnn.as_tensor(
            position_ids.to(torch.int32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        cos, sin = self.lookup_device(ids_tt, position_ids.numel())
        ttnn.deallocate(ids_tt)
        return cos, sin


def position_ids_from_grid(h_patches: int, w_patches: int) -> torch.Tensor:
    """
    Flat 2D position ids for an h×w patch grid, using stride = VISION_MAX_PATCHES_PER_SIDE
    so the ids index into the global [P*P, head_dim] RoPE table.

    Returns shape [h*w] long tensor.
    """
    P = VISION_MAX_PATCHES_PER_SIDE
    rows = torch.arange(h_patches).view(-1, 1).expand(h_patches, w_patches)
    cols = torch.arange(w_patches).view(1, -1).expand(h_patches, w_patches)
    return (rows * P + cols).reshape(-1).to(torch.long)
