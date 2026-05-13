# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MM audio codebook embedding table on TTNN (``mm_audio_embeddings.*`` consolidated weights)."""

from __future__ import annotations

import torch
import ttnn


class VoxtralTTAudioCodebookEmbedding:
    """``F.embedding``-compatible lookup: 2D int indices ``[B, T]`` → ``[B, T, dim]`` (tile BF16)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        weight_bf16: torch.Tensor,
        dtype=ttnn.bfloat16,
    ) -> None:
        if weight_bf16.dim() != 2:
            raise ValueError(f"Expected 2D embedding weight, got {tuple(weight_bf16.shape)}")
        self.mesh_device = mesh_device
        self.num_embeddings = int(weight_bf16.shape[0])
        self.embedding_dim = int(weight_bf16.shape[1])
        w = weight_bf16.to(torch.bfloat16).contiguous()
        self.weight_tt = ttnn.from_torch(
            w,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, indices_bt: ttnn.Tensor) -> ttnn.Tensor:
        """``indices_bt``: ``[B, T]`` uint32/int on device. Returns ``[B, T, dim]`` tile BF16."""
        if len(indices_bt.shape) != 2:
            raise ValueError(f"Expected [B, T] indices, got {tuple(indices_bt.shape)}")
        return ttnn.embedding(
            indices_bt,
            self.weight_tt,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
