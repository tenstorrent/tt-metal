# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Semantic vector quantizer (EMA buffers ``embedding_sum`` / ``cluster_usage`` → centroids)."""

from __future__ import annotations

import torch
import ttnn

from models.experimental.voxtraltts.reference.audio_tokenizer_ops import semantic_codebook_centroids_bf16
from models.experimental.voxtraltts.utils.mesh import voxtral_from_torch


class VoxtralTTSemanticCodebookQuantizer:
    """Argmin L2 to EMA centroid table on device; bf16 matmul → f32 distances → ``argmax(−dist)``."""

    _tile: int = 32

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        state_dict: dict,
        dtype=ttnn.bfloat16,
    ) -> None:
        self.mesh_device = mesh_device
        c_bf16 = semantic_codebook_centroids_bf16(state_dict)
        self.n_codes = int(c_bf16.shape[0])
        self.semantic_dim = int(c_bf16.shape[1])
        self._c_norm_f32 = (c_bf16.float().pow(2).sum(-1)).view(1, -1)
        self._c_norm_f32_tt = voxtral_from_torch(
            self._c_norm_f32.to(torch.float32).contiguous(),
            mesh_device,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.centroids_sn_tile = voxtral_from_torch(
            c_bf16.transpose(0, 1).contiguous(),
            mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.centroid_embedding_weight_tt = voxtral_from_torch(
            c_bf16.contiguous(),
            mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _padded_row_count(self, m: int) -> int:
        """Tile-aligned row count so all-DRAM matmul does not hit ``per_core_M==0``."""
        tile = self._tile
        ratio_floor = (self.n_codes + 7) // 8 + tile
        need = max(m, ratio_floor)
        return (need + tile - 1) // tile * tile

    def __call__(self, x_b1ts: ttnn.Tensor) -> ttnn.Tensor:
        """``[B,1,T,S]`` tile → ``[B,T]`` uint32 indices on device (float32 distance + ``argmax(−dist)``)."""
        if len(x_b1ts.shape) != 4 or int(x_b1ts.shape[1]) != 1:
            raise ValueError(f"Expected [B,1,T,S], got {tuple(x_b1ts.shape)}")
        b, _, t, s = (int(x_b1ts.shape[i]) for i in range(4))
        if s != self.semantic_dim:
            raise ValueError(f"Expected semantic_dim={self.semantic_dim}, got {s}")

        mem = ttnn.DRAM_MEMORY_CONFIG
        m = b * t
        m_pad = self._padded_row_count(m)

        x_rm = ttnn.to_layout(x_b1ts, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (m, s))
        if m < m_pad:
            pad = ttnn.zeros(
                (m_pad - m, s),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=mem,
            )
            x_rm = ttnn.concat([x_rm, pad], dim=0, memory_config=mem)

        x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_rm)

        dots = ttnn.matmul(
            x_tile,
            self.centroids_sn_tile,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        ttnn.deallocate(x_tile)

        dots_f32 = ttnn.typecast(dots, ttnn.float32)
        ttnn.deallocate(dots)
        dots_m = ttnn.slice(dots_f32, [0, 0], [m, self.n_codes])
        ttnn.deallocate(dots_f32)

        x_rows_rm = ttnn.to_layout(x_b1ts, ttnn.ROW_MAJOR_LAYOUT)
        x_rows_rm = ttnn.reshape(x_rows_rm, (m, s))
        x_f32 = ttnn.typecast(x_rows_rm, ttnn.float32)
        ttnn.deallocate(x_rows_rm)
        x_sq = ttnn.pow(x_f32, 2)
        xn = ttnn.sum(x_sq, dim=-1, keepdim=True)
        ttnn.deallocate(x_sq)
        ttnn.deallocate(x_f32)

        two_dots = ttnn.multiply(dots_m, 2.0)
        xn_cn = ttnn.add(xn, self._c_norm_f32_tt)
        dist = ttnn.subtract(xn_cn, two_dots)
        ttnn.deallocate(xn_cn)
        ttnn.deallocate(two_dots)
        ttnn.deallocate(xn)
        ttnn.deallocate(dots_m)

        neg_dist = ttnn.multiply(dist, -1.0)
        ttnn.deallocate(dist)
        neg_dist_rm = ttnn.to_layout(neg_dist, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ttnn.deallocate(neg_dist)
        idx_flat = ttnn.argmax(neg_dist_rm, dim=-1, use_multicore=True)
        ttnn.deallocate(neg_dist_rm)
        idx_bt = ttnn.reshape(idx_flat, (b, t))
        idx_tile = ttnn.to_layout(idx_bt, ttnn.TILE_LAYOUT, memory_config=mem)
        return ttnn.typecast(idx_tile, ttnn.uint32)

    def decode_semantic_embeddings(self, indices_bt: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, T]`` uint32 indices → ``[B, T, semantic_dim]`` bf16 via EMA centroid lookup."""
        if len(indices_bt.shape) != 2:
            raise ValueError(f"Expected [B, T] indices, got {tuple(indices_bt.shape)}")
        return ttnn.embedding(
            indices_bt,
            self.centroid_embedding_weight_tt,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
