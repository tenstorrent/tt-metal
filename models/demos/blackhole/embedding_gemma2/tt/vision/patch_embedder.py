# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionPatchEmbedder (HF `Gemma4VisionPatchEmbedder`, modeling_gemma4.py:585-627).

    h = (2 * (px - 0.5)) @ W_in                         W_in 768 -> 768, no bias
      + table[0][x] + table[1][y]                       learned table [2, 10240, 768]
    position embedding is zero on padding patches (position (-1, -1))

Padding is handled by index: each table gets one extra all-zero row at index
`position_embedding_size`, and the host maps padding positions to that row
(`VisionInputs.pos_x/pos_y`), so the device does two plain lookups and one add.

Contract
    input   pixel_values [B, 1, S, 768] float32 TILE DRAM (processor output in [0, 1], rows >= N zero)
            pos_x, pos_y [B, S] uint32 ROW_MAJOR DRAM (padding -> position_embedding_size)
    output  [B, 1, S, hidden] bf16 TILE DRAM
    `2 * px - 1` runs in fp32 on device, then the matmul (weight dtype, fidelity, output dtype from the
    precision policy `vision.weights.patch_embed`; E5b: bf16, HiFi4, fp32 accumulation).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.precision import VisionPrecision, resolve_precision
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class VisionPatchEmbedder(LightweightModule):
    def __init__(
        self,
        config: VisionConfig,
        w_in: ttnn.Tensor,
        table_x: ttnn.Tensor,
        table_y: ttnn.Tensor,
        precision: VisionPrecision | None = None,
    ):
        super().__init__()
        precision = precision or resolve_precision(None).vision
        self.config = config
        self.w_in = w_in
        self.table_x = table_x
        self.table_y = table_y
        self.policy = precision.matmul("patch_embed")
        self.compute_kernel_config = self.policy.compute_kernel_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: VisionConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "vision.patch_embedder",
        precision: VisionPrecision | None = None,
    ) -> "VisionPatchEmbedder":
        """`state_dict` holds the HF `vision_tower.patch_embedder.` sub-state; `precision` the vision
        policy (`vision.weights.patch_embed`; None: the default precision file)."""
        precision = precision or resolve_precision(None).vision
        H, P, T = config.hidden_size, config.patch_dim, config.position_embedding_size
        w = state_dict["input_proj.weight"]
        if tuple(w.shape) != (H, P):
            raise ValueError(f"{name}.input_proj: expected {(H, P)}, got {tuple(w.shape)}")
        table = state_dict["position_embedding_table"]
        if tuple(table.shape) != (2, T, H):
            raise ValueError(f"{name}.position_embedding_table: expected {(2, T, H)}, got {tuple(table.shape)}")
        w_in = as_weight(
            linear_weight(w),
            device,
            cache_dir=cache_dir,
            name=f"{name}.input_proj.weight",
            dtype=precision.matmul("patch_embed").ttnn_dtype,
        )
        tables = []
        for axis in (0, 1):
            padded = torch.cat([table[axis], torch.zeros(1, H, dtype=table.dtype)], dim=0)  # zero row at index T
            tables.append(
                as_weight(
                    padded,
                    device,
                    cache_dir=cache_dir,
                    name=f"{name}.position_table_{'xy'[axis]}_padrow.weight",
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )
        return cls(config, w_in, *tables, precision=precision)

    def lookup(self, idx: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
        B, S = idx.shape[0], idx.shape[1]
        e = ttnn.embedding(
            idx, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(e, (B, 1, S, self.config.hidden_size))

    def forward(self, pixel_values: ttnn.Tensor, pos_x: ttnn.Tensor, pos_y: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        px = ttnn.multiply(pixel_values, 2.0, memory_config=dram)
        px1 = ttnn.subtract(px, 1.0, memory_config=dram)
        ttnn.deallocate(px)
        h = ttnn.linear(
            px1,
            self.w_in,
            dtype=self.policy.out_ttnn_dtype,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=dram,
        )
        ttnn.deallocate(px1)
        ex = self.lookup(pos_x, self.table_x)
        ey = self.lookup(pos_y, self.table_y)
        pe = ttnn.add(ex, ey, memory_config=dram)
        ttnn.deallocate(ex)
        ttnn.deallocate(ey)
        out = ttnn.add(h, pe, memory_config=dram)
        ttnn.deallocate(h)
        ttnn.deallocate(pe)
        return out
