# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT attention block.

QKV-packed projection, 2D RoPE applied to Q/K, variable-length attention
via `ttnn.transformer.windowed_scaled_dot_product_attention` with
cu_seqlens (NaViT-style block-diagonal mask).

Reference: `MoonVitEncoderLayer.attention_qkvpacked` in modeling_kimi_vl.py.
"""
from __future__ import annotations

from models.common.lightweightmodule import LightweightModule


class MoonVisionAttention(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
    ):
        super().__init__()
        raise NotImplementedError("Phase 1 — MoonVisionAttention")

    def forward(self, x, cu_seqlens, rot_mats):
        raise NotImplementedError("Phase 1 — MoonVisionAttention.forward")
