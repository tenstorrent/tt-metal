# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT encoder layer.

Pre-norm transformer block:
    x = x + Attn(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Reference: `MoonVitEncoderLayer` in modeling_kimi_vl.py.
"""
from __future__ import annotations

from models.common.lightweightmodule import LightweightModule


class MoonVisionBlock(LightweightModule):
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
        raise NotImplementedError("Phase 1 — MoonVisionBlock")

    def forward(self, x, cu_seqlens, rot_mats):
        raise NotImplementedError("Phase 1 — MoonVisionBlock.forward")
