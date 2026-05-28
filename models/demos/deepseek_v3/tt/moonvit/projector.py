# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Multi-modal projector: vision-hidden -> LLM-hidden.

Layers (matching KimiVLMultiModalProjector):
    LayerNorm(vision_hidden=1152)
    Linear(merged_dim=4608 -> 4608)
    GELU
    Linear(4608 -> text_hidden)

merged_dim = vision_hidden * merge_kernel_h * merge_kernel_w = 1152 * 4.
text_hidden is the LLM's hidden size (DeepSeek-V3 = 7168; confirm
against V4 config when shipped).

Reference: `KimiVLMultiModalProjector` in modeling_kimi_vl.py.
"""
from __future__ import annotations

from models.common.lightweightmodule import LightweightModule


class MoonViTProjector(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        weight_cache_path,
    ):
        super().__init__()
        raise NotImplementedError("Phase 1 — MoonViTProjector")

    def forward(self, x):
        raise NotImplementedError("Phase 1 — MoonViTProjector.forward")
