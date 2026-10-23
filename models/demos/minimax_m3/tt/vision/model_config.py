# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMaxM3VLModelArgs: dimensions of the MiniMax-M3 vision tower and projector for the TT modules.

The defaults are the checkpoint's config.vision_config; the Torch reference (reference/vision.py,
MiniMaxM3VisionConfig) has the same defaults. The checkpoint directory comes from HF_MODEL with no
default, as for the text model (tt/model_config.py). Only the real-weight loader (loader.build_reference)
reads it, so the random-weight unit tests run without it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


def _next_tile_multiple(x: int, tile: int = 32) -> int:
    return ((x + tile - 1) // tile) * tile


@dataclass
class MiniMaxM3VLModelArgs:
    """Vision tower hyperparameters for MiniMax-M3 (a CLIP vision tower with 3D RoPE)."""

    mesh_device: Any = None
    dtype: Any = None  # ttnn.bfloat16 (kept Any to avoid importing ttnn at module import)

    # Authoritative architecture values (config.vision_config).
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_axis_dim: int = 26  # per-axis rotary dim; 3*26 = 78 of head_dim rotated, 2 pass-through

    # Projector / connector.
    projector_hidden_size: int = 6144
    text_hidden_size: int = 6144  # final vision-token dim (= LLM embedding dim)

    # Checkpoint directory, from HF_MODEL (None when it is not set).
    snapshot_dir: Optional[str] = field(default_factory=lambda: os.getenv("HF_MODEL"))

    # ---- derived ----
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 80

    @property
    def padded_head_dim(self) -> int:
        return _next_tile_multiple(self.head_dim)  # 96

    @property
    def rope_rot_dim(self) -> int:
        return 3 * self.rope_axis_dim  # 78

    @property
    def patch_flat_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size  # 1176

    @property
    def merged_hidden_size(self) -> int:
        return self.text_hidden_size * (self.spatial_merge_size**2)  # 24576
