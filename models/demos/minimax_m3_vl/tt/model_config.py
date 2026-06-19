# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMaxM3VLModelArgs — config for the MiniMax-M3-VL vision tower + projector.

Unlike the MoonViT scaffold this is based on, the HF reference for M3-VL
cannot be loaded in the main tt-metal env: the `MiniMaxM3VLVisionModel`
class only exists in transformers >= 5.x, while the repo pins 4.53. So we
use a **golden-file** flow: a sibling generator (`tests/gen_goldens.py`)
runs in an isolated transformers-5.12 venv to produce reference activations
on disk, and the ttnn PCC tests compare against those goldens. Module
weights are read directly from the checkpoint safetensors (no transformers
dependency) by `_m3_loader.py`.

All hyperparameters below are the authoritative values from
`MiniMaxAI/MiniMax-M3`'s `config.vision_config` (verified 2026-06-09).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Checkpoint snapshot (vision shards 00026 + 00059) and golden cache.
DEFAULT_SNAPSHOT_DIR = os.environ.get(
    "MINIMAX_M3_SNAPSHOT",
    "/localdev/zbaczewski/hf_cache/hub/models--MiniMaxAI--MiniMax-M3/snapshots/051e8f961274fb4e18ac3b57991f13bffedde212",
)
DEFAULT_GOLDENS_DIR = os.environ.get(
    "MINIMAX_M3_GOLDENS",
    os.path.join(os.path.dirname(__file__), "..", "tests", "goldens"),
)


def _next_tile_multiple(x: int, tile: int = 32) -> int:
    return ((x + tile - 1) // tile) * tile


@dataclass
class MiniMaxM3VLModelArgs:
    """Vision-tower hyperparameters for MiniMax-M3-VL (clip_vision_model + 3D RoPE)."""

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

    snapshot_dir: str = field(default_factory=lambda: DEFAULT_SNAPSHOT_DIR)
    goldens_dir: str = field(default_factory=lambda: os.path.abspath(DEFAULT_GOLDENS_DIR))

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
