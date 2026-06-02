# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage layout for pi0.5 Option B (4 stages × 8 chips on a Blackhole Galaxy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage's logical placement on a 4×2 submesh."""

    stage_idx: int
    name: str
    submesh_shape: tuple = (4, 2)  # 4×2 = 8 chips
    siglip_layer_range: tuple = (0, 0)  # (start, end) half-open; empty if no SigLIP layers
    vlm_layer_range: tuple = (0, 0)
    expert_layer_range: tuple = (0, 0)
    holds_embed_tokens: bool = False
    holds_vlm_final_norm: bool = False
    holds_suffix_mlp: bool = False
    holds_mm_projector: bool = False
    runs_denoise_loop: bool = False
    receives_kv_migration: bool = False
    emits_kv_migration: bool = False


@dataclass
class StageLayout:
    """Full 4-stage layout for Option B on a 32-chip Galaxy."""

    stages: List[StageSpec] = field(default_factory=list)
    parent_mesh_shape: tuple = (8, 4)
    submesh_shape: tuple = (4, 2)

    def __post_init__(self) -> None:
        if len(self.stages) != 4:
            raise ValueError(f"Option B requires exactly 4 stages, got {len(self.stages)}")
        if self.parent_mesh_shape != (8, 4):
            raise ValueError(f"Parent mesh must be (8, 4) for a Galaxy, got {self.parent_mesh_shape}")
        if self.submesh_shape != (4, 2):
            raise ValueError(f"Option B submesh must be (4, 2), got {self.submesh_shape}")

    def stage(self, idx: int) -> StageSpec:
        return self.stages[idx]


def build_default_layout() -> StageLayout:
    """The canonical Option B mapping from PI0_5_GALAXY_DEPLOYMENT_PLAN.md §3."""

    stages = [
        StageSpec(
            stage_idx=0,
            name="vision_embed",
            siglip_layer_range=(0, 27),
            holds_embed_tokens=True,
            holds_mm_projector=True,
        ),
        StageSpec(
            stage_idx=1,
            name="vlm_first_half",
            vlm_layer_range=(0, 9),
        ),
        StageSpec(
            stage_idx=2,
            name="vlm_second_half",
            vlm_layer_range=(9, 18),
            holds_vlm_final_norm=True,
            emits_kv_migration=True,
        ),
        StageSpec(
            stage_idx=3,
            name="expert_denoise",
            expert_layer_range=(0, 18),
            holds_suffix_mlp=True,
            runs_denoise_loop=True,
            receives_kv_migration=True,
        ),
    ]
    return StageLayout(stages=stages)


def build_shrunk_layout(vlm_depth: int = 2, expert_depth: int = 1) -> StageLayout:
    """A test-only Option B layout with reduced VLM / expert depth so replicated
    weights fit under the 180 MB per-chip cap. Used by the end-to-end smoke
    test until TP=8 weight sharding lands.

    vlm_depth is split evenly between stages 1 and 2 (one each by default).
    expert_depth lives entirely on stage 3.
    """
    if vlm_depth < 2 or vlm_depth % 2 != 0:
        raise ValueError(f"vlm_depth must be ≥2 and even, got {vlm_depth}")
    if expert_depth < 1:
        raise ValueError(f"expert_depth must be ≥1, got {expert_depth}")
    half = vlm_depth // 2
    stages = [
        StageSpec(
            stage_idx=0,
            name="vision_embed",
            siglip_layer_range=(0, 27),
            holds_embed_tokens=True,
            holds_mm_projector=True,
        ),
        StageSpec(
            stage_idx=1,
            name="vlm_first_half",
            vlm_layer_range=(0, half),
        ),
        StageSpec(
            stage_idx=2,
            name="vlm_second_half",
            vlm_layer_range=(half, vlm_depth),
            holds_vlm_final_norm=True,
            emits_kv_migration=True,
        ),
        StageSpec(
            stage_idx=3,
            name="expert_denoise",
            expert_layer_range=(0, expert_depth),
            holds_suffix_mlp=True,
            runs_denoise_loop=True,
            receives_kv_migration=True,
        ),
    ]
    return StageLayout(stages=stages)
