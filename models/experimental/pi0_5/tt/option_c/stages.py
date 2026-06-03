# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage layout for pi0.5 Option C (heterogeneous 3-stage pipeline on a Galaxy).

Physical placement on the 8x4 Galaxy:

       col→  0 1 2 3
    row↓  0  V V _ _      V = vision  4 chips,  shape (2,2) offset (0,0)
          1  V V _ _      _ = spare   4 chips,  shape (2,2) offset (0,2)
          2  P P P D      P = prefill 18 chips, shape (6,3) offset (2,0)
          3  P P P D      D = denoise 6 chips,  shape (6,1) offset (2,3)
          4  P P P D
          5  P P P D
          6  P P P D
          7  P P P D

Layer assignment:
  - Vision (4 chips): 3 chips hold SigLIP transformer layers (9 layers each,
    27 total), 1 chip holds patch_conv + pos_embed + final LN + mm_projector
    (the "embed" chip, ~16 MB) + the VLM embed_tokens table (~527 MB) — UNLESS
    we host-resolve text embeddings (recommended; see deployment plan §3.1).
  - Prefill (18 chips): 1 VLM transformer layer per chip; final RMS norm on
    chip 17.
  - Denoise (6 chips): 3 expert layers per chip (18 layers / 6 chips); the
    suffix MLP (action_in/out_proj + time_mlp) is replicated on every denoise
    chip so each step is local. Denoise loop runs 10x entirely on this submesh.

No TP within any stage — every chip owns whole layer(s) by itself, no
all_reduce inside a stage. Inter-stage transport is host-bounce (matches
Option B's transport.py) until tt-blaze D2D sockets land.

KV migration is LAYER-PAIRED: prefill chip i (which owns VLM layer i) ships
its (K, V) to the denoise chip that owns expert layer i. With 18 prefill
chips and 6 denoise chips, denoise chip d receives KV from prefill chips
3d, 3d+1, 3d+2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ----- Physical placement constants ---------------------------------------- #

# Parent mesh — Blackhole Galaxy.
GALAXY_PARENT_SHAPE: Tuple[int, int] = (8, 4)

# (shape, offset) for each submesh.
VISION_SUBMESH_SHAPE: Tuple[int, int] = (2, 2)
VISION_SUBMESH_OFFSET: Tuple[int, int] = (0, 0)
SPARE_SUBMESH_SHAPE: Tuple[int, int] = (2, 2)
SPARE_SUBMESH_OFFSET: Tuple[int, int] = (0, 2)
PREFILL_SUBMESH_SHAPE: Tuple[int, int] = (6, 3)
PREFILL_SUBMESH_OFFSET: Tuple[int, int] = (2, 0)
DENOISE_SUBMESH_SHAPE: Tuple[int, int] = (6, 1)
DENOISE_SUBMESH_OFFSET: Tuple[int, int] = (2, 3)

NUM_VISION_CHIPS = VISION_SUBMESH_SHAPE[0] * VISION_SUBMESH_SHAPE[1]  # 4
NUM_PREFILL_CHIPS = PREFILL_SUBMESH_SHAPE[0] * PREFILL_SUBMESH_SHAPE[1]  # 18
NUM_DENOISE_CHIPS = DENOISE_SUBMESH_SHAPE[0] * DENOISE_SUBMESH_SHAPE[1]  # 6
NUM_SPARE_CHIPS = SPARE_SUBMESH_SHAPE[0] * SPARE_SUBMESH_SHAPE[1]  # 4

# Per-stage layer distribution (default mapping; overridable via StageSpec).
SIGLIP_LAYERS_PER_VISION_CHIP = 9  # 3 chips × 9 = 27 SigLIP layers
NUM_VISION_TRANSFORMER_CHIPS = 3  # SigLIP chips
NUM_VISION_EMBED_CHIPS = 1  # mm_projector + (optional) embed_tokens chip
VLM_LAYERS_PER_PREFILL_CHIP = 1  # 1 layer per chip × 18 chips = 18 VLM layers
EXPERT_LAYERS_PER_DENOISE_CHIP = 3  # 3 layers × 6 chips = 18 expert layers


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage's logical placement on a submesh.

    Mirrors `option_b.stages.StageSpec` so test scaffolding can use the same
    shape, but allows the heterogeneous shape/offset Option C needs.
    """

    stage_idx: int
    name: str
    submesh_shape: Tuple[int, int]
    submesh_offset: Tuple[int, int]

    # Layer ranges this stage owns (half-open).
    siglip_layer_range: Tuple[int, int] = (0, 0)
    vlm_layer_range: Tuple[int, int] = (0, 0)
    expert_layer_range: Tuple[int, int] = (0, 0)

    # Per-chip layer chunking (how layers in `*_layer_range` are split across
    # the submesh's chips). 0 = use submesh num_chips as the splitter.
    siglip_layers_per_chip: int = 0
    vlm_layers_per_chip: int = 0
    expert_layers_per_chip: int = 0

    holds_embed_tokens: bool = False
    holds_vlm_final_norm: bool = False
    holds_suffix_mlp: bool = False
    holds_mm_projector: bool = False
    runs_denoise_loop: bool = False
    receives_kv_migration: bool = False
    emits_kv_migration: bool = False

    @property
    def num_chips(self) -> int:
        return self.submesh_shape[0] * self.submesh_shape[1]


@dataclass
class StageLayout:
    """Full 3-stage heterogeneous layout for Option C on a 32-chip Galaxy.

    Stages (in execution order):
        0 — vision  (4 chips)   StageSpec(stage_idx=0)
        1 — prefill (18 chips)  StageSpec(stage_idx=1)
        2 — denoise (6 chips)   StageSpec(stage_idx=2)

    Spares (4 chips) are not represented as a stage — they're available for a
    future feature (denoise replica, speculative branch, batched serving) but
    not opened by default.
    """

    stages: List[StageSpec] = field(default_factory=list)
    parent_mesh_shape: Tuple[int, int] = GALAXY_PARENT_SHAPE

    def __post_init__(self) -> None:
        if len(self.stages) != 3:
            raise ValueError(f"Option C requires exactly 3 stages, got {len(self.stages)}")
        if self.parent_mesh_shape != GALAXY_PARENT_SHAPE:
            raise ValueError(f"Parent mesh must be {GALAXY_PARENT_SHAPE} for a Galaxy, got {self.parent_mesh_shape}")
        for i, s in enumerate(self.stages):
            if s.stage_idx != i:
                raise ValueError(f"stages[{i}].stage_idx must be {i}, got {s.stage_idx}")
        self._validate_no_overlap()

    def _validate_no_overlap(self) -> None:
        """Make sure no two submeshes occupy the same chip on the parent grid."""
        occupied = {}
        for s in self.stages:
            r0, c0 = s.submesh_offset
            rH, cW = s.submesh_shape
            for r in range(r0, r0 + rH):
                for c in range(c0, c0 + cW):
                    if not (0 <= r < self.parent_mesh_shape[0] and 0 <= c < self.parent_mesh_shape[1]):
                        raise ValueError(
                            f"Stage {s.name} chip ({r},{c}) outside parent mesh " f"{self.parent_mesh_shape}"
                        )
                    if (r, c) in occupied:
                        raise ValueError(f"Stage {s.name} chip ({r},{c}) already used by stage " f"{occupied[(r, c)]}")
                    occupied[(r, c)] = s.name

    def stage(self, idx: int) -> StageSpec:
        return self.stages[idx]


def build_default_layout(vlm_depth: int = 18, expert_depth: int = 18) -> StageLayout:
    """Canonical Option C mapping from PI0_5_GALAXY_DEPLOYMENT_PLAN.md §3.

    Defaults are the real pi0.5 sizes (VLM depth=18, expert depth=18). Pass
    smaller values for shrunk-config tests — but note that the chip-count of
    each submesh stays the same; we just leave some chips idle on the prefill
    submesh when vlm_depth < NUM_PREFILL_CHIPS.
    """
    if vlm_depth < 1 or vlm_depth > NUM_PREFILL_CHIPS:
        raise ValueError(f"vlm_depth must be in [1, {NUM_PREFILL_CHIPS}] for Option C; got {vlm_depth}")
    if expert_depth < 1 or expert_depth > NUM_DENOISE_CHIPS * EXPERT_LAYERS_PER_DENOISE_CHIP:
        raise ValueError(
            f"expert_depth must be in [1, {NUM_DENOISE_CHIPS * EXPERT_LAYERS_PER_DENOISE_CHIP}] for "
            f"Option C; got {expert_depth}"
        )

    stages = [
        StageSpec(
            stage_idx=0,
            name="vision",
            submesh_shape=VISION_SUBMESH_SHAPE,
            submesh_offset=VISION_SUBMESH_OFFSET,
            siglip_layer_range=(0, 27),
            siglip_layers_per_chip=SIGLIP_LAYERS_PER_VISION_CHIP,
            holds_embed_tokens=True,
            holds_mm_projector=True,
        ),
        StageSpec(
            stage_idx=1,
            name="prefill",
            submesh_shape=PREFILL_SUBMESH_SHAPE,
            submesh_offset=PREFILL_SUBMESH_OFFSET,
            vlm_layer_range=(0, vlm_depth),
            vlm_layers_per_chip=VLM_LAYERS_PER_PREFILL_CHIP,
            holds_vlm_final_norm=True,
            emits_kv_migration=True,
        ),
        StageSpec(
            stage_idx=2,
            name="denoise",
            submesh_shape=DENOISE_SUBMESH_SHAPE,
            submesh_offset=DENOISE_SUBMESH_OFFSET,
            expert_layer_range=(0, expert_depth),
            expert_layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
            holds_suffix_mlp=True,
            runs_denoise_loop=True,
            receives_kv_migration=True,
        ),
    ]
    return StageLayout(stages=stages)


def build_shrunk_layout(vlm_depth: int = 2, expert_depth: int = 1) -> StageLayout:
    """Test-only Option C layout with reduced depth so a single-chip baseline fits.

    The submesh shapes stay the same — just fewer layers are mapped onto them.
    Used by the smoke test until full-depth weight upload + KV migration land.
    """
    return build_default_layout(vlm_depth=vlm_depth, expert_depth=expert_depth)
