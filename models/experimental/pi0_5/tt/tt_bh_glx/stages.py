# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage layout + timing dataclasses for the BH-Galaxy host-bounce pipeline.

Physical placement on the 8x4 BH Galaxy parent mesh:

    col→  0 1 2 3
 row↓  0  V V V V    V = vision  shape (1,4) offset (0,0)  4 chips
       1  P P P D    P = prefill shape (6,3) offset (1,0) 18 chips
       2  P P P D    D = denoise shape (6,1) offset (1,3)  6 chips
       3  P P P D    (row 7 = 4 spare chips)
       4  P P P D
       5  P P P D
       6  P P P D
       7  . . . .

Vision chip roles (chip 0 of vision_per_chip is row=0,col=0):
    chip 0: patch_embed + pos_emb only (pure I/O)
    chip 1: SigLIP layers 0..8
    chip 2: SigLIP layers 9..17
    chip 3: SigLIP layers 18..26 + post_layernorm + mm_projector

Prefill chips: 1 Gemma-2B block per chip (18 total).
Denoise chips: 3 AdaRMS Gemma-300M blocks per chip (6 total).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


PARENT_MESH_SHAPE: Tuple[int, int] = (8, 4)

VISION_SUBMESH_SHAPE: Tuple[int, int] = (1, 4)
VISION_SUBMESH_OFFSET: Tuple[int, int] = (0, 0)
VISION_NUM_CHIPS: int = 4

PREFILL_SUBMESH_SHAPE: Tuple[int, int] = (6, 3)
PREFILL_SUBMESH_OFFSET: Tuple[int, int] = (1, 0)
PREFILL_NUM_CHIPS: int = 18

DENOISE_SUBMESH_SHAPE: Tuple[int, int] = (6, 1)
DENOISE_SUBMESH_OFFSET: Tuple[int, int] = (1, 3)
DENOISE_NUM_CHIPS: int = 6

SIGLIP_TOTAL_LAYERS: int = 27
SIGLIP_LAYERS_PER_CHIP: int = 9  # chips 1..3 each hold 9; chip 0 holds embedding only

EXPERT_TOTAL_LAYERS: int = 18
EXPERT_LAYERS_PER_CHIP: int = 3  # 6 chips × 3 layers

VLM_TOTAL_LAYERS: int = 18  # 1 layer per chip across 18 chips


@dataclass
class MeshHandles:
    """Live submesh handles returned by open_galaxy_mesh()."""

    parent: object
    vision_submesh: object
    prefill_submesh: object
    denoise_submesh: object
    vision_per_chip: List[object] = field(default_factory=list)
    prefill_per_chip: List[object] = field(default_factory=list)
    denoise_per_chip: List[object] = field(default_factory=list)


@dataclass
class StageTimings:
    """Wall-clock per stage in ms. denoise_step_ms holds per-step times."""

    vision_ms: float = 0.0
    transport_v2p_ms: float = 0.0
    prefill_ms: float = 0.0
    kv_migration_ms: float = 0.0
    denoise_step_ms: List[float] = field(default_factory=list)
    transport_d2h_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def denoise_total_ms(self) -> float:
        return sum(self.denoise_step_ms)
