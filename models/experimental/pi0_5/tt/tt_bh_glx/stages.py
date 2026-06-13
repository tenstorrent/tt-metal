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

The physical parent is opened as the full (8,4) torus so FABRIC_2D can train
every ethernet link (each torus link needs a live partner; carving the parent
itself down to (7,4) leaves dangling links → router-sync handshake timeout).

All compute lives on a (7,4) COMPUTE submesh (offset (0,0)) carved from the
parent — exactly the 28 used chips, rows 0..6. The stage submeshes and the
per-chip 1x1s are carved from this compute submesh, and trace capture/replay
runs on it (not the parent). That bounds a captured trace's blocking finish to
the 28 commanded devices: a parent-rooted trace would default its finish to the
full (8,4) range and deadlock waiting on row-7's empty completion queue.

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

# Compute submesh: the 28 used chips (rows 0..6) carved from the (8,4) parent.
# Trace capture/replay roots here so a trace's finish covers only these chips.
COMPUTE_SUBMESH_SHAPE: Tuple[int, int] = (7, 4)
COMPUTE_SUBMESH_OFFSET: Tuple[int, int] = (0, 0)

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


# ─────────────── Fully-traced (collinear, FABRIC_1D) layout ───────────────
# All 3 stages traced. Every cross-stage hand-off is COLLINEAR (shares a row or
# column) so fabric sockets (cross-mesh) and point_to_point (traced, intra-mesh)
# coexist under a single FABRIC_1D config. Global coords on the (8,4) parent:
#
#         col 0      col 1      col 2          col 3
#  row 0  P:L0  →    P:L1  →    P:L2  ─┐        D:L0–2     (denoise traced, col 3)
#  row 1  P:L5  ←    P:L4  ←    P:L3 ←─┘        D:L3–5
#  row 2  P:L6  →    P:L7  →    P:L8  ─┐        D:L6–8
#  row 3  P:L11 ←    P:L10 ←    P:L9 ←─┘        D:L9–11
#  row 4  P:L12 →    P:L13 →    P:L14 ─┐        D:L12–14
#  row 5  P:L17 ←    P:L16 ←    P:L15←─┘        D:L15–17
#  row 6  V:tail ←   V:9–17 ←   V:0–8 ←  V:embed            (vision traced, row 6)
#  row 7  ·          ·          ·              ·            (spare)
#
#   vision out (6,0) → prefill in (0,0) : same col 0  (socket)
#   prefill  (r,c)   → denoise (r,3)     : same row r  (socket; KV migration)
#   prefill snake chain + denoise p2p    : collinear    (in-trace)
TRACED_VISION_SHAPE: Tuple[int, int] = (1, 4)
TRACED_VISION_OFFSET: Tuple[int, int] = (6, 0)  # row 6; chain embed(6,3)→tail(6,0)
TRACED_PREFILL_SHAPE: Tuple[int, int] = (6, 3)
TRACED_PREFILL_OFFSET: Tuple[int, int] = (0, 0)  # rows 0–5, cols 0–2; row r = VLM layers 3r..3r+2
TRACED_DENOISE_SHAPE: Tuple[int, int] = (6, 1)
TRACED_DENOISE_OFFSET: Tuple[int, int] = (0, 3)  # rows 0–5, col 3


def prefill_snake_order(rows: int = 6, cols: int = 3) -> List[Tuple[int, int]]:
    """Boustrophedon chain order over the (rows,cols) prefill block. Consecutive
    pairs are collinear: same-row within a row, same-col on the turn to the next
    row — so every p2p hop is legal under FABRIC_1D."""
    order: List[Tuple[int, int]] = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cs:
            order.append((r, c))
    return order


def prefill_layer_to_coord(layer: int, cols: int = 3) -> Tuple[int, int]:
    """VLM layer index -> (row,col) in the prefill block. Row r holds layers
    3r..3r+2; the column follows the snake direction. Used to map the produced
    per-layer KV to the denoise chip in the SAME row (collinear migration)."""
    r = layer // cols
    p = layer % cols
    c = p if r % 2 == 0 else (cols - 1 - p)
    return (r, c)


@dataclass
class MeshHandles:
    """Live submesh handles returned by open_galaxy_mesh()."""

    parent: object
    trace_root: object  # (7,4) compute submesh; trace capture/replay roots here
    vision_submesh: object
    prefill_submesh: object
    denoise_submesh: object
    vision_per_chip: List[object] = field(default_factory=list)
    prefill_per_chip: List[object] = field(default_factory=list)
    denoise_per_chip: List[object] = field(default_factory=list)


@dataclass
class TracedMeshHandles:
    """Live handles for the fully-traced (collinear, FABRIC_1D) pipeline.

    Each stage runs on ONE mesh (traced); cross-stage hand-offs are collinear
    sockets between these meshes (outside the per-stage traces).
    """

    parent: object  # (8,4) torus (FABRIC_1D); kept whole so every link trains
    vision_mesh: object  # (1,4) @ (6,0)
    prefill_mesh: object  # (6,3) @ (0,0)
    denoise_mesh: object  # (6,1) @ (0,3)


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
