# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 mesh layout: SP = mesh axis 0 (sequence), TP = mesh axis 1 (hidden and heads).

Every module derives its split from ``mesh_device.shape``; this module states which (sp, tp) the V4.1 layer can run
on and rejects the rest at construction, so any mesh the checks accept (LoudBox 2x4 / 4x2, Galaxy 8x4 / 4x8) needs
no layout-specific code. The requirements, with their owners:

* hidden / tp in whole tiles: row-parallel projections, distributed norms, mHC streams, Engram, DSpark;
* heads divisible by tp: ``wq_b`` column-parallel over heads, ``nlp_create_qkv_heads`` on H/tp heads per chip;
* heads a multiple of 32: after the head->sequence all-to-all every chip attends with all heads and
  ``sparse_sdpa`` needs a multiple of 32 heads per chip;
* output groups divisible by tp: the grouped low-rank output projection keeps whole groups per chip;
* routed experts divisible by the chip count: ``TtMoe`` places ``experts // chips`` experts per chip (floors);
* chunk / (sp * tp) query rows in whole tiles: the head->sequence all-to-all and the indexer's query split give each
  chip its contiguous S/(sp*tp) queries. The SP-only rule (ratio-2 groups and tiles per SP rank) is the cache
  geometry's (``cache.V41CacheGeometry``).
"""

from dataclasses import dataclass

SP_AXIS, TP_AXIS = 0, 1
TILE = 32
SDPA_HEAD_MULTIPLE = 32  # sparse_sdpa heads per chip


@dataclass(frozen=True)
class V41MeshLayout:
    sp: int
    tp: int

    @classmethod
    def of(cls, mesh_device) -> "V41MeshLayout":
        shape = tuple(mesh_device.shape)
        assert len(shape) == 2, f"V4.1 needs a 2D (SP, TP) mesh, got shape {shape}"
        return cls(*shape)

    @property
    def chips(self) -> int:
        return self.sp * self.tp

    def check_model(self, config) -> None:
        """Reject a mesh the V4.1 layer's dims cannot be split over."""
        tp, name = self.tp, f"{self.sp}x{self.tp}"
        hidden, heads, groups = config.EMB_SIZE, config.NUM_ATTENTION_HEADS, config.O_GROUPS
        assert hidden % (TILE * tp) == 0, f"mesh {name}: hidden {hidden} does not split over TP={tp} in whole tiles"
        assert heads % tp == 0, f"mesh {name}: {heads} attention heads do not split over TP={tp}"
        assert (
            heads % SDPA_HEAD_MULTIPLE == 0
        ), f"mesh {name}: sparse_sdpa needs a multiple of {SDPA_HEAD_MULTIPLE} heads per chip, got {heads}"
        assert groups % tp == 0, f"mesh {name}: {groups} output groups do not split over TP={tp}"
        experts = config.NUM_ROUTED_EXPERTS
        assert experts % self.chips == 0, f"mesh {name}: {experts} routed experts do not split over {self.chips} chips"

    def check_chunk(self, chunk: int) -> None:
        """Reject a chunk whose per-chip query rows (chunk / (sp * tp)) are not whole tiles."""
        align = TILE * self.chips
        assert chunk % align == 0, f"chunk {chunk} must be a multiple of 32*sp*tp = {align} (query split)"
