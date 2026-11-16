# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill mesh layouts. The SP x TP choice is an OPEN DECISION, kept in this one place.

The prefill runner shards attention heads over TP and the sequence (and the block-cyclic KV cache)
over SP. On the 32-chip SC1 Galaxy that gives two candidates:

  * ``glm_prefill_sp8_tp4`` (default): GLM-5.3's prefill layout. SP 8 x TP 4, 64 / 4 = 16 heads per
    chip. Every prefill kernel Hy4 reuses runs on GLM-5.3 in exactly this layout.
  * ``decode_aligned_sp4_tp8``: the decode plan's head split (#65885: heads over the 8 columns, 8
    heads per chip). Would make prefill weight shards match decode's, but no prefill kernel has run
    with 8 heads per chip (sparse_sdpa's head-count constraint is the open item in #65887 / #65921).
    Listed so the decision can be tried by flipping one env var, not validated.

Selection: ``PREFILL_HY4_LAYOUT`` names a layout. The runner still takes the mesh from
``PREFILL_SP`` / ``PREFILL_TP`` / ``PREFILL_SP_AXIS`` (set together in the manifest); when a layout is
named, the adapter rejects a mesh that disagrees with it. With no layout named, any mesh runs (smaller
test meshes) and the adapter only logs which layout, if any, it matches.

The weight cache is keyed by the mesh (``{sp}x{tp}`` under the cache root), so the two layouts never
share cache files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

LAYOUT_ENV = "PREFILL_HY4_LAYOUT"


@dataclass(frozen=True)
class Hy4PrefillLayout:
    name: str
    sp: int
    tp: int
    sp_axis: int
    validated: bool
    note: str

    def heads_per_chip(self, num_heads: int) -> int:
        if num_heads % self.tp:
            raise ValueError(f"{num_heads} heads do not split over TP={self.tp}")
        return num_heads // self.tp

    def matches(self, sp: int, tp: int, sp_axis: int) -> bool:
        return (self.sp, self.tp, self.sp_axis) == (sp, tp, sp_axis)


HY4_PREFILL_LAYOUTS = {
    layout.name: layout
    for layout in (
        Hy4PrefillLayout(
            name="glm_prefill_sp8_tp4",
            sp=8,
            tp=4,
            sp_axis=0,
            validated=True,
            note="GLM-5.3 prefill layout: 16 heads per chip",
        ),
        Hy4PrefillLayout(
            name="decode_aligned_sp4_tp8",
            sp=4,
            tp=8,
            sp_axis=0,
            validated=False,
            note="decode head split (#65885): 8 heads per chip; untested in prefill",
        ),
    )
}
DEFAULT_LAYOUT = "glm_prefill_sp8_tp4"


def requested_layout() -> Optional[Hy4PrefillLayout]:
    """The layout named by ``PREFILL_HY4_LAYOUT``, or None when unset."""
    name = os.environ.get(LAYOUT_ENV, "").strip()
    if not name:
        return None
    try:
        return HY4_PREFILL_LAYOUTS[name]
    except KeyError:
        raise ValueError(f"{LAYOUT_ENV}={name!r} is not a Hy4 prefill layout; valid: {sorted(HY4_PREFILL_LAYOUTS)}")


def resolve_layout(sp: int, tp: int, sp_axis: int, num_heads: int) -> Optional[Hy4PrefillLayout]:
    """Check the runner's mesh against the requested layout and return the layout it runs.

    Raises when a layout is requested and the mesh disagrees, or when the heads do not split over TP.
    Returns None for a mesh that matches no named layout (allowed only when none was requested)."""
    if num_heads % tp:
        raise ValueError(f"Hy4 has {num_heads} attention heads, which do not split over PREFILL_TP={tp}")
    want = requested_layout()
    if want is not None:
        if not want.matches(sp, tp, sp_axis):
            raise ValueError(
                f"{LAYOUT_ENV}={want.name} needs PREFILL_SP={want.sp} PREFILL_TP={want.tp} "
                f"PREFILL_SP_AXIS={want.sp_axis}, but the runner has SP={sp} TP={tp} SP_AXIS={sp_axis}"
            )
        return want
    return next((lay for lay in HY4_PREFILL_LAYOUTS.values() if lay.matches(sp, tp, sp_axis)), None)
