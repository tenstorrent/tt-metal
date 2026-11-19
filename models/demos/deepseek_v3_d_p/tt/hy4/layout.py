# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill mesh layouts.

The Hy4 layout is TP32 / DP1 (``tp32``): one engine on the full Blackhole Galaxy with every weight sharded over all
32 chips once and nothing replicated per row, the SC1 unit of #65884. #65884 writes the mesh as 4x8 (4 rows x 8
columns). The fabric's torus is 8x4 (``single_bh_galaxy_torus_xy``, dims [8, 4]) and ``tp32`` runs on it as is (a
logical 4x8 view of it hung in the TP all-to-all), so #65884's 8 columns are mesh axis 0 (8 chips) and its 4 rows
are mesh axis 1 (4 chips):

  * hidden (6144) over axis 1, 4 ways: every weight that contracts over the hidden splits it there, and its other
    dimension 8 ways over axis 0;
  * heads over axis 0: 8 heads per chip in the attention projections (q_a, q_b, kv_a, kv_b, gate, o_proj; #65885),
    the intermediate dim of the shared expert and the dense layer-0 FFN (#65890);
  * the prefill residual / activations between sublayers are ``[S / 8, 6144 / 4]`` per chip: the sequence over
    axis 0, the hidden over axis 1, the same for attention, FFN / MoE and iHC (#65891, ``MeshPlan(hidden_axis=1)``),
    so nothing is reshuffled between sublayers;
  * MLA latent + index-key cache: context parallel, 128-token blocks block-cyclic over all 32 chips (#65886);
  * indexer: each chip scores its own stripe and keeps a local top-2048, one gather + merge picks the global
    top-2048; sparse attention runs per chip with all 64 heads over the rows it holds, partials merge by LSE
    (#65887);
  * routed experts EP32 (8 per chip); embedding and LM head vocab-sharded over 32 chips (#65892).

In the runner's knobs that is the sequence (SP) on axis 0 and the hidden (TP) on axis 1: ``PREFILL_SP=8
PREFILL_TP=4 PREFILL_SP_AXIS=0``, the same mesh as the dev layout below. The two differ in where the weights go,
not in the mesh: ``tp32`` puts the heads on the SP axis (``heads_over="sp"``) and shards every weight once, the dev
layout puts the heads on the TP axis and replicates attention over SP. ``PREFILL_HY4_LAYOUT`` picks one.

The shared prefill stack cannot run ``tp32`` yet (``TP32_BLOCKERS``), so selecting it fails at runner start
with the blockers and their issues. An issue that removes a blocker deletes its entry.

``glm_prefill_sp8_tp4`` is a DEV / NUMERICS layout, not a deployment candidate: GLM-5.3's prefill layout
(SP 8 x TP 4, 16 heads per chip, attention / shared-expert / dense weights replicated over the 8 SP rows). It
is kept because every prefill kernel Hy4 reuses runs in it today, so per-layer numerics (KV and index-key
PCC against the HF golden) can be brought up while the TP32 blockers are open. Select it explicitly with
``PREFILL_HY4_LAYOUT=glm_prefill_sp8_tp4``.

Selection: ``PREFILL_HY4_LAYOUT`` names a layout; unset means ``tp32``. The runner still opens its mesh from
``PREFILL_SP`` / ``PREFILL_TP`` / ``PREFILL_SP_AXIS`` (default 8 / 4 / 0); the adapter rejects a mesh that disagrees
with the selected layout. A mesh of fewer than 32 chips with no layout named is a test mesh and runs unchecked.

Weight caches are keyed by mesh and layout (``Hy4PrefillLayout.cache_key``): the dev layout keeps the bare
``{rows}x{cols}`` key, so existing 8x4 caches stay valid; ``tp32`` adds ``_tp32``, because its 2D-TP shards
differ from the dev layout's on the same 8x4 mesh. (Earlier ``tp32`` caches, built with the hidden over axis 0, sit
under other mesh keys, ``4x8_tp32`` / ``8x4_sp1_tp32``, and are never read.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

LAYOUT_ENV = "PREFILL_HY4_LAYOUT"
SC1_CHIPS = 32

# What the shared prefill stack (models/demos/deepseek_v3_d_p/tt) lacks for the SC1 placement: (what, owning issue).
# An issue that lands one deletes its entry here. Not a blocker: sparse_sdpa's multiple-of-32 head limit
# (sparse_sdpa_device_operation.cpp). ttMLA's sparse path all-to-alls heads to sequence before the op
# (reshards_heads_to_seq), so every chip attends with all 64 heads at any TP, and SC1's planned core (64 heads
# over the local KV stripe, LSE merge) does too. Context-parallel KV over all 32 chips and EP32 routed experts
# (256 / 32 = 8 per chip) already run in the dev layout, which is the same 8x4 mesh with the same SP / TP axes.
# The 2D-TP shared expert and dense layer-0 FFN (#65890) are ``TtMlp2D`` (tt/moe/tt_mlp_2d.py: hidden over TP, axis 1,
# intermediate over SP, axis 0), which the prefill block builds when the model config sets FFN_2D_TP; the tp32 layout
# does (``ffn_2d_tp`` below, ``Hy4TP32Config``).
TP32_BLOCKERS = (
    (
        "2D TP attention in the prefill block: the projections, output gate and sink are 2D TP in ttMLA2DTP "
        "2D TP attention in the prefill block: the projections, output gate and sink are 2D TP in ttMLA2DTP "
        "(tt/mla/mla_2d.py: every attention weight sharded once over all 32 chips, the hidden over TP (axis 1, 4 "
        "chips), 8 heads per chip over SP (axis 0, 8 chips), validated against HF as a module) and it takes and "
        "returns the block's residual [S/sp, hidden/tp] = [S/8, 6144/4], as TtMlp2D (#65890) does, but "
        "TtPrefillBlock still builds ttMLA (mla_class(config) with config.mla_2d_tp selects it) and ttMLA2DTP does "
        "not build the DSA indexer (#65887)",
        "#65885",
    ),
    (
        "DSA indexer on the 2D layout: TtIndexer runs on this mesh (128 index dims / TP 4 = 32 per chip, tile "
        "aligned) but shards its weights over TP only and replicates them over the 8 SP chips (indexer.py)",
        "#65885",
    ),
    (
        "DSA indexer on the 2D layout: TtIndexer runs on this mesh (128 index dims / TP 4 = 32 per chip, tile "
        "aligned) but shards its weights over TP only and replicates them over the 8 SP chips (indexer.py)",
        "#65887",
    ),
    (
        "vocab-sharded embedding over 32 chips: TtParallelEmbedding shards the hidden dim over the TP axis and "
        "replicates over the other",
        "#65892",
    ),
)

TP32_LAYOUT = "tp32"
DEV_LAYOUT = "glm_prefill_sp8_tp4"
DEFAULT_LAYOUT = TP32_LAYOUT


@dataclass(frozen=True)
class Hy4PrefillLayout:
    name: str
    # The runner knobs that open this layout's mesh: SP (the sequence) and TP (the hidden) mesh sizes, and the mesh
    # axis SP is on; TP is on the other one.
    sp: int
    tp: int
    sp_axis: int
    role: str  # "production" or "dev"
    note: str
    # Appended to the mesh key of the weight-cache dir; "" keeps the bare {rows}x{cols} key.
    cache_suffix: str = ""
    blockers: tuple = ()
    # The mesh axis the attention heads split over in the projections: "tp" (ttMLA, GLM-5.3: heads with the hidden)
    # or "sp" (2D TP, #65884: heads with the sequence, the hidden alone on TP).
    heads_over: str = "tp"
    # Shared expert and dense layer-0 FFN 2D tensor parallel (TtMlp2D: hidden over the TP axis, intermediate over the
    # SP axis, nothing replicated; #65890). The dev layout keeps GLM-5.3's TP-only modules and caches.
    ffn_2d_tp: bool = False

    @property
    def runnable(self) -> bool:
        return not self.blockers

    @property
    def hidden_axis(self) -> int:
        """The mesh axis the hidden (and the residual's hidden) splits over: the TP axis."""
        return 1 - self.sp_axis

    def heads_per_chip(self, num_heads: int) -> int:
        """Heads per chip in the attention projections (q_b's / the gate's N, o_proj's K)."""
        n = self.sp if self.heads_over == "sp" else self.tp
        if num_heads % n:
            raise ValueError(f"{num_heads} heads do not split over {self.heads_over.upper()}={n}")
        return num_heads // n

    def matches(self, sp: int, tp: int, sp_axis: int) -> bool:
        return (self.sp, self.tp, self.sp_axis) == (sp, tp, sp_axis)

    def mesh_env(self) -> str:
        return f"PREFILL_SP={self.sp} PREFILL_TP={self.tp} PREFILL_SP_AXIS={self.sp_axis}"

    def cache_key(self, mesh_key: str) -> str:
        return mesh_key + self.cache_suffix

    def blocker_message(self) -> str:
        items = "; ".join(f"({i}) {what} [{issue}]" for i, (what, issue) in enumerate(self.blockers, 1))
        issues = ", ".join(dict.fromkeys(issue for _, issue in self.blockers))
        dev = HY4_PREFILL_LAYOUTS[DEV_LAYOUT]
        return (
            f"Hy4 prefill layout {self.name} ({self.note}) cannot run on the shared prefill stack yet; blocked by "
            f"{issues}: {items}. For numerics bring-up select the dev layout: {LAYOUT_ENV}={dev.name} with "
            f"{dev.mesh_env()}."
        )


HY4_PREFILL_LAYOUTS = {
    layout.name: layout
    for layout in (
        Hy4PrefillLayout(
            name=TP32_LAYOUT,
            sp=8,
            tp=4,
            sp_axis=0,
            role="production",
            note="TP32 / DP1, the SC1 placement of #65884: hidden over the 4-chip axis 1, heads (8 per chip) and "
            "sequence over the 8-chip axis 0, every weight sharded once over the 8x4 mesh",
            cache_suffix="_tp32",
            blockers=TP32_BLOCKERS,
            heads_over="sp",
            ffn_2d_tp=True,
        ),
        Hy4PrefillLayout(
            name=DEV_LAYOUT,
            sp=8,
            tp=4,
            sp_axis=0,
            role="dev",
            note="DEV / NUMERICS only: GLM-5.3's prefill layout, 16 heads per chip, attention replicated over 8 SP rows",
        ),
    )
}


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
    """The layout this mesh runs: the requested one, else ``tp32`` on a full 32-chip mesh.

    Raises NotImplementedError when the layout cannot run yet (naming its blockers and their issues), and
    ValueError when the mesh disagrees with the layout or the heads do not split over TP. Returns None for
    a test mesh of fewer than 32 chips with no layout requested."""
    if num_heads % tp:
        raise ValueError(f"Hy4 has {num_heads} attention heads, which do not split over PREFILL_TP={tp}")
    layout = requested_layout()
    if layout is None:
        if sp * tp < SC1_CHIPS:
            return None
        layout = HY4_PREFILL_LAYOUTS[DEFAULT_LAYOUT]
    if not layout.runnable:
        raise NotImplementedError(layout.blocker_message())
    if not layout.matches(sp, tp, sp_axis):
        raise ValueError(
            f"Hy4 prefill layout {layout.name} needs {layout.mesh_env()}, but the runner has PREFILL_SP={sp} "
            f"PREFILL_TP={tp} PREFILL_SP_AXIS={sp_axis}"
        )
    return layout
