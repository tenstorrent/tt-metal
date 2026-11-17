# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill mesh layouts.

The Hy4 layout is TP32 / DP1 (``tp32``): one engine on the full 4x8 Blackhole Galaxy with every weight sharded
over all 32 chips once and nothing replicated per row, the SC1 unit of #65884:

  * attention projections (q_a, q_b, kv_a, kv_b, gate, o_proj): 2D tensor parallel, heads over the 8 columns
    (8 heads per chip) and hidden over the 4 rows (#65885);
  * MLA latent + index-key cache: context parallel, 128-token blocks block-cyclic over all 32 chips (#65886);
  * indexer: each chip scores its own stripe and keeps a local top-2048, one gather + merge picks the global
    top-2048; sparse attention runs per chip with all 64 heads over the rows it holds, partials merge by LSE
    (#65887);
  * routed experts EP32 (8 per chip); shared expert and dense layer 0 2D TP like attention (#65890);
  * embedding and LM head vocab-sharded over 32 chips (#65892).

The shared prefill stack cannot run ``tp32`` yet (``TP32_BLOCKERS``), so selecting it fails at runner start
with the blockers and their issues. An issue that removes a blocker deletes its entry.

``glm_prefill_sp8_tp4`` is a DEV / NUMERICS layout, not a deployment candidate: GLM-5.3's prefill layout
(SP 8 x TP 4, 16 heads per chip, attention / shared-expert / dense weights replicated over the 8 SP rows). It
is kept because every prefill kernel Hy4 reuses runs in it today, so per-layer numerics (KV and index-key
PCC against the HF golden) can be brought up while the TP32 blockers are open. Select it explicitly with
``PREFILL_HY4_LAYOUT=glm_prefill_sp8_tp4``.

Selection: ``PREFILL_HY4_LAYOUT`` names a layout; unset means ``tp32``. The runner still opens its mesh from
``PREFILL_SP`` / ``PREFILL_TP`` / ``PREFILL_SP_AXIS`` (default 8 / 4 / 0, the dev mesh); the adapter rejects a
mesh that disagrees with the selected layout. A mesh of fewer than 32 chips with no layout named is a test
mesh and runs unchecked.

Weight caches are keyed by mesh and layout (``Hy4PrefillLayout.cache_key``): the dev layout keeps the bare
``{rows}x{cols}`` key, so existing 8x4 caches stay valid; ``tp32`` adds ``_tp32``, because its 2D-TP shards
differ from an SP x TP cache on the same 4x8 mesh.
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
# (256 / 32 = 8 per chip) already run in the dev layout.
TP32_BLOCKERS = (
    (
        "2D TP attention: ttMLA shards q_a / q_b / kv_a / wkv_b / o_proj / g_proj over the TP mesh axis only and "
        "replicates them over the other axis, which carries the sequence (mla.py _convert_and_cache_weights); "
        "there is no hidden split over the 4 rows. The output gate likewise all-gathers the hidden over TP before "
        "g_proj (ttMLA._output_gate); with hidden over the rows its partial x @ W_gate must be all-reduced over the "
        "rows before the sigmoid",
        "#65885",
    ),
    (
        "a 4x8 mesh on the Galaxy fabric: the torus descriptor is 8x4 (single_bh_galaxy_torus_xy, dims [8, 4]); a "
        "logical 4x8 view of it hung in the TP all-to-all after sparse_sdpa (seen on the #65921 branch)",
        "#65885",
    ),
    (
        "DSA indexer at 8-way head TP: TtIndexer asserts index_head_dim % (tp * 32) == 0 for its TP all-gather of "
        "the index keys; 128 / 8 = 16 columns per chip is not tile aligned (indexer.py)",
        "#65887",
    ),
    (
        "2D TP shared expert and dense layer-0 FFN: TtSharedExpert and TtFfn shard over the TP axis only and "
        "replicate over the other",
        "#65890",
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
    # The runner knobs that open this layout's mesh (mesh shape (sp, tp) at sp_axis 0).
    sp: int
    tp: int
    sp_axis: int
    role: str  # "production" or "dev"
    note: str
    # Appended to the mesh key of the weight-cache dir; "" keeps the bare {rows}x{cols} key.
    cache_suffix: str = ""
    blockers: tuple = ()

    @property
    def runnable(self) -> bool:
        return not self.blockers

    def heads_per_chip(self, num_heads: int) -> int:
        if num_heads % self.tp:
            raise ValueError(f"{num_heads} heads do not split over TP={self.tp}")
        return num_heads // self.tp

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
            sp=4,
            tp=8,
            sp_axis=0,
            role="production",
            note="TP32 / DP1, the SC1 placement of #65884: 2D TP attention with 8 heads per chip, every weight "
            "sharded once over the 4x8 mesh",
            cache_suffix="_tp32",
            blockers=TP32_BLOCKERS,
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
