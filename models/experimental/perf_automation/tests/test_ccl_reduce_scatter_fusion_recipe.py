# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The CCL-fusion catalog must carry the reduce_scatter sibling, not only all_gather.

The GUIDELINES catalog had `all_gather_matmul` (the column-fracture / all_gather fused primitive)
but NOT its row-parallel sibling `matmul_reduce_scatter_async`. So on a row-fractured model whose
collective is a `reduce_scatter` (e.g. xtts_v2's TP=8 GPT, where all_reduce decomposes into
reduce_scatter + all_gather), the agent hit the collective's fusion rung, found no matching recipe,
and improvised the *weak* fusion — collapsing reduce_scatter+all_gather back into one all_reduce —
which is still a standalone collective per projection and did not pay (a measured no-gain).

This pins the added recipe: the `matmul_reduce_scatter_async` sibling in 08 §7, routed to `ccl`, plus
the explicit anti-pattern (don't re-merge into all_reduce).
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import router  # noqa: E402

_G = _PA / "GUIDELINES" / "08_DECODE_PREFILL_AND_MULTIDEVICE.md"


def _fracture_entry():
    """The §7 lever section the fusion recipe lives in."""
    idx = router.build_index()
    return next((e for e in idx if e["id"] == "multidevice-fracturing"), None)


# --------------------------------------------------------------------------- routing
def test_recipe_section_routes_to_ccl_and_matmul():
    e = _fracture_entry()
    assert e is not None, "the multi-device fracturing section must be indexed"
    assert "ccl" in e["op_class"], f"must route to ccl so recall_knobs(op_class='ccl') finds it: {e['op_class']}"
    assert "matmul" in e["op_class"], "it fuses a collective INTO a matmul -> both classes"
    # the route header must still be valid (no parse/vocab warnings introduced by the edit)
    assert not e.get("route_warnings"), e.get("route_warnings")


# --------------------------------------------------------------------------- recipe content
def test_reduce_scatter_fused_sibling_is_present():
    body = _G.read_text(encoding="utf-8")
    # the row-parallel fused primitive, next to the pre-existing all_gather one
    assert "matmul_reduce_scatter_async" in body, "the reduce_scatter fused sibling recipe is missing"
    assert "all_gather_matmul" in body, "the all_gather fused primitive must still be documented"
    # it must be tied to the reduce_scatter / row-fracture case, not left generic
    lo = body.lower()
    assert "reduce_scatter" in lo and "row" in lo


def test_anti_pattern_is_called_out():
    """The exact weak fusion the agent improvised (collapse to all_reduce) must be warned against."""
    lo = _G.read_text(encoding="utf-8").lower()
    i = lo.find("matmul_reduce_scatter_async")
    assert i != -1
    # somewhere in the recipe: don't just re-merge reduce_scatter+all_gather into all_reduce
    assert "all_reduce" in lo
    assert "do not" in lo or "don't" in lo, "the collapse-to-all_reduce anti-pattern must be explicit"


def test_no_other_guideline_route_broke():
    """The whole catalog still indexes cleanly (the edit didn't corrupt a route block anywhere)."""
    idx = router.build_index()
    warned = [e["id"] for e in idx if e.get("route_warnings")]
    assert not warned, f"route warnings introduced: {warned}"
