# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP recipe routing (Increments 3 & 4). The fracture recipe is indexed under op_class=matmul and the
CCL-knob ladder under op_class=ccl, so recall_knobs surfaces them to the agent at the right rung."""
from pathlib import Path

from agent import router

GUIDELINES = Path(__file__).parent.parent / "GUIDELINES"


def _ids_for(op_class):
    index = router.build_index(GUIDELINES)
    return {e["id"] for e in router.route(index, {"op_class": op_class})}


def test_fracture_recipe_routed_to_matmul():
    assert "tp-fracture" in _ids_for("matmul")


def test_ccl_knob_ladder_routed_to_ccl():
    assert "ccl-knobs" in _ids_for("ccl")


def test_tp_sections_are_well_formed_route_blocks():
    index = router.build_index(GUIDELINES)
    by_id = {e["id"]: e for e in index}
    assert by_id["tp-fracture"]["lever_type"] == "structural"
    assert by_id["ccl-knobs"]["lever_type"] == "knob"
