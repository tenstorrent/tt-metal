# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The stage axis was built end to end and nothing ever travelled on it.

    router.DIMENSIONS            includes "regime"                                   built
    router.declare_stages()      called from run.py with the model's PIPELINE_STAGES  live
    recall_knobs(regime=...)     narrows the catalogue by stage                       built
    knob catalogue               has regime-tagged sections                           populated
    op -> bucket tag             "regime": "na"   # TBD(regime-source)                UNWIRED
    entry -> next_target         no regime key at all                                 MISSING

Two links, not one. recall_knobs' own docstring gives the cost: "the KV-cache section is
`op_class: attention, datamove` + `regime: decode`, so narrowing by op_class ALONE cannot find it
from a decode target." A lever written for a stage is unreachable from an op in that stage.

This file covers the SECOND link -- the target carrying what the op knows. The first (giving a
profiled op its stage at all) is still open: the profiled run executes one opaque `run_head` call, so
there is nowhere for the harness to observe a stage boundary, and the model's own method for the
`prefill` stage is named `decode_prefill`, which no convention lets the tool infer.

Until that lands this link is wired and starved, which is the correct state: it must carry a value
when one exists and stay silent when none does.
"""
import ast
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


def _entry_block() -> str:
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('        entry = {\n            "op": op_code,')
    return src[i : src.index("done, rung, reason = _op_ladder_status(", i)]


def test_the_target_carries_the_stage_when_the_op_knows_it():
    blk = _entry_block()
    assert 'entry["regime"] = _rg' in blk, "a per-op target still cannot carry a stage"


def test_it_is_absent_rather_than_na_when_the_op_does_not_know():
    """ "na" is a VALUE in that vocabulary -- it asserts "belongs to no stage". Passing it would
    narrow the search to levers tagged stage-less instead of leaving the axis open. Ops carry "na"
    today because their source is unwired, and an unwired source must read as silence."""
    blk = _entry_block()
    assert '_rg != "na"' in blk, 'the placeholder "na" would be forwarded as a real stage'
    assert 'if _rg and _rg != "na"' in blk


def test_the_knob_vs_kernel_classifier_cannot_be_mistaken_for_a_stage():
    """roofline.classify_regime answers a DIFFERENT question with the same word -- knob-reachable vs
    kernel-level. If its verdict were ever written onto an op as `regime`, this link would forward
    "knob" as though it were a stage name and quietly poison the axis."""
    src = (_PA / "agent" / "roofline.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "regime":
            raise AssertionError("roofline writes a 'regime' key onto an op -- the two meanings collide")
    assert '"regime":' not in src


def test_the_vocabulary_is_the_model_s_own_stages():
    """Not a fixed {prefill, decode, na}: a fixed set could not tag a lever for an audio encoder."""
    import agent.router as R

    R._DECLARED_STAGES.clear()
    assert R._vocab_for("regime") is None  # unknown, not empty
    R.declare_stages(["encode", "prefill", "decode"])
    v = R._vocab_for("regime")
    assert {"encode", "prefill", "decode", "na"} <= v
    R._DECLARED_STAGES.clear()


def test_declare_stages_is_actually_called_by_the_run():
    """A vocabulary nothing populates validates nothing."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "declare_stages(declared_stage_names(model_root))" in src


def test_the_first_link_is_still_marked_open():
    """The TBD is the honest record that a profiled op has no stage yet. Removing it while the source
    is still unwired would hide the remaining half of this."""
    src = (_PA / "agent" / "tracy_tool.py").read_text()
    assert "TBD(regime-source)" in src, "the marker was dropped while the source is still unwired"
