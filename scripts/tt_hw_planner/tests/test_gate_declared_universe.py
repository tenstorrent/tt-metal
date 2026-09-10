"""The gate's universe is the DECLARED component set, not the emitted-test-file set.

Regression for the "silently-unattempted component" bug: because
termination_check enumerated components via test-file existence
(_list_component_pcc_tests' `test_path.is_file()` filter), a component declared
REUSE/ADAPT in bringup_status.json whose PCC test was never written slipped out
of the gate universe — never selected as next_target AND not blocking can_stop,
so the gate reported "all graduated" over a component it never saw. Both auto-up
and promote obey this same gate, so both were affected.

The fix: _components() enumerates the declared set (minus no_emit), and
termination_check auto-emits any missing per-component test (deterministic
scaffolder) so a declared component can never stay test-less/unattended.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fresh_mcp(demo_dir: Path):
    os.environ["BRINGUP_MCP_DEMO_DIR"] = str(demo_dir)
    os.environ["BRINGUP_MCP_MODEL_ID"] = "org/m"
    os.environ.setdefault("TT_HW_PLANNER_OFFLINE", "1")
    for m in list(sys.modules):
        if m.startswith("scripts.tt_hw_planner.bringup_mcp"):
            del sys.modules[m]
    return importlib.import_module("scripts.tt_hw_planner.bringup_mcp")


def _build_demo(*, comp_b_status: str = "ADAPT") -> Path:
    d = Path(tempfile.mkdtemp()) / "models" / "demos" / "hf_eager" / "m"
    (d / "tests" / "pcc").mkdir(parents=True)
    (d / "_stubs").mkdir(parents=True)
    (d / "bringup_status.json").write_text(
        json.dumps(
            {
                "components": [
                    {"name": "comp_a", "status": "NEW", "submodule_path": "model.a"},
                    {"name": "comp_b", "status": comp_b_status, "submodule_path": "model.b"},
                ]
            }
        )
    )
    # comp_a: graduated — native stub + last_good_native snapshot + its test present
    native = "import ttnn\n\n\ndef forward(x):\n    return ttnn.rms_norm(x)\n"
    (d / "_stubs" / "comp_a.py").write_text(native)
    (d / "_stubs" / "comp_a.py.last_good_native").write_text(native)
    (d / "tests" / "pcc" / "test_comp_a.py").write_text("def test_comp_a():\n    pass\n")
    # comp_b: declared, NOT graduated, and its PCC test was NEVER emitted
    (d / "_stubs" / "comp_b.py").write_text(native)
    return d


def test_declared_testless_component_is_in_universe_blocks_and_auto_emits():
    demo = _build_demo()
    pcc = demo / "tests" / "pcc"
    assert not (pcc / "test_comp_b.py").is_file(), "precondition: comp_b starts with NO test"

    mcp = _fresh_mcp(demo)

    # 1) declaration-based universe includes the test-less component
    universe = mcp._declared_components()
    assert "comp_b" in universe and "comp_a" in universe, universe

    # 2) the gate self-heals: calling it emits the missing test
    tc = mcp.termination_check()
    assert (pcc / "test_comp_b.py").is_file(), "termination_check must auto-emit the missing PCC test"

    # 3) it is NOT done, and comp_b (the only ungraduated one) is the next target
    assert tc["can_stop"] is False
    assert tc["next_target"] and tc["next_target"]["unit"] == "comp_b", tc["next_target"]


def test_can_stop_false_even_if_emission_fails(monkeypatch):
    """Belt-and-suspenders: even if the scaffolder can't produce a file, a declared
    ungraduated component must still block can_stop (never silently 'done')."""
    demo = _build_demo()
    mcp = _fresh_mcp(demo)
    # force the auto-emit to be a no-op so the test file stays missing
    monkeypatch.setattr(mcp, "_ensure_component_tests", lambda: [])
    tc = mcp.termination_check()
    assert "comp_b" in mcp._declared_components()
    assert tc["can_stop"] is False
    assert tc["next_target"]["unit"] == "comp_b"
    # and the directive is the deterministic 'missing test' emit rung, not a run_component dead-end
    assert tc["next_target"]["rung"] == "emit"
    assert "no tests/pcc/test_comp_b.py" in tc["next_target"]["reason"]


def test_no_emit_component_stays_excluded(monkeypatch):
    """A component deliberately marked no_emit must NOT be resurrected into the
    universe — otherwise the fix would over-correct and block on intentionally
    excluded modules."""
    demo = _build_demo()
    mcp = _fresh_mcp(demo)
    from scripts.tt_hw_planner import overlay_manager

    monkeypatch.setattr(overlay_manager, "load_no_emit_tests", lambda model_id: {"comp_b": {}})

    universe = mcp._declared_components()
    assert "comp_b" not in universe and "comp_a" in universe, universe
    # comp_a is graduated and comp_b is excluded -> nothing left to do
    tc = mcp.termination_check()
    assert tc["can_stop"] is True, tc
