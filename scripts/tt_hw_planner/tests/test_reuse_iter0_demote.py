"""REUSE gets one iter-0 shot, then is demoted to ADAPT on gate failure.

A REUSE component emits a stub + PCC test (like everyone) and goes through the
per-component gate. If it doesn't graduate after that first attempt (PCC fail or
the stub is still a torch wrapper), the gate re-tags it REUSE->ADAPT so the ADAPT
refine path takes over. A genuine drop-in that passes iter-0 stays REUSE.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _mcp(demo: Path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(demo))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "org/m")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(demo / ".bringup_cc_state.json"))
    mod = importlib.import_module("scripts.tt_hw_planner.bringup_mcp")
    importlib.reload(mod)  # re-read _DEMO_DIR from env
    return mod


def _write_status(demo: Path, comps):
    (demo / "bringup_status.json").write_text(json.dumps({"components": comps}))


def test_reuse_demoted_to_adapt_on_failure(tmp_path, monkeypatch):
    demo = tmp_path / "demo"
    demo.mkdir()
    _write_status(
        demo,
        [
            {"name": "reuse_a", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"},
            {"name": "new_b", "status": "NEW"},
        ],
    )
    mcp = _mcp(demo, monkeypatch)

    assert mcp._demote_reuse_to_adapt("reuse_a") is True
    out = {c["name"]: c["status"] for c in json.loads((demo / "bringup_status.json").read_text())["components"]}
    assert out["reuse_a"] == "ADAPT"  # demoted
    assert out["new_b"] == "NEW"  # untouched


def test_demote_is_noop_for_non_reuse_or_already_adapt(tmp_path, monkeypatch):
    demo = tmp_path / "demo"
    demo.mkdir()
    _write_status(demo, [{"name": "x", "status": "NEW"}, {"name": "y", "status": "ADAPT"}])
    mcp = _mcp(demo, monkeypatch)
    assert mcp._demote_reuse_to_adapt("x") is False
    assert mcp._demote_reuse_to_adapt("y") is False
    assert mcp._demote_reuse_to_adapt("missing") is False
