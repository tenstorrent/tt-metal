from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools/ci/testing_thread_persona_session.py"
    spec = importlib.util.spec_from_file_location("testing_thread_persona_session", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_summary_contains_scenario_rows() -> None:
    mod = _load_module()
    md = mod.build_summary(
        {
            "anchor_ts": "123.45",
            "scenarios": [
                {
                    "name": "active_plan",
                    "progress_state": "fix_in_progress",
                    "defer_disable": True,
                    "fix_request_requested": False,
                }
            ],
        }
    )
    assert "Thread Persona Simulation Session" in md
    assert "active_plan" in md
