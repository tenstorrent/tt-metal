"""Tests for skip_diagnoser ↔ iter-loop wiring (2026-06-03).

Covers:
  * `_run_skip_diagnoser_at_loop_end` — gate behavior and persistence
  * `_final_outcome_banner` surfacing of skip_diagnosis.json
"""

from __future__ import annotations

import json
from pathlib import Path


def test_harness_skip_verdicts_are_persisted_for_the_banner(tmp_path, monkeypatch):
    """REPOINTED. The loop-end wrapper this used to call
    (auto_iterate._run_skip_diagnoser_at_loop_end) went with the retired engine; the live producer of
    the same two artifacts is the MCP tool bringup_mcp.mark_harness_skipped, whose own docstring says
    it writes "harness_skipped.json + skip_diagnosis.json (the SAME artifacts the fsm loop's
    skip_diagnoser writes, which the OUTCOME banner surfaces)".

    What is still worth pinning is the CONTRACT the banner depends on: every marked component ends up
    in skip_diagnosis.json with a verdict, and harness_skipped.json lists it. The gating tests that
    sat beside this one were deleted rather than repointed -- they asserted that an LLM spawn is
    suppressed when disabled or when no agent binary is present, and no spawn happens at loop end any
    more, so there is nothing left to suppress.
    """
    import importlib

    monkeypatch.setenv("TT_HW_PLANNER_DEMO_DIR", str(tmp_path))
    mcp = importlib.import_module("scripts.tt_hw_planner.bringup_mcp")
    monkeypatch.setattr(mcp, "_DEMO_DIR", tmp_path, raising=False)

    for comp, verdict in (("a", "manual"), ("b", "manual"), ("c", "manual")):
        mcp.mark_harness_skipped(comp, verdict=verdict, reason="uncallable submodule")

    diag = tmp_path / "skip_diagnosis.json"
    hs = tmp_path / "harness_skipped.json"
    assert diag.is_file() and hs.is_file()

    diagnoses = json.loads(diag.read_text())["diagnoses"]
    assert {d["component"] for d in diagnoses} == {"a", "b", "c"}
    assert all(d.get("verdict") for d in diagnoses)
    assert json.loads(hs.read_text())["harness_skipped_components"] == ["a", "b", "c"]


def test_marking_the_same_component_twice_does_not_duplicate_it(tmp_path, monkeypatch):
    """The banner counts components; a duplicate entry would inflate the count."""
    import importlib

    mcp = importlib.import_module("scripts.tt_hw_planner.bringup_mcp")
    monkeypatch.setattr(mcp, "_DEMO_DIR", tmp_path, raising=False)

    mcp.mark_harness_skipped("a", verdict="manual", reason="r")
    mcp.mark_harness_skipped("a", verdict="manual", reason="r")
    assert json.loads((tmp_path / "harness_skipped.json").read_text())["harness_skipped_components"] == ["a"]


# ─── OUTCOME banner surfaces skip_diagnosis.json ─────────────────────


def test_outcome_banner_surfaces_skip_diagnosis(tmp_path, capsys):
    """When skip_diagnosis.json exists in the demo_dir, the OUTCOME
    banner must summarize the verdicts."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    (tmp_path / "skip_diagnosis.json").write_text(
        json.dumps(
            {
                "diagnoses": [
                    {"component": "conformer_layer", "verdict": "fixed", "summary": "added [0]"},
                    {"component": "decoder", "verdict": "manual", "summary": "needs human"},
                    {"component": "encoder", "verdict": "decompose", "summary": "break apart"},
                ]
            }
        )
    )

    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )

    out = capsys.readouterr().out
    assert "SKIP-DIAGNOSER" in out
    assert "fixed" in out and "manual" in out and "decompose" in out
    assert "conformer_layer" in out
    # When there are "fixed" verdicts, must prompt user to re-run.
    assert "Re-run" in out or "re-run" in out


def test_outcome_banner_no_section_when_no_diagnosis_file(tmp_path, capsys):
    """Healthy runs (no skip_diagnosis.json) shouldn't show the section."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "SKIP-DIAGNOSER" not in out


def test_outcome_banner_handles_malformed_diagnosis_file(tmp_path, capsys):
    """Malformed file must NOT crash the banner."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    (tmp_path / "skip_diagnosis.json").write_text("invalid json")

    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "SKIP-DIAGNOSER" not in out
