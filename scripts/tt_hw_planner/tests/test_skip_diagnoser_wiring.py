"""Tests for skip_diagnoser ↔ iter-loop wiring (2026-06-03).

Covers:
  * `_run_skip_diagnoser_at_loop_end` — gate behavior and persistence
  * `_final_outcome_banner` surfacing of skip_diagnosis.json
"""

from __future__ import annotations

import json
from pathlib import Path


def test_run_skip_diagnoser_skips_when_disabled(tmp_path, monkeypatch):
    """When the gate `enabled=False` is passed, the wiring must NOT
    invoke the LLM and must NOT write skip_diagnosis.json."""
    from scripts.tt_hw_planner._cli_helpers import auto_iterate

    spawned = []

    def fake_diagnose(**kwargs):
        spawned.append(True)
        return [{"component": "x", "verdict": "fixed", "summary": "", "agent_stdout": "", "rc": 0}]

    monkeypatch.setattr(
        "scripts.tt_hw_planner._cli_helpers.skip_diagnoser.diagnose_skips_in_demo",
        fake_diagnose,
    )

    auto_iterate._run_skip_diagnoser_at_loop_end(
        demo_dir=tmp_path,
        harness_skipped={"x"},
        skip_reasons={"x": "harness reason"},
        agent_bin="/usr/bin/claude",
        enabled=False,
    )

    assert spawned == [], "must not spawn diagnoser when enabled=False"
    assert not (tmp_path / "skip_diagnosis.json").exists()


def test_run_skip_diagnoser_skips_when_no_agent_bin(tmp_path, monkeypatch):
    """No agent_bin → no LLM call. Common when the user runs without
    --auto or before authenticating the CLI."""
    from scripts.tt_hw_planner._cli_helpers import auto_iterate

    spawned = []

    def fake_diagnose(**kwargs):
        spawned.append(True)
        return []

    monkeypatch.setattr(
        "scripts.tt_hw_planner._cli_helpers.skip_diagnoser.diagnose_skips_in_demo",
        fake_diagnose,
    )

    auto_iterate._run_skip_diagnoser_at_loop_end(
        demo_dir=tmp_path,
        harness_skipped={"x"},
        skip_reasons={"x": "harness reason"},
        agent_bin=None,
    )

    assert spawned == []
    assert not (tmp_path / "skip_diagnosis.json").exists()


def test_run_skip_diagnoser_skips_when_no_harness_skipped(tmp_path, monkeypatch):
    """Empty harness_skipped → no LLM call (healthy run)."""
    from scripts.tt_hw_planner._cli_helpers import auto_iterate

    spawned = []

    def fake_diagnose(**kwargs):
        spawned.append(True)
        return []

    monkeypatch.setattr(
        "scripts.tt_hw_planner._cli_helpers.skip_diagnoser.diagnose_skips_in_demo",
        fake_diagnose,
    )

    auto_iterate._run_skip_diagnoser_at_loop_end(
        demo_dir=tmp_path,
        harness_skipped=set(),
        skip_reasons={},
        agent_bin="/usr/bin/claude",
    )

    assert spawned == []
    assert not (tmp_path / "skip_diagnosis.json").exists()


def test_run_skip_diagnoser_persists_verdicts(tmp_path, monkeypatch, capsys):
    """When the LLM diagnoser runs, results must be persisted to
    skip_diagnosis.json and a summary line printed."""
    from scripts.tt_hw_planner._cli_helpers import auto_iterate

    def fake_diagnose(**kwargs):
        return [
            {"component": "a", "verdict": "fixed", "summary": "x", "agent_stdout": "", "rc": 0},
            {"component": "b", "verdict": "manual", "summary": "y", "agent_stdout": "", "rc": 0},
            {"component": "c", "verdict": "fixed", "summary": "z", "agent_stdout": "", "rc": 0},
        ]

    monkeypatch.setattr(
        "scripts.tt_hw_planner._cli_helpers.skip_diagnoser.diagnose_skips_in_demo",
        fake_diagnose,
    )

    auto_iterate._run_skip_diagnoser_at_loop_end(
        demo_dir=tmp_path,
        harness_skipped={"a", "b", "c"},
        skip_reasons={"a": "r1", "b": "r2", "c": "r3"},
        agent_bin="/usr/bin/claude",
    )

    diag = tmp_path / "skip_diagnosis.json"
    assert diag.is_file()
    data = json.loads(diag.read_text())
    diagnoses = data["diagnoses"]
    verdicts = [d["verdict"] for d in diagnoses]
    assert sorted(verdicts) == sorted(["fixed", "manual", "fixed"])

    out = capsys.readouterr().out
    assert "fixed=2" in out
    assert "manual=1" in out


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
