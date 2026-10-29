# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``execute_step_import_resume`` in orchestrator_steps.sh.

A continued attempt (the dashboard's Continue, or an automatic retry of a label
solve) keeps the failed run's analysis and fix plan and skips the solve session;
setup_worktree has already applied its change. Anything else solves normally.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

CODEGEN = Path(__file__).resolve().parents[2]
STEPS = CODEGEN / "scripts" / "issue_solver" / "orchestrator_steps.sh"


def _bash(
    snippet: str, cwd: Path, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Source the steps library and run one snippet with cwd = <wt>/tt_metal/tt-llk."""
    return subprocess.run(
        ["bash", "-c", f'source "{STEPS}"\n{snippet}'],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


@pytest.fixture
def attempt(tmp_path: Path):
    """A new attempt's worktree and log dir, and the failed run it may continue."""
    llk = tmp_path / "wt" / "tt_metal" / "tt-llk"
    (llk / "codegen" / "artifacts").mkdir(parents=True)
    log = tmp_path / "log"
    log.mkdir()
    (log / "state.json").write_text(
        json.dumps({"LOG_DIR": str(log), "ISSUE_NUMBER": "42"})
    )
    (log / "run.json").write_text(
        json.dumps(
            {
                "run_id": "new",
                "status": "running",
                "current_step": "setup",
                "steps_completed": [],
                "step_history": [],
            }
        )
    )
    (llk / ".codegen_run_state.json").write_text(json.dumps({"LOG_DIR": str(log)}))
    source = tmp_path / "source"
    source.mkdir()
    (source / "issue_42_analysis.md").write_text("in_scope: true\n")
    (source / "issue_42_fix_plan.md").write_text("## Test Strategy\n")
    return llk, log, source


def _resumed(llk: Path, run_id: str = "2026-10-28_issue_42_source") -> None:
    state = json.loads((llk / ".codegen_run_state.json").read_text())
    state["RESUMED_FROM_RUN_ID"] = run_id
    (llk / ".codegen_run_state.json").write_text(json.dumps(state))


def test_a_new_solve_is_not_a_continue(attempt):
    llk, log, source = attempt
    out = _bash(
        "execute_step_import_resume", llk, {"CODEGEN_RESUME_RUN_DIR": str(source)}
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "RESUMED=0"
    assert not list((llk / "codegen" / "artifacts").iterdir())


def test_a_continue_imports_the_analysis_and_plan_and_skips_the_solve(attempt):
    llk, log, source = attempt
    _resumed(llk)
    out = _bash(
        "execute_step_import_resume", llk, {"CODEGEN_RESUME_RUN_DIR": str(source)}
    )
    assert out.returncode == 0, out.stderr
    assert (
        "RESUMED=1 SOURCE_RUN_ID=2026-10-28_issue_42_source ARTIFACTS=2" in out.stdout
    )
    for name in ("issue_42_analysis.md", "issue_42_fix_plan.md"):
        assert (llk / "codegen" / "artifacts" / name).read_text() == (
            source / name
        ).read_text()
        assert (log / name).is_file()
    run = json.loads((log / "run.json").read_text())
    assert run["current_step"] == "solve"
    assert "Continuing from 2026-10-28_issue_42_source" in run["current_step_message"]
    assert json.loads((log / "state.json").read_text())["PREVIOUS_AGENT"] == "solve"


@pytest.mark.parametrize("missing", ["analysis", "source dir"])
def test_a_continue_without_the_analysis_solves_normally(attempt, missing):
    llk, log, source = attempt
    _resumed(llk)
    env = {"CODEGEN_RESUME_RUN_DIR": str(source)}
    if missing == "analysis":
        (source / "issue_42_analysis.md").unlink()
    else:
        env = {"CODEGEN_RESUME_RUN_DIR": ""}
    out = _bash("execute_step_import_resume", llk, env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "RESUMED=0"
    assert json.loads((log / "run.json").read_text())["current_step"] == "setup"
