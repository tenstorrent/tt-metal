# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for review and verification helpers in orchestrator_steps.sh.

Covers the two helpers a review round cannot be correct without:

* ``execute_step_seed_review_state`` — the round inherits its scope, arches, and
  verification route from the solve that produced the PR. Getting that wrong means
  verifying the wrong thing on the wrong hardware.
* ``execute_step_record_review_dispositions`` — the gate that stops a reply the
  addresser never actually thought about from reaching a reviewer.
* ``execute_step_combine_verification_results`` — the deterministic backstop
  that prevents zero or malformed test counts from becoming success.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CODEGEN = Path(__file__).resolve().parents[2]
STEPS = CODEGEN / "scripts" / "issue_solver" / "orchestrator_steps.sh"


def _bash(
    snippet: str, cwd: Path, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Source the steps library and run one snippet with cwd = <wt>/tt_metal/tt-llk."""
    full_env = {**os.environ, "PATH": os.environ.get("PATH", "")}
    full_env.update(env or {})
    return subprocess.run(
        ["bash", "-c", f'source "{STEPS}"\n{snippet}'],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=full_env,
    )


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """A minimal fake worktree: <wt>/tt_metal/tt-llk with a codegen/ symlink."""
    wt = tmp_path / "wt"
    llk = wt / "tt_metal" / "tt-llk"
    llk.mkdir(parents=True)
    (llk / "codegen").symlink_to(CODEGEN)
    return wt


@pytest.fixture
def source_run(tmp_path: Path) -> Path:
    """A finished single-arch solve run dir, as the dashboard would hand us."""
    d = tmp_path / "source_run"
    d.mkdir()
    (d / "state.json").write_text(
        json.dumps(
            {
                "RUN_ID": "2026-07-31_issue_36142_d030627a",
                "RUN_MODE": "single",
                "TARGET_ARCH": "blackhole",
                "TARGET_ARCHES_JSON": ["blackhole"],
                "ISSUE_NUMBER": "36142",
                "ISSUE_TITLE": "[CI Failure] unary broadcast hangs",
                "ISSUE_BODY": "the test hangs",
                "ISSUE_LABELS": "LLK,ci-bug",
                "ISSUE_COMMENTS": "",
                "ISSUE_URL": "https://github.com/tenstorrent/tt-metal/issues/36142",
                "TEST_BACKEND": "local",
                "VERIFY_ROUTE": "metal",
                "METAL_FILTER": "LLKMeshDeviceFixture.TensixComputeSingleTileUnaryBroadcast",
            }
        ),
        encoding="utf-8",
    )
    (d / "issue_36142_analysis.md").write_text(
        "fix_layer: compute_api\nverification_required: yes\n", encoding="utf-8"
    )
    (d / "issue_36142_fix_plan.md").write_text("## Test Strategy\n", encoding="utf-8")
    return d


def _review_input(tmp_path: Path, comment_ids) -> Path:
    p = tmp_path / "review_input.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "round": 1,
                "pr": {"number": 51772},
                "actionable_threads": [{"comment_id": c} for c in comment_ids],
            }
        ),
        encoding="utf-8",
    )
    return p


def _seed_env(source_run: Path, review_input: Path, **over) -> dict:
    env = {
        "CODEGEN_SOURCE_RUN_DIR": str(source_run),
        "CODEGEN_REVIEW_INPUT": str(review_input),
        "CODEGEN_PR_NUMBER": "51772",
    }
    env.update({k: str(v) for k, v in over.items()})
    return env


def _bootstrap(wt: Path) -> dict:
    return json.loads(
        (wt / "tt_metal" / "tt-llk" / ".codegen_run_state.json").read_text(
            encoding="utf-8"
        )
    )


# ── execute_step_seed_review_state ────────────────────────────────────────────
def test_seed_inherits_issue_and_arch_from_the_source_run(
    worktree, source_run, tmp_path
):
    ri = _review_input(tmp_path, [1, 2])
    r = _bash(
        f'execute_step_seed_review_state "{worktree}"',
        worktree / "tt_metal" / "tt-llk",
        _seed_env(source_run, ri),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    state = _bootstrap(worktree)
    assert state["RUN_KIND"] == "review"
    assert state["RUN_MODE"] == "single"
    assert state["TARGET_ARCH"] == "blackhole"
    assert state["ISSUE_NUMBER"] == "36142"
    assert state["ISSUE_TITLE"] == "[CI Failure] unary broadcast hangs"
    assert state["PR_NUMBER"] == "51772"
    assert state["SOURCE_RUN_ID"] == "2026-07-31_issue_36142_d030627a"
    # A review round updates an existing PR; it must never open one.
    assert state["CREATE_PR"] == "no"
    assert state["CREATE_LOCAL_BRANCH"] == "yes"
    # Hardware verification is the point of the round.
    assert state["TEST_BACKEND"] == "local"


def test_seed_widens_to_multi_when_the_dashboard_adds_an_arch(
    worktree, source_run, tmp_path
):
    """A reviewer asking about Wormhole must flip a Blackhole-only solve to multi."""
    ri = _review_input(tmp_path, [1])
    r = _bash(
        f'execute_step_seed_review_state "{worktree}"',
        worktree / "tt_metal" / "tt-llk",
        _seed_env(source_run, ri, CODEGEN_REVIEW_ARCHES='["blackhole","wormhole"]'),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    state = _bootstrap(worktree)
    assert state["RUN_MODE"] == "multi"
    assert json.loads(state["TARGET_ARCHES"]) == ["blackhole", "wormhole"]
    assert "TARGET_ARCH" not in state


def test_setup_run_honors_codegen_logs_root_override(worktree, tmp_path):
    """Local/dev runs must not fall back to the shared dashboard archive."""
    llk = worktree / "tt_metal" / "tt-llk"
    (llk / ".codegen_run_state.json").write_text(
        json.dumps(
            {
                "RUN_MODE": "single",
                "RUN_KIND": "issue",
                "ISSUE_NUMBER": "123",
                "ISSUE_TITLE": "storage contract",
                "ISSUE_BODY": "",
                "ISSUE_LABELS": "",
                "ISSUE_COMMENTS": "",
                "ISSUE_URL": "https://example.test/issues/123",
                "WORKTREE_BRANCH": "test",
                "TEST_BACKEND": "ttsim",
                "CREATE_LOCAL_BRANCH": "no",
                "CREATE_PR": "no",
                "TARGET_ARCH": "blackhole",
            }
        ),
        encoding="utf-8",
    )
    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir()
    (fake_scripts / "state.py").symlink_to(CODEGEN / "scripts" / "state.py")
    # Session discovery belongs to a real runner and is unrelated to storage.
    (fake_scripts / "session_cost.py").symlink_to("/dev/null")
    logs_root = tmp_path / "logs"

    r = _bash(
        f'_ORCH_SCRIPTS="{fake_scripts}"\nexecute_step_setup_run',
        llk,
        {"CODEGEN_LOGS_ROOT": str(logs_root)},
    )
    assert r.returncode == 0, r.stdout + r.stderr

    log_dir = Path(_bootstrap(worktree)["LOG_DIR"])
    assert _bootstrap(worktree)["RUN_ID"]
    state = json.loads((log_dir / "state.json").read_text(encoding="utf-8"))
    assert _bootstrap(worktree)["RUN_ID"] == state["RUN_ID"]
    assert log_dir.parent == logs_root / "issue_solver"
    assert state["CODEGEN_LOGS_ROOT"] == str(logs_root)
    assert state["DASHBOARD_PROJECT_ID"] == "issue_solver"


@pytest.mark.parametrize(
    "drop", ["CODEGEN_SOURCE_RUN_DIR", "CODEGEN_REVIEW_INPUT", "CODEGEN_PR_NUMBER"]
)
def test_seed_rejects_a_missing_required_input(worktree, source_run, tmp_path, drop):
    env = _seed_env(source_run, _review_input(tmp_path, [1]))
    env[drop] = ""
    r = _bash(
        f'execute_step_seed_review_state "{worktree}"',
        worktree / "tt_metal" / "tt-llk",
        env,
    )
    assert r.returncode != 0
    assert "REJECT" in r.stdout + r.stderr


def test_setup_review_run_requires_the_analysis_artifact(
    worktree, source_run, tmp_path
):
    """A fix plan alone cannot provide a verification route."""
    (source_run / "issue_36142_analysis.md").unlink()
    log_dir = tmp_path / "review_log"
    log_dir.mkdir()
    llk = worktree / "tt_metal" / "tt-llk"
    (llk / ".codegen_run_state.json").write_text(
        json.dumps(
            {
                "LOG_DIR": str(log_dir),
                "SOURCE_RUN_DIR": str(source_run),
                "PR_NUMBER": "51772",
                "PR_HEAD_SHA": "abc",
                "REVIEW_INPUT": "input.json",
                "SOURCE_RUN_ID": "source-1",
            }
        ),
        encoding="utf-8",
    )
    (log_dir / "state.json").write_text(
        json.dumps(
            {
                "LOG_DIR": str(log_dir),
                "WORKTREE_DIR": str(worktree),
                "SOURCE_RUN_DIR": str(source_run),
                "ISSUE_NUMBER": "36142",
            }
        ),
        encoding="utf-8",
    )
    r = _bash("execute_step_setup_review_run", llk)
    assert r.returncode != 0
    assert "missing issue_36142_analysis.md" in r.stdout + r.stderr


# ── execute_step_combine_verification_results ────────────────────────────────
def _combine_case(tmp_path, worktree, suite_results, route="llk"):
    log_dir = tmp_path / "combine-log"
    log_dir.mkdir()
    state = {
        "LOG_DIR": str(log_dir),
        "VERIFY_ROUTE": route,
        "TARGET_ARCHES_JSON": ["blackhole"],
    }
    (log_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    llk = worktree / "tt_metal" / "tt-llk"
    (llk / ".codegen_run_state.json").write_text(
        json.dumps({"LOG_DIR": str(log_dir)}), encoding="utf-8"
    )
    (log_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "arch_results": {"blackhole": {"suite_results": suite_results}},
            }
        ),
        encoding="utf-8",
    )
    result = _bash("execute_step_combine_verification_results", llk)
    return result, json.loads((log_dir / "run.json").read_text())


def test_combine_accepts_only_terminal_nonzero_complete_success(tmp_path, worktree):
    result, run = _combine_case(
        tmp_path,
        worktree,
        {
            "llk": {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": 3,
                "tests_passed": 3,
                "obstacle": None,
            }
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    combined = run["arch_results"]["blackhole"]
    assert combined["verdict"] == "SUCCESS"
    assert combined["tests_total"] == 3
    assert combined["tests_passed"] == 3


@pytest.mark.parametrize(
    ("suite", "reason"),
    [
        (
            {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": 0,
                "tests_passed": 0,
            },
            "ZERO_SELECTED",
        ),
        (
            {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": "1",
                "tests_passed": 1,
            },
            "COUNT_INVALID",
        ),
        (
            {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": 1,
                "tests_passed": 2,
            },
            "COUNT_INVALID",
        ),
        (
            {
                "status": "running",
                "verdict": "SUCCESS",
                "tests_total": 1,
                "tests_passed": 1,
            },
            "RESULT_NOT_TERMINAL",
        ),
    ],
)
def test_combine_rejects_false_success_contracts(tmp_path, worktree, suite, reason):
    result, run = _combine_case(tmp_path, worktree, {"llk": suite})
    assert result.returncode == 0, result.stdout + result.stderr
    combined = run["arch_results"]["blackhole"]
    assert combined["verdict"] == "ENV_ERROR"
    assert reason in combined["obstacle"]


def test_combine_requires_every_suite_to_succeed(tmp_path, worktree):
    result, run = _combine_case(
        tmp_path,
        worktree,
        {
            "llk": {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": 2,
                "tests_passed": 2,
            },
            "metal": {
                "status": "done",
                "verdict": "SUCCESS",
                "tests_total": 0,
                "tests_passed": 0,
            },
        },
        route="both",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    combined = run["arch_results"]["blackhole"]
    assert combined["verdict"] == "ENV_ERROR"
    assert "metal: ZERO_SELECTED" in combined["obstacle"]


def test_combine_preserves_valid_candidate_failure(tmp_path, worktree):
    result, run = _combine_case(
        tmp_path,
        worktree,
        {
            "llk": {
                "status": "done",
                "verdict": "TESTS_FAILED",
                "tests_total": 1,
                "tests_passed": 0,
                "obstacle": "assertion failed",
            }
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    combined = run["arch_results"]["blackhole"]
    assert combined["verdict"] == "TESTS_FAILED"
    assert combined["tests_total"] == 1
    assert combined["tests_passed"] == 0


# ── execute_step_record_review_dispositions ───────────────────────────────────
def _disposition_case(tmp_path, worktree, threads, actionable=(101, 202)):
    """Write a run state + dispositions file and run the validator."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    ri = _review_input(tmp_path, list(actionable))
    (log_dir / "state.json").write_text(
        json.dumps(
            {
                "LOG_DIR": str(log_dir),
                "REVIEW_INPUT": str(ri),
                "RUN_KIND": "review",
            }
        ),
        encoding="utf-8",
    )
    llk = worktree / "tt_metal" / "tt-llk"
    (llk / ".codegen_run_state.json").write_text(
        json.dumps({"LOG_DIR": str(log_dir)}), encoding="utf-8"
    )
    (log_dir / "review_dispositions.json").write_text(
        json.dumps({"version": 1, "threads": threads}), encoding="utf-8"
    )
    # run.json must exist for the `rj metric` patch at the end of the helper.
    (log_dir / "run.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    return _bash("execute_step_record_review_dispositions 600", llk), log_dir


def test_dispositions_accepted_when_every_actionable_thread_is_answered(
    tmp_path, worktree
):
    r, log_dir = _disposition_case(
        tmp_path,
        worktree,
        [
            {
                "comment_id": 101,
                "action": "changed",
                "reply": "Shared the constexpr across all five sites.",
            },
            {
                "comment_id": 202,
                "action": "no_change",
                "reply": "The unpacker already clears SrcB on that path.",
            },
        ],
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (
        json.loads((log_dir / "run.json").read_text())["review_dispositions"]["count"]
        == 2
    )


def test_dispositions_rejected_when_a_thread_is_unanswered(tmp_path, worktree):
    """The defect this gate exists for: a silent fallback to a generic reply."""
    r, _ = _disposition_case(
        tmp_path,
        worktree,
        [
            {"comment_id": 101, "action": "changed", "reply": "Done."},
        ],
    )
    assert r.returncode != 0
    assert "202" in r.stdout + r.stderr


@pytest.mark.parametrize(
    "bad,expect",
    [
        (
            {
                "comment_id": 101,
                "action": "changed",
                "reply": "Addressed in 6373e5439f7.",
            },
            "sha",
        ),
        ({"comment_id": 101, "action": "changed", "reply": "x" * 601}, "limit"),
        ({"comment_id": 101, "action": "maybe", "reply": "Not sure."}, "action"),
    ],
)
def test_dispositions_reject_a_reply_that_breaks_the_contract(
    tmp_path, worktree, bad, expect
):
    """The dashboard owns commit attribution, and a reply is one short paragraph."""
    r, _ = _disposition_case(
        tmp_path,
        worktree,
        [bad, {"comment_id": 202, "action": "no_change", "reply": "Already covered."}],
    )
    assert r.returncode != 0
    assert expect in (r.stdout + r.stderr).lower()


def test_dispositions_ignore_ids_that_are_not_actionable(tmp_path, worktree):
    """Review summaries have no reply target; an extra id must not fail the round."""
    r, _ = _disposition_case(
        tmp_path,
        worktree,
        [
            {"comment_id": 101, "action": "changed", "reply": "Split the helper out."},
            {"comment_id": 202, "action": "no_change", "reply": "Already covered."},
            {
                "comment_id": 999,
                "action": "no_change",
                "reply": "Overview, nothing to do.",
            },
        ],
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_dispositions_reject_duplicate_actionable_ids(tmp_path, worktree):
    r, _ = _disposition_case(
        tmp_path,
        worktree,
        [
            {"comment_id": 101, "action": "changed", "reply": "First answer."},
            {"comment_id": 101, "action": "changed", "reply": "Second answer."},
            {"comment_id": 202, "action": "no_change", "reply": "Already covered."},
        ],
    )
    assert r.returncode != 0
    assert "duplicate disposition" in r.stdout + r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
