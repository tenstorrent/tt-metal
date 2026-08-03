# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the review-round step helpers in orchestrator_steps.sh.

Covers the two helpers a review round cannot be correct without:

* ``execute_step_seed_review_state`` — the round inherits its scope, arches, and
  verification route from the solve that produced the PR. Getting that wrong means
  verifying the wrong thing on the wrong hardware.
* ``execute_step_record_review_dispositions`` — the gate that stops a reply the
  addresser never actually thought about from reaching a reviewer.
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
