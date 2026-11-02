# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure comparison logic of compare_workflow_runs.py (no gh calls)."""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_workflow_runs as cwr  # noqa: E402

QB2 = "blackhole-multi-card-fast-sanity-tests / TTNN multi-device CCL [bh_quietbox_2]"
ELTWISE = "ttnn-sanity-tests / ttnn eltwise group 1 [wh_n300_civ2]"


def job(name, conclusion="success", status="completed"):
    return cwr.JobResult(name=name, conclusion=conclusion, status=status)


def run(jobs, run_id=1, branch="main", event="push", created_at="2026-10-13T00:00:00Z", conclusion="success"):
    return cwr.WorkflowRun(
        id=run_id,
        url=f"https://example.invalid/runs/{run_id}",
        workflow_name="Sanity tests",
        branch=branch,
        conclusion=conclusion,
        status="completed",
        jobs=jobs,
        workflow_id=7,
        event=event,
        created_at=created_at,
    )


def test_failed_branch_only_job_is_reported_not_dropped():
    branch = run([job(ELTWISE), job(QB2, "failure")], branch="feature", event="workflow_dispatch", conclusion="failure")
    main = run([job(ELTWISE)])
    c = cwr.compare_runs(branch, main)
    assert c["same_success"] == [ELTWISE]
    assert c["branch_only"] == [QB2]
    assert c["unmatched_failures"] == [QB2]
    assert c["new_failures"] == [] and c["same_failures"] == []


def test_branch_only_in_progress_job_counts_as_in_progress():
    branch = run([job(QB2, None, status="in_progress")], branch="feature")
    assert cwr.compare_runs(branch, None)["in_progress"] == [QB2]


def test_no_main_run_still_surfaces_failures():
    branch = run([job(ELTWISE), job(QB2, "failure")], branch="feature")
    c = cwr.compare_runs(branch, None)
    assert c["main_url"] is None
    assert c["unmatched_failures"] == [QB2]
    assert c["in_progress"] == []


@pytest.mark.parametrize(
    "main_conclusion, bucket",
    [("success", "new_failures"), ("failure", "same_failures"), ("timed_out", "same_failures")],
)
def test_history_baseline_classifies_unmatched_failure(main_conclusion, bucket):
    branch = run([job(QB2, "failure")], branch="feature")
    c = cwr.compare_runs(branch, run([job(ELTWISE)], run_id=10))
    history = [run([job(QB2, main_conclusion)], run_id=9, event="schedule", created_at="2026-10-12T20:16:47Z")]
    cwr.resolve_unmatched_failures(c, history)
    assert c[bucket] == [QB2]
    assert c["unmatched_failures"] == []
    assert c["main_history_scanned"] == 1
    assert c["history_baseline"][QB2] == {
        "main_url": "https://example.invalid/runs/9",
        "main_event": "schedule",
        "main_created_at": "2026-10-12T20:16:47Z",
        "main_conclusion": main_conclusion,
    }


def test_history_skips_runs_where_job_was_skipped_or_cancelled_and_stops_when_resolved():
    branch = run([job(QB2, "failure")], branch="feature")
    c = cwr.compare_runs(branch, run([], run_id=10))
    fetched = []

    def history():
        for r in [
            run([job(QB2, "skipped")], run_id=9),
            run([job(QB2, "cancelled")], run_id=8),
            run([job(QB2, "success")], run_id=7, event="workflow_dispatch"),
            run([job(QB2, "failure")], run_id=6),
        ]:
            fetched.append(r.id)
            yield r

    cwr.resolve_unmatched_failures(c, history())
    assert c["new_failures"] == [QB2]
    assert c["history_baseline"][QB2]["main_url"].endswith("/7")
    assert c["main_history_scanned"] == 3
    assert fetched == [9, 8, 7], "stops fetching once every unmatched job is resolved"


def test_history_leaves_unknown_jobs_unmatched():
    branch = run([job(QB2, "failure")], branch="feature")
    c = cwr.compare_runs(branch, run([], run_id=10))
    cwr.resolve_unmatched_failures(c, [run([job(ELTWISE)], run_id=9), run([job(ELTWISE)], run_id=8)])
    assert c["unmatched_failures"] == [QB2]
    assert c["history_baseline"] == {}
    assert c["main_history_scanned"] == 2


def test_gh_failure_raises_instead_of_returning_none(monkeypatch):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed to determine base repo")

    monkeypatch.setattr(cwr.subprocess, "run", fake_run)
    with pytest.raises(cwr.GhCommandError, match="base repo"):
        cwr.run_gh_command(["run", "list"])


def _run_cli(monkeypatch, capsys, branch_run, main_run, *flags):
    """Run main() in-process with gh calls faked; returns (exit code, stdout)."""
    monkeypatch.setattr(cwr, "check_prerequisites", lambda: True)
    monkeypatch.setattr(cwr, "get_latest_run", lambda workflow, branch: main_run if branch == "main" else branch_run)
    monkeypatch.setattr(cwr, "iter_main_history", lambda workflow, exclude_run_id: iter([]))
    argv = ["compare_workflow_runs.py", "--branch", "feature", "--workflows", "sanity-tests.yaml", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        cwr.main()
    return exc.value.code, capsys.readouterr().out


@pytest.mark.parametrize("flags", [(), ("--json",)], ids=["text", "json"])
@pytest.mark.parametrize(
    "branch_jobs, main_jobs, expected",
    [
        ([job(ELTWISE), job(QB2, "failure")], [job(ELTWISE)], 1),  # failed job not found on main
        ([job(ELTWISE, "failure")], [job(ELTWISE)], 1),  # new failure
        ([job(ELTWISE, "failure")], [job(ELTWISE, "failure")], 0),  # pre-existing failure
        ([job(ELTWISE)], [job(ELTWISE)], 0),  # all passing
        ([job(ELTWISE, None, "in_progress")], [job(ELTWISE)], 0),  # still running
    ],
    ids=["unmatched", "new", "same", "passing", "running"],
)
def test_cli_exit_code_is_the_same_in_text_and_json_mode(monkeypatch, capsys, flags, branch_jobs, main_jobs, expected):
    branch_run = run(branch_jobs, branch="feature", event="workflow_dispatch", conclusion="failure")
    code, out = _run_cli(monkeypatch, capsys, branch_run, run(main_jobs, run_id=10), *flags)
    assert code == expected
    if flags:
        json.loads(out)  # stdout is only the JSON document


def test_cli_gh_failure_exits_1_without_a_verdict(monkeypatch, capsys):
    def boom(args):
        raise cwr.GhCommandError(["gh", "run", "list"], "failed to determine base repo: not a git repository")

    monkeypatch.setattr(cwr, "check_prerequisites", lambda: True)
    monkeypatch.setattr(cwr, "run_gh_command", boom)
    monkeypatch.setattr(
        sys, "argv", ["compare_workflow_runs.py", "--branch", "feature", "--workflows", "sanity-tests.yaml"]
    )
    with pytest.raises(SystemExit) as exc:
        cwr.main()
    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "Safe to merge" not in captured.out + captured.err
    assert "No verdict" in captured.err
    assert "GH_REPO" in captured.err


def test_cli_text_output_reports_unmatched_failure_and_fails_verdict(monkeypatch, capsys):
    branch_run = run(
        [job(ELTWISE), job(QB2, "failure")], branch="feature", event="workflow_dispatch", conclusion="failure"
    )
    code, out = _run_cli(monkeypatch, capsys, branch_run, run([job(ELTWISE)], run_id=10))
    assert code == 1
    assert "FAILED ON BRANCH, NOT FOUND ON MAIN" in out
    assert QB2 in out
    assert "1 failures not found on main" in out
    assert "VERDICT: Failed jobs on branch not found in recent main runs" in out
    assert "Safe to merge" not in out


def test_cli_text_output_prints_the_history_baseline(monkeypatch, capsys):
    branch_run = run(
        [job(ELTWISE), job(QB2, "failure")], branch="feature", event="workflow_dispatch", conclusion="failure"
    )
    older = run([job(QB2)], run_id=5, event="schedule", created_at="2026-10-12T20:16:47Z")
    monkeypatch.setattr(cwr, "check_prerequisites", lambda: True)
    monkeypatch.setattr(
        cwr,
        "get_latest_run",
        lambda workflow, branch: run([job(ELTWISE)], run_id=10) if branch == "main" else branch_run,
    )
    monkeypatch.setattr(cwr, "iter_main_history", lambda workflow, exclude_run_id: iter([older]))
    monkeypatch.setattr(
        sys, "argv", ["compare_workflow_runs.py", "--branch", "feature", "--workflows", "sanity-tests.yaml"]
    )
    with pytest.raises(SystemExit) as exc:
        cwr.main()
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "NEW FAILURES" in out
    assert "success on main schedule run https://example.invalid/runs/5 from 2026-10-12T20:16:47Z" in out
    assert "VERDICT: New failures detected" in out
