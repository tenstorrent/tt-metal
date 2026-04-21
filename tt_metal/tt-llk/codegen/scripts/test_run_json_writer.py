# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_json_writer.py — dashboard-compatibility schema."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "run_json_writer.py"


def _run(log_dir, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--log-dir", str(log_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_init_emits_dashboard_fields(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "test_2026-04-17_issue_1_abcd1234",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
        "--git-branch",
        "ai-code-gen/issue-1-v1",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["git_branch"] == "ai-code-gen/issue-1-v1"
    assert "num_turns" in doc
    assert doc["num_turns"] == 0
    assert doc["tokens"] == {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "total": 0,
        "cost_usd": 0,
    }
    assert doc.get("solver_state") is None  # only set by finalize


def test_issue_url_preserved(tmp_path):
    issue = {
        "number": 1148,
        "title": "Foo",
        "url": "https://github.com/x/y/issues/1148",
        "labels": [],
    }
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1148",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--issue",
        json.dumps(issue),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue"]["url"] == "https://github.com/x/y/issues/1148"


def test_finalize_sets_solver_state(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--solver-state",
        "working",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] == "working"
    assert doc["status"] == "success"
    assert doc["final_result"] == "success"
    assert doc["final_message"] == "done"


def test_finalize_rejects_bad_solver_state(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finalize",
            "--log-dir",
            str(tmp_path),
            "--status",
            "success",
            "--final-result",
            "success",
            "--final-message",
            "x",
            "--solver-state",
            "bogus",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "bogus" in (r.stderr + r.stdout)


def test_finalize_without_solver_state_preserves_none(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] is None


def test_finalize_solver_state_wins_over_patch_json(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--solver-state",
        "working",
        "--patch-json",
        '{"solver_state": "bogus_via_patch"}',
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] == "working"


def test_finalize_computes_duration_seconds(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
        "--start-time",
        "2026-04-17T12:00:00Z",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--end-time",
        "2026-04-17T12:03:45Z",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["duration_seconds"] == 225  # 3m45s


# --------------------------------------------------------------------------
# Legacy multi-arch grouping — issue_run_id + sibling_runs
#
# Older multi-arch issue-solver runs produced N per-arch runs, each with its
# own run.json. They are grouped via an `issue_run_id` and a `sibling_runs`
# array. New multi-arch issue-solver runs use one run.json with arch="multi",
# target_arches, and arch_results; these fields stay optional for backwards
# compatibility.
# --------------------------------------------------------------------------


def test_init_without_multi_arch_fields_defaults(tmp_path):
    """Single-arch (today's default) runs get issue_run_id=None, sibling_runs=[]."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_solo",
        "--kernel",
        "issue_42",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] is None
    assert doc["sibling_runs"] == []


def test_init_accepts_issue_run_id(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--issue-run-id",
        "issue-1089-multi-abc",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] == "issue-1089-multi-abc"


def test_init_accepts_sibling_runs(tmp_path):
    siblings = [{"arch": "wormhole", "run_id": "r_wh"}]
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--sibling-runs",
        json.dumps(siblings),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["sibling_runs"] == siblings


def test_init_accepts_single_run_multi_arch_patch(tmp_path):
    """init with a multi-arch patch-json stores target_arches, arch_results, and multi_arch_run."""
    arch_results = {
        "wormhole": {"status": "pending"},
        "blackhole": {"status": "pending"},
    }
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--patch-json",
        json.dumps(
            {
                "target_arches": ["wormhole", "blackhole"],
                "arch_results": arch_results,
                "multi_arch_run": True,
            }
        ),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["arch"] == "multi"
    assert doc["target_arches"] == ["wormhole", "blackhole"]
    assert doc["arch_results"] == arch_results
    assert doc["multi_arch_run"] is True
    assert doc["issue_run_id"] is None
    assert doc["sibling_runs"] == []


def test_metric_merges_nested_arch_results(tmp_path):
    """metric deep-merges per-arch entries so updating one arch does not wipe the others."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "tester",
        "--first-message",
        "go",
        "--patch-json",
        json.dumps(
            {
                "arch_results": {
                    "wormhole": {"status": "pending", "tests_total": 0},
                    "blackhole": {"status": "pending", "tests_total": 0},
                }
            }
        ),
    )
    _run(
        tmp_path,
        "metric",
        "--patch-json",
        json.dumps(
            {
                "arch_results": {
                    "wormhole": {
                        "status": "done",
                        "verdict": "SUCCESS",
                        "tests_total": 32,
                    }
                },
                "tests_total": 32,
            }
        ),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["tests_total"] == 32
    assert doc["arch_results"]["wormhole"]["status"] == "done"
    assert doc["arch_results"]["wormhole"]["verdict"] == "SUCCESS"
    assert doc["arch_results"]["blackhole"] == {"status": "pending", "tests_total": 0}


def test_metric_accepts_dotted_keys_as_nested_compatibility(tmp_path):
    """metric expands dotted keys (e.g. arch_results.wormhole.verdict) into nested dicts."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "tester",
        "--first-message",
        "go",
    )
    _run(
        tmp_path,
        "metric",
        "--patch-json",
        '{"arch_results.wormhole.verdict": "SUCCESS"}',
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["arch_results"]["wormhole"]["verdict"] == "SUCCESS"
    assert "arch_results.wormhole.verdict" not in doc


def test_link_siblings_replaces_sibling_runs(tmp_path):
    """link-siblings patches the sibling_runs list on an existing run.json."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    siblings = [
        {"arch": "wormhole", "run_id": "r_wh"},
        {"arch": "quasar", "run_id": "r_qs"},
    ]
    _run(tmp_path, "link-siblings", "--siblings", json.dumps(siblings))
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["sibling_runs"] == siblings


def test_link_siblings_sets_issue_run_id(tmp_path):
    """link-siblings --issue-run-id sets the shared issue_run_id used for dashboard grouping."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    _run(
        tmp_path,
        "link-siblings",
        "--issue-run-id",
        "issue-1089-shared",
        "--siblings",
        "[]",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] == "issue-1089-shared"
    # link-siblings also accepts an empty siblings list (valid — single-arch
    # runs may still want to set issue_run_id for dashboard grouping).
    assert doc["sibling_runs"] == []


def test_link_siblings_preserves_other_fields(tmp_path):
    """Regression: link-siblings must not overwrite unrelated run.json state."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    # Advance so step_history has a closed entry; link-siblings must not reset it.
    _run(
        tmp_path,
        "advance",
        "--new-step",
        "planner",
        "--new-message",
        "planning",
        "--prev-result",
        "success",
    )
    _run(
        tmp_path,
        "link-siblings",
        "--siblings",
        json.dumps([{"arch": "wormhole", "run_id": "r_wh"}]),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["current_step"] == "planner"
    assert len(doc["step_history"]) == 2
    assert doc["step_history"][0]["result"] == "success"
    assert doc["sibling_runs"] == [{"arch": "wormhole", "run_id": "r_wh"}]
