# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
