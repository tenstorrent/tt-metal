# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Safety and execution tests for the live agentic-coding harness."""

from __future__ import annotations

import json
from pathlib import Path

from models.autoports.meta_models_muse_glimmer_30b.tests.tool_calling_harness import _make_fixture, execute_tool


def test_fixture_starts_failing_then_can_be_repaired(tmp_path: Path):
    _make_fixture(tmp_path)
    before = json.loads(execute_tool(tmp_path, "run_tests", {}))
    assert before["exit_code"] != 0

    execute_tool(
        tmp_path,
        "write_file",
        {
            "path": "math_utils.py",
            "content": "def add(left: int, right: int) -> int:\n    return left + right\n",
        },
    )
    after = json.loads(execute_tool(tmp_path, "run_tests", {}))
    assert after["exit_code"] == 0


def test_read_file_cannot_escape_workspace(tmp_path: Path, expect_error):
    _make_fixture(tmp_path)
    with expect_error(ValueError, "escapes workspace"):
        execute_tool(tmp_path, "read_file", {"path": "../secret.py"})


def test_write_file_cannot_create_an_unexpected_file(tmp_path: Path, expect_error):
    _make_fixture(tmp_path)
    with expect_error(ValueError, "refusing to create"):
        execute_tool(tmp_path, "write_file", {"path": "new.py", "content": ""})


def test_only_python_source_files_are_accessible(tmp_path: Path, expect_error):
    _make_fixture(tmp_path)
    with expect_error(ValueError, "only Python"):
        execute_tool(tmp_path, "read_file", {"path": "notes.txt"})


def test_unknown_tools_are_rejected(tmp_path: Path, expect_error):
    _make_fixture(tmp_path)
    with expect_error(ValueError, "unknown tool"):
        execute_tool(tmp_path, "run_command", {"command": "true"})
