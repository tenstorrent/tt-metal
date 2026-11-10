# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared checker-test runner."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_checker_tests as runner  # noqa: E402

CONFIG = os.path.join(runner.REPO_ROOT, ".pre-commit-config.yaml")
BOOL_TEST = "scripts/test_check_bool_from_arithmetic.py"
HELPERS_TEST = "scripts/test_check_redefined_helpers.py"
DEST_TEST = "tt_metal/tt-llk/infra/tests/test_check_dest_to_src_wait.py"


def test_one_checker_runs_only_its_suite():
    assert runner.tests_for(["scripts/check_bool_from_arithmetic.py"]) == [BOOL_TEST]
    assert runner.tests_for(["tt_metal/tt-llk/infra/check_mutex_balance.py"]) == [DEST_TEST]


def test_shared_helper_runs_both_suites_that_import_it():
    assert runner.tests_for(["scripts/precommit_common.py"]) == [BOOL_TEST, HELPERS_TEST]


def test_runner_change_runs_every_suite():
    assert runner.tests_for([runner.RUNNER]) == [test for test, _triggers in runner.CHECKS]


def test_no_paths_runs_every_suite():
    assert runner.tests_for([]) == [test for test, _triggers in runner.CHECKS]


def test_unrelated_path_runs_nothing():
    assert runner.tests_for(["tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h"]) == []


def test_absolute_path_is_repo_relative():
    path = os.path.join(runner.REPO_ROOT, "scripts/check_redefined_helpers.py")
    assert runner.tests_for([path]) == [HELPERS_TEST]


def test_main_invokes_pytest_on_the_selected_modules(monkeypatch):
    seen = {}

    def fake_call(cmd):
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr(runner.subprocess, "call", fake_call)
    assert runner.main(["scripts/precommit_common.py"]) == 0
    assert seen["cmd"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--noconftest",
        BOOL_TEST,
        HELPERS_TEST,
    ]


def test_main_skips_pytest_when_nothing_matches(monkeypatch):
    def fake_call(cmd):
        raise AssertionError(cmd)

    monkeypatch.setattr(runner.subprocess, "call", fake_call)
    assert runner.main(["README.md"]) == 0


def _hook_body():
    lines = open(CONFIG).read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- id: checker-tests")
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("- id:") or line.startswith("- repo:"):
            break
        body.append(line)
    return "\n".join(body)


def _files_regex():
    """The hook's `files` pattern, as pre-commit compiles it (block indent stripped)."""
    body = _hook_body().splitlines()
    idx = next(i for i, line in enumerate(body) if line.strip() == "files: |")
    indent = len(body[idx + 1]) - len(body[idx + 1].lstrip(" "))
    parts = []
    for line in body[idx + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip(" "))) < indent:
            break
        parts.append(line[indent:])
    return re.compile("\n".join(parts))


def test_hook_receives_every_trigger_path():
    """pass_filenames is on, and the filter names every path the runner dispatches on."""
    body = _hook_body()
    assert "entry: python3 scripts/run_checker_tests.py" in body
    assert "pass_filenames: true" in body
    pattern = _files_regex()
    for test, triggers in runner.CHECKS:
        assert pattern.search(test), test
        for path in triggers:
            assert pattern.search(path), path
    assert not pattern.search("tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h")
    assert not pattern.search("scripts/check_format_field_mask.py")
