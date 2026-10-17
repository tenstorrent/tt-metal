# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit tests for merge-gate process helpers. No device."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_gate_process import (
    is_family_test_path,
    parse_collected_count,
    parse_expected_count,
    select_pytest_targets,
)


def test_parse_expected_count_skips_comments():
    text = "# note\n\n# another\n717\n"
    assert parse_expected_count(text) == 717


def test_parse_collected_count_fraction_form():
    output = "tests/a.py::test_one\n717/11342 tests collected (10625 deselected) in 4.15s\n"
    assert parse_collected_count(output) == 717


def test_parse_collected_count_plain_form():
    output = "===== 205 tests collected in 1.02s =====\n"
    assert parse_collected_count(output) == 205


def test_is_family_test_path_accepts_only_family_test_modules():
    assert is_family_test_path("tests/ttnn/unit_tests/operations/reduce/test_topk.py")
    assert is_family_test_path("tests/ttnn/unit_tests/operations/matmul/test_matmul.py")
    assert is_family_test_path("tests/ttnn/unit_tests/operations/fused/test_softmax.py")
    assert not is_family_test_path("tests/ttnn/unit_tests/operations/reduce/conftest.py")
    assert not is_family_test_path("tests/ttnn/unit_tests/operations/eltwise/test_unary.py")
    assert not is_family_test_path("tests/ttnn/nightly/unit_tests/operations/reduction/test_topk.py")


def test_select_added_file_runs_the_file():
    path = "tests/ttnn/unit_tests/operations/reduce/test_new.py"
    targets = select_pytest_targets([("A", path)], "", {})
    assert targets == [path]


def test_select_deleted_file_is_ignored():
    path = "tests/ttnn/unit_tests/operations/reduce/test_old.py"
    targets = select_pytest_targets([("D", path)], "", {})
    assert targets == []


def test_select_one_changed_test_function():
    path = "tests/ttnn/unit_tests/operations/reduce/test_sample.py"
    source = (
        "import pytest\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def test_alpha(device):\n"
        "    assert helper() == 1\n"
        "\n"
        "def test_beta(device):\n"
        "    assert True\n"
    )
    # test_beta body is lines 9-10 in this file (1-based).
    diff = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -10,1 +10,1 @@\n"
        "-    assert False\n"
        "+    assert True\n"
    )
    targets = select_pytest_targets([("M", path)], diff, {path: source})
    assert targets == [f"{path}::test_beta"]


def test_select_helper_edit_runs_the_whole_file():
    path = "tests/ttnn/unit_tests/operations/reduce/test_sample.py"
    source = "def helper():\n    return 2\n\ndef test_alpha(device):\n    assert helper() == 2\n"
    diff = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -2,1 +2,1 @@\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    targets = select_pytest_targets([("M", path)], diff, {path: source})
    assert targets == [path]
