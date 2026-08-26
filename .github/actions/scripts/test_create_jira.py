#!/usr/bin/env python3
"""Tests for the RTL sim check-detail parser and the relevance matcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from create_jira import format_test, match_entry, parse_failed  # noqa: E402

MAP_PATH = SCRIPTS_DIR / "ai_ip_tests.json"


@pytest.fixture(scope="module")
def relevance_map():
    return json.loads(MAP_PATH.read_text())


def test_parses_gtest_line():
    detail = "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`"
    assert parse_failed(detail) == [("1x3", "unit_tests_api", "Foo.Bar", "gtest")]


def test_parses_pytest_node_id():
    detail = "- `[2x3_DISPATCH] tests/tt_metal/tools/profiler/test_device_profiler.py::test_full_buffer`"
    assert parse_failed(detail) == [
        (
            "2x3_DISPATCH",
            "tests/tt_metal/tools/profiler/test_device_profiler.py",
            "test_full_buffer",
            "pytest",
        )
    ]


def test_parses_whole_file_pytest():
    detail = "- `[2x3] models/demos/x/test_add.py`"
    assert parse_failed(detail) == [("2x3", "models/demos/x/test_add.py", "", "pytest")]


def test_pytest_row_rendered_with_gtest_separator():
    """The sim reporter hardcodes --gtest_filter= for every runner."""
    detail = "- `[2x3] models/demos/x/test_add.py --gtest_filter=test_foo`"
    assert parse_failed(detail) == [("2x3", "models/demos/x/test_add.py", "test_foo", "pytest")]


def test_ignores_prose_and_dedups():
    detail = (
        "RTL sim: 2 test(s) failed:\n"
        "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`\n"
        "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`\n"
        "- … and 3 more (truncated)\n"
        "No RTL sim test failures were recorded.\n"
    )
    assert len(parse_failed(detail)) == 1


def test_omitted_field_is_a_wildcard():
    mapping = {"relevant_tests": [{"group": "unit_tests_api", "requirement": "R"}]}
    assert match_entry("1x3", "unit_tests_api", "Anything", "gtest", mapping)["requirement"] == "R"
    assert match_entry("1x3", "unit_tests_legacy", "Anything", "gtest", mapping) is None


def test_runner_is_matched():
    mapping = {"relevant_tests": [{"runner": "pytest", "requirement": "R"}]}
    assert match_entry("1x3", "a.py", "t", "pytest", mapping) is not None
    assert match_entry("1x3", "unit_tests_api", "Foo.Bar", "gtest", mapping) is None


def test_back2back_batch_matches_any_component():
    """select_quasar_tests.py merges back2back entries into one ':'-joined filter."""
    mapping = {"relevant_tests": [{"group": "unit_tests_legacy", "filter": "*DmLoopback*"}]}
    batch = "*SingleDmL1Write*:*DmLoopback*:*QuasarComputeKernelSingleThread*"
    assert match_entry("2x3_DISPATCH", "unit_tests_legacy", batch, "gtest", mapping) is not None
    other = "*SingleDmL1Write*:*QuasarComputeKernelSingleThread*"
    assert match_entry("2x3_DISPATCH", "unit_tests_legacy", other, "gtest", mapping) is None


def test_format_test_round_trips_into_the_parser():
    for row in [
        ("1x3", "unit_tests_api", "Foo.Bar", "gtest"),
        ("2x3", "models/demos/x/test_add.py", "test_foo", "pytest"),
        ("2x3", "models/demos/x/test_add.py", "", "pytest"),
    ]:
        rendered = format_test(*row)
        if row[3] == "pytest" and not row[2]:
            # "(whole file)" is a human label, not a parseable suffix
            assert parse_failed(rendered)[0] == row
        else:
            assert parse_failed(rendered) == [row]


def test_shipped_map_is_valid_and_ordered(relevance_map):
    """The config-only 2x3_DISPATCH wildcard must not shadow specific entries."""
    entries = relevance_map["relevant_tests"]
    wildcard_idx = [
        i for i, e in enumerate(entries) if e.get("config") == "2x3_DISPATCH" and "group" not in e and "filter" not in e
    ]
    assert len(wildcard_idx) == 1, "expected exactly one config-only 2x3_DISPATCH entry"
    specific_at_dispatch = [i for i, e in enumerate(entries) if e.get("group") and e.get("config") != "1x3"]
    assert all(i < wildcard_idx[0] for i in specific_at_dispatch)


@pytest.mark.parametrize(
    "row,expected",
    [
        (("1x3", "unit_tests_legacy", "*DmLoopback*", "gtest"), "AIIPSW-2"),
        (("1x3", "unit_tests_api", "*TensixSingleCoreDirectDramReaderDatacopyWriter", "gtest"), "AIIPSW-6"),
        (("2x3", "unit_tests_dispatch", "*QuasarDispatchSInstantiatedAndRunning*", "gtest"), "AIIPSW-6"),
        (("2x3_DISPATCH", "unit_tests_legacy", "*SingleDmL1Write*", "gtest"), "AIIPSW-6"),
        (
            ("2x3_DISPATCH", "tests/tt_metal/tools/profiler/test_device_profiler.py", "test_full_buffer", "pytest"),
            "AIIPSW-13",
        ),
        (
            ("2x3", "models/demos/vision/classification/resnet50/quasar/tests/ops/test_add.py", "", "pytest"),
            "AIIPSW-4",
        ),
    ],
)
def test_shipped_map_attributes_known_tests(relevance_map, row, expected):
    entry = match_entry(*row, relevance_map)
    assert entry is not None, f"no entry matched {format_test(*row)}"
    assert entry.get("requirement") == expected
