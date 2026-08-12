#!/usr/bin/env python3
"""Tests for the release test-evidence report.

The risky part is classify(): passes are *derived* (expected minus failed)
because the sim CI reports only failures. These tests pin down when that
derivation is allowed to claim a pass and when it must refuse.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from file_rtl_sim_jira import parse_failed  # noqa: E402
from release_test_report import (  # noqa: E402
    FAILED,
    INCONCLUSIVE,
    PASSED,
    build,
    classify,
    load_expected,
    render_markdown,
    render_plain,
)

MAP_PATH = SCRIPTS_DIR / "ai_ip_tests.json"
SIM_YAML = SCRIPTS_DIR.parents[2] / "tests/scripts/quasar/quasar_sim_regresion_tests.yaml"

META = {
    "version": "v9.9.9",
    "sha": "deadbeef",
    "url": "http://sim",
    "run_url": "http://run",
    "config": "1x3",
    "sim_yaml_name": "quasar_sim_regresion_tests.yaml",
}


@pytest.fixture(scope="module")
def mapping():
    return json.loads(MAP_PATH.read_text())


@pytest.fixture
def expected(tmp_path):
    path = tmp_path / "sim.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            unit_tests_legacy:
              - filter: "*Alpha*"
                config: 1x3
              - filter: "*Beta*"
                config: 1x3
              - filter: "*Gamma*"
                config: 2x3
            unit_tests_api:
              - filter: "*Delta*"
                config: 1x3
            """
        )
    )
    return load_expected(path, "1x3")


def test_load_expected_filters_to_one_config(expected):
    assert [r["filter"] for r in expected] == ["*Alpha*", "*Beta*", "*Delta*"]
    assert all(r["config"] == "1x3" and r["runner"] == "gtest" for r in expected)


def test_green_run_passes_everything(expected):
    verdict, passed, failed = classify(expected, [], "success", "")
    assert verdict == PASSED
    assert len(passed) == 3 and failed == []


def test_red_run_passes_everything_not_reported_failed(expected):
    rows = parse_failed("- `[1x3] unit_tests_legacy --gtest_filter=*Beta*`")
    verdict, passed, failed = classify(expected, rows, "failure", "")
    assert verdict == FAILED
    assert [r["filter"] for r in passed] == ["*Alpha*", "*Delta*"]
    assert [r["filter"] for r in failed] == ["*Beta*"]


def test_red_run_without_detail_claims_no_passes(expected):
    verdict, passed, failed = classify(expected, [], "failure", "")
    assert verdict == INCONCLUSIVE
    assert passed == [] and failed == []


def test_timeout_claims_no_passes(expected):
    verdict, passed, _ = classify(expected, [], "timed_out (no result within 60 min)", "")
    assert verdict == INCONCLUSIVE and passed == []


def test_missing_manifest_claims_no_passes(expected):
    detail = "RTL sim: failure manifest missing\nThe sim failure manifest was not produced."
    verdict, passed, _ = classify(expected, parse_failed(detail), "failure", detail)
    assert verdict == INCONCLUSIVE and passed == []


def test_back2back_batch_fails_each_component(expected):
    rows = parse_failed("- `[1x3] unit_tests_legacy --gtest_filter=*Alpha*:*Beta*`")
    _verdict, passed, failed = classify(expected, rows, "failure", "")
    assert [r["filter"] for r in failed] == ["*Alpha*", "*Beta*"]
    assert [r["filter"] for r in passed] == ["*Delta*"]


def test_failure_outside_the_expected_set_is_surfaced(expected):
    rows = parse_failed("- `[1x3] unit_tests_legacy --gtest_filter=*Unknown*`")
    _verdict, passed, failed = classify(expected, rows, "failure", "")
    assert len(passed) == 3, "no expected test was reported failed"
    assert [r["filter"] for r in failed] == ["*Unknown*"], "the stray failure is still reported"


def test_inconclusive_report_claims_nothing(mapping, expected):
    report = build(mapping, expected, [], [], INCONCLUSIVE)
    assert all(not r["passed"] for r in report["requirements"])
    plain = render_plain(report, META)
    assert "INCONCLUSIVE" in plain
    assert "--- Requirements with passing test evidence ---\n(none)" in plain


def test_report_over_the_shipped_yaml_and_map(mapping):
    """End-to-end on the real gating list: a green run covers AIIPSW-2 and -6."""
    rows = load_expected(SIM_YAML, "1x3")
    verdict, passed, failed = classify(rows, [], "success", "")
    report = build(mapping, rows, passed, failed, verdict)

    covered = {r["key"] for r in report["requirements"] if r["passed"]}
    assert covered == {"AIIPSW-2", "AIIPSW-6"}

    # Every requirement in the inventory is accounted for, either way.
    assert len(report["requirements"]) == len(mapping["requirements"])

    # *Bmm runs in the gate but no requirement claims it -- it must not vanish.
    unattributed = [r["filter"] for r in report["unattributed"][PASSED]]
    assert "*Bmm" in unattributed

    markdown = render_markdown(report, META)
    assert "AIIPSW-2" in markdown and "AIIPSW-13" in markdown
    assert "Tests executed that map to no requirement" in markdown


def test_renderers_cover_every_requirement(mapping):
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED)
    plain = render_plain(report, META)
    for req in mapping["requirements"]:
        assert req["key"] in plain, f"{req['key']} missing from the report"
