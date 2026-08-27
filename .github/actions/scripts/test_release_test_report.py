#!/usr/bin/env python3
"""Tests for the release test-evidence report.

The risky part is classify(): passes are *derived* (expected minus failed)
because the sim CI reports only failures. These tests pin down when that
derivation is allowed to claim a pass and when it must refuse.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from create_jira import parse_failed  # noqa: E402
from release_test_report import (  # noqa: E402
    FAILED,
    INCONCLUSIVE,
    PASSED,
    build,
    classify,
    load_expected,
    parse_junit_dir,
    parse_results_block,
    render_markdown,
    render_plain,
    suite_totals,
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


# --- the sim CI's authoritative result block (rtl-sim-results/v1) -------------


def _block(passed, failed, extra=""):
    payload = {
        "schema": "rtl-sim-results/v1",
        "total": len(passed) + len(failed),
        "passed": [{"config": "1x3", "group": g, "filter": f} for g, f in passed],
        "failed": [{"config": "1x3", "group": g, "filter": f} for g, f in failed],
    }
    body = json.dumps(payload)[:-1] + extra + "}" if extra else json.dumps(payload)
    return f"<!-- rtl-sim-results/v1\n{body}\n-->"


def test_results_block_is_authoritative_over_derivation(expected):
    """When the sim CI reports passes explicitly, nothing is inferred."""
    detail = _block(passed=[("unit_tests_legacy", "*Zeta*")], failed=[])
    verdict, passed, failed = classify(expected, [], "success", detail)
    assert verdict == PASSED
    # *Zeta* is not in the expected set at all -- proof the block won, not the yaml.
    assert [r["filter"] for r in passed] == ["*Zeta*"] and failed == []


def test_results_block_reports_failures(expected):
    detail = _block(passed=[("unit_tests_legacy", "*Alpha*")], failed=[("unit_tests_api", "*Delta*")])
    verdict, passed, failed = classify(expected, [], "failure", detail)
    assert verdict == FAILED
    assert [r["filter"] for r in passed] == ["*Alpha*"]
    assert [r["filter"] for r in failed] == ["*Delta*"]


def test_truncated_block_falls_back_rather_than_reporting_zero_passes(expected):
    detail = _block([], [], extra=',"truncated":"passed list omitted: output size limit"')
    assert classify(expected, [], "failure", detail)[0] == INCONCLUSIVE


def test_malformed_block_is_ignored(expected):
    detail = "<!-- rtl-sim-results/v1\n{not json}\n-->"
    assert parse_results_block(detail) is None
    # falls through to the derivation path
    assert classify(expected, [], "success", detail)[0] == PASSED


def test_no_block_still_derives(expected):
    assert parse_results_block("1 test(s) failed:\n- `[1x3] g --gtest_filter=*X*`") is None


# --- other release testing: JUnit XML from release-demo-tests ----------------

JUNIT_PYTEST = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="1" tests="4">
<testcase classname="models.demos.llama3.tests.test_x" name="test_a"/>
<testcase classname="models.demos.llama3.tests.test_x" name="test_b"/>
<testcase classname="models.demos.llama3.tests.test_x" name="test_c"><failure message="boom">E</failure></testcase>
<testcase classname="models.demos.llama3.tests.test_x" name="test_d"><skipped message="n/a"/></testcase>
</testsuite></testsuites>
"""

JUNIT_ERROR = """<?xml version="1.0"?>
<testsuites><testsuite name="GtestSuite" tests="1">
<testcase classname="GtestSuite" name="Boom"><error message="segv">crash</error></testcase>
</testsuite></testsuites>
"""


def test_junit_dir_absent_is_empty():
    assert parse_junit_dir("/nonexistent/path/xyz") == []
    assert parse_junit_dir("") == []


def test_junit_counts_and_failure_names(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "most_recent_tests_1.xml").write_text(JUNIT_PYTEST)
    suites = parse_junit_dir(tmp_path)
    assert len(suites) == 1
    s = suites[0]
    # pytest labels every suite "pytest"; the filename is the useful label
    assert s["name"] == "most_recent_tests_1"
    assert (s["passed"], s["failed"], s["skipped"]) == (2, 1, 1)
    assert s["failures"] == ["models.demos.llama3.tests.test_x::test_c"]


def test_junit_counts_errors_as_failures(tmp_path):
    (tmp_path / "g.xml").write_text(JUNIT_ERROR)
    s = parse_junit_dir(tmp_path)[0]
    assert s["name"] == "GtestSuite" and s["failed"] == 1


def test_junit_skips_unparseable_files(tmp_path):
    (tmp_path / "ok.xml").write_text(JUNIT_ERROR)
    (tmp_path / "broken.xml").write_text("not xml at all")
    suites = parse_junit_dir(tmp_path)
    assert len(suites) == 1, "the corrupt file must not take the whole report down"


def test_junit_merges_suites_of_the_same_name(tmp_path):
    (tmp_path / "one.xml").write_text(JUNIT_ERROR)
    (tmp_path / "two.xml").write_text(JUNIT_ERROR)
    s = parse_junit_dir(tmp_path)
    assert len(s) == 1 and s[0]["failed"] == 2


def test_suite_totals(tmp_path):
    (tmp_path / "a.xml").write_text(JUNIT_PYTEST)
    (tmp_path / "b.xml").write_text(JUNIT_ERROR)
    t = suite_totals(parse_junit_dir(tmp_path))
    assert t == {"suites": 2, "passed": 2, "failed": 2, "skipped": 1}


def test_suites_render_but_do_not_touch_requirement_evidence(mapping, tmp_path):
    (tmp_path / "most_recent_tests_1.xml").write_text(JUNIT_PYTEST)
    suites = parse_junit_dir(tmp_path)
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED, suites)

    md = render_markdown(report, META)
    assert "Other release testing" in md and "most_recent_tests_1" in md
    assert "test_c" in md, "failed model tests must be listed"
    # the model suites are not Quasar: they must not become requirement evidence
    covered = {r["key"] for r in report["requirements"] if r["passed"]}
    assert covered == {"AIIPSW-2", "AIIPSW-6"}
    assert "map to no AIIPSW requirement" in md

    plain = render_plain(report, META)
    assert "Other release testing" in plain and "FAILED" in plain


def test_no_suites_omits_the_section(mapping):
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED, [])
    assert "Other release testing" not in render_markdown(report, META)


def test_truncated_failure_list_is_inconclusive(expected):
    """Hidden failures behind '… and N more (truncated)' must not become passes."""
    detail = (
        "3 of 9 RTL sim test(s) failed:\n"
        "- `[1x3] unit_tests_legacy --gtest_filter=*Beta*`\n"
        "- … and 2 more (truncated)"
    )
    verdict, passed, _ = classify(expected, parse_failed(detail), "failure", detail)
    assert verdict == INCONCLUSIVE and passed == []


def test_results_block_wins_even_when_the_summary_is_truncated(expected):
    """Truncation only affects the derived path; an explicit block is still exact."""
    detail = "- … and 2 more (truncated)\n" + _block(
        passed=[("unit_tests_legacy", "*Alpha*")], failed=[("unit_tests_api", "*Delta*")]
    )
    verdict, passed, failed = classify(expected, parse_failed(detail), "failure", detail)
    assert verdict == FAILED
    assert [r["filter"] for r in passed] == ["*Alpha*"] and len(failed) == 1


def test_evidence_carries_no_test_counts(mapping):
    """Counts drift whenever a test lands; the map must not assert them.

    Inverted on purpose: strip the numeric forms that are immutable (PR refs,
    release versions, ticket keys, PR tallies for a closed window) and flag what is left,
    rather than trying to enumerate every way a count can be phrased.
    """
    allowed = re.compile(r"(?:PR\s*)?#\d+|\bPR\s+\d+|\bv\d+(?:\.\d+)+|\b\d+\s+PRs\b|\bAIIPSW-\d+\b", re.IGNORECASE)
    offenders = []
    for r in mapping["requirements"]:
        residue = allowed.sub("", r.get("evidence", ""))
        found = re.findall(r"\b\d+\b", residue)
        if found:
            offenders.append((r["key"], found))
    assert not offenders, f"hard-coded counts will go stale: {offenders}"


def test_out_of_scope_requirements_leave_the_ratio_alone(mapping):
    """A platform this gate cannot test must not inflate the denominator."""
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED)
    scoped = [r for r in report["requirements"] if r.get("in_scope", True)]
    oos = [r for r in report["requirements"] if not r.get("in_scope", True)]
    assert oos, "fixture expects at least one out-of-scope requirement"

    md = render_markdown(report, META)
    assert f"of {len(scoped)} requirements" in md
    assert f"of {len(report['requirements'])} requirements" not in md, "denominator must exclude out-of-scope"

    # they are still named once, so nothing looks forgotten
    for r in oos:
        assert r["key"] in md and r["key"] in render_plain(report, META)
    # ...but not as a row in the no-evidence table
    assert md.count(oos[0]["key"]) == 1


def test_the_count_guard_actually_catches_a_violation(mapping):
    """Guards that scan the wrong key pass silently; prove this one bites."""
    import copy

    allowed = re.compile(r"(?:PR\s*)?#\d+|\bPR\s+\d+|\bv\d+(?:\.\d+)+|\b\d+\s+PRs\b|\bAIIPSW-\d+\b", re.IGNORECASE)

    def offenders(m):
        return [r["key"] for r in m["requirements"] if re.findall(r"\b\d+\b", allowed.sub("", r.get("evidence", "")))]

    assert offenders(mapping) == [], "the shipped map must be clean"
    bad = copy.deepcopy(mapping)
    bad["requirements"][0]["evidence"] = "NOT EXECUTED. 50 op tests in models/x/"
    assert offenders(bad), "an injected count must be caught"
