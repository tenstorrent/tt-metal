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
    HORIZON_EMU_REQUIREMENT,
    HORIZON_REQUIREMENT,
    FAILED,
    INCONCLUSIVE,
    PASSED,
    QUASAR_EMU_SCHEMA,
    build,
    classify,
    load_expected,
    parse_horizon,
    parse_horizon_emu,
    parse_quasar_emu,
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


# --- Horizon (AIIPSW-15) evidence, pulled from the tt-umd-horizon repo ---------


def _horizon(tmp_path, tests, ts=None, schema="horizon-test-results/v1", **doc):
    """A tt-umd-horizon results file; `doc` adds top-level fields (the emulator's platform, variant, only, trial)."""
    from datetime import datetime, timezone

    ts = ts or datetime.now(timezone.utc).isoformat()
    p = tmp_path / "horizon-results.json"
    p.write_text(
        json.dumps(
            {
                "schema": schema,
                "tested_sha": "abc123def456",
                "timestamp": ts,
                "run_url": "http://horizon/run",
                "tests": tests,
                **doc,
            }
        )
    )
    return str(p)


def _horizon_emu(tmp_path, tests, **doc):
    """A Horizon Release document: what the emulator pipeline posts, built with the horizon register map."""
    return _horizon(tmp_path, tests, **{"platform": "emu-horizon", "tt_metal_quasar_variant": "horizon", **doc})


def test_horizon_fresh_green_becomes_evidence(tmp_path):
    path = _horizon(tmp_path, [{"name": "test_horizon_cluster", "result": "passed"}])
    status, evidence = parse_horizon(path)
    assert status == PASSED
    rows = evidence[HORIZON_REQUIREMENT][PASSED]
    assert [r["filter"] for r in rows] == ["test_horizon_cluster"]
    assert rows[0]["config"] == "horizon" and rows[0]["runner"] == "gtest"


def test_horizon_failure_is_reported(tmp_path):
    path = _horizon(
        tmp_path,
        [{"name": "test_axi_device", "result": "passed"}, {"name": "test_horizon_dma", "result": "failed"}],
    )
    status, evidence = parse_horizon(path)
    assert status == FAILED
    hits = evidence[HORIZON_REQUIREMENT]
    assert [r["filter"] for r in hits[PASSED]] == ["test_axi_device"]
    assert [r["filter"] for r in hits[FAILED]] == ["test_horizon_dma"]


def test_horizon_missing_file_is_inconclusive(tmp_path):
    assert parse_horizon("") == (INCONCLUSIVE, {})
    assert parse_horizon("/nonexistent/horizon.json") == (INCONCLUSIVE, {})
    truncated = tmp_path / "half.json"
    truncated.write_text('{"schema": "horizon-test-results/v1", "tests": [')
    assert parse_horizon(str(truncated)) == (INCONCLUSIVE, {}), "a truncated upload must not crash the report"


def test_horizon_non_object_root_is_inconclusive(tmp_path):
    """Valid JSON that is not an object (array, string, null) must degrade, not raise."""
    for i, body in enumerate(['[{"name": "t", "result": "passed"}]', '"horizon-test-results/v1"', "null", "42"]):
        p = tmp_path / f"root{i}.json"
        p.write_text(body)
        assert parse_horizon(str(p)) == (INCONCLUSIVE, {}), body


def test_horizon_stale_is_inconclusive(tmp_path):
    path = _horizon(tmp_path, [{"name": "t", "result": "passed"}], ts="2020-01-01T00:00:00Z")
    assert parse_horizon(path, max_age_days=7) == (INCONCLUSIVE, {})


def test_horizon_wrong_schema_is_inconclusive(tmp_path):
    path = _horizon(tmp_path, [{"name": "t", "result": "passed"}], schema="something-else/v1")
    assert parse_horizon(path) == (INCONCLUSIVE, {})


def test_horizon_evidence_flows_into_the_requirement(mapping, tmp_path):
    """A green Horizon result makes AIIPSW-15 render with passing evidence."""
    path = _horizon(tmp_path, [{"name": "test_horizon_cluster", "result": "passed"}])
    _status, evidence = parse_horizon(path)
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED, extra_evidence=evidence)

    covered = {r["key"] for r in report["requirements"] if r["passed"]}
    assert HORIZON_REQUIREMENT in covered
    md = render_markdown(report, META)
    assert "AIIPSW-15" in md and "test_horizon_cluster" in md


def test_no_horizon_evidence_leaves_the_requirement_uncovered(mapping):
    """Without Horizon input, AIIPSW-15 stays in the no-evidence section."""
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED)  # no extra_evidence
    covered = {r["key"] for r in report["requirements"] if r["passed"]}
    assert HORIZON_REQUIREMENT not in covered


def test_horizon_emu_rows_keep_their_group_and_filter(tmp_path):
    """Horizon emulator results name each tt-metal test; tt-umd-horizon ones only a binary."""
    path = _horizon(
        tmp_path,
        [
            {
                "name": "[1x3] unit_tests_legacy --gtest_filter=*Bmm",
                "result": "passed",
                "config": "horizon",
                "group": "unit_tests_legacy",
                "filter": "*Bmm",
                "runner": "gtest",
            },
            {"name": "test_axi_device", "result": "failed"},
        ],
    )
    status, evidence = parse_horizon(path, req_key=HORIZON_EMU_REQUIREMENT)
    assert status == FAILED and list(evidence) == [HORIZON_EMU_REQUIREMENT]
    hits = evidence[HORIZON_EMU_REQUIREMENT]
    assert hits[PASSED] == [{"config": "horizon", "group": "unit_tests_legacy", "filter": "*Bmm", "runner": "gtest"}]
    assert hits[FAILED] == [{"config": "horizon", "group": "tests/axi", "filter": "test_axi_device", "runner": "gtest"}]


RESNET_OP = "models/demos/vision/classification/resnet50/quasar/tests/ops/test_add.py"


def test_horizon_emu_credits_only_resnet_tests_to_aiipsw9(mapping, tmp_path):
    """AIIPSW-9 is the ResNet LLK API on Horizon: other Horizon tests are executed, not evidence for it."""
    (tmp_path / "emu").mkdir()
    emu = _horizon_emu(
        tmp_path / "emu",
        [
            {"name": "a", "result": "passed", "group": RESNET_OP, "filter": "", "runner": "pytest"},
            {"name": "b", "result": "passed", "group": "unit_tests_legacy", "filter": "*Bmm"},
        ],
    )
    umd = _horizon(tmp_path, [{"name": "test_horizon_cluster", "result": "passed"}])
    _s, umd_evidence = parse_horizon(umd)
    _s, emu_evidence = parse_horizon_emu(emu)
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED, extra_evidence={**umd_evidence, **emu_evidence})

    by_key = {r["key"]: r for r in report["requirements"]}
    assert [r["group"] for r in by_key[HORIZON_EMU_REQUIREMENT]["passed"]] == [RESNET_OP]
    assert by_key[HORIZON_REQUIREMENT]["passed"], "tt-umd-horizon still covers AIIPSW-15"
    horizon_bmm = {"config": "horizon", "group": "unit_tests_legacy", "filter": "*Bmm", "runner": "gtest"}
    assert horizon_bmm in report["unattributed"][PASSED], "reported as executed, not as AIIPSW-9 evidence"
    md = render_markdown(report, META)
    assert "AIIPSW-9" in md and "test_add.py" in md and "test_horizon_cluster" in md


# --- Quasar emulator evidence, from the tt-umd-simulators "Quasar Release" check ---


def _quasar_emu(tmp_path, tests, ts=None, schema=QUASAR_EMU_SCHEMA, **doc):
    """A Quasar Release document; `doc` overrides top-level fields (platform, variant, only, trial)."""
    from datetime import datetime, timezone

    p = tmp_path / "quasar-emu-results.json"
    p.write_text(
        json.dumps(
            {
                "schema": schema,
                "tested_sha": "fedcba987654",
                "timestamp": ts or datetime.now(timezone.utc).isoformat(),
                "run_url": "http://quasar/run",
                "platform": "emu-quasar",
                "tt_metal_quasar_variant": "",
                "only": [],
                "trial": "",
                "totals": {
                    "passed": sum(t["result"] == "passed" for t in tests),
                    "failed": sum(t["result"] == "failed" for t in tests),
                },
                "tests": tests,
                **doc,
            }
        )
    )
    return str(p)


def _q(config, group, filt, result, runner="gtest"):
    return {
        "name": f"[{config}] {group}",
        "result": result,
        "config": config,
        "source_config": config,
        "group": group,
        "filter": filt,
        "runner": runner,
    }


def test_quasar_emu_rows_keep_config_and_whole_file_filters(tmp_path):
    path = _quasar_emu(
        tmp_path,
        [_q("2x3", RESNET_OP, "", "passed", "pytest"), _q("1x3", "unit_tests_legacy", "*Bmm", "failed")],
    )
    status, rows = parse_quasar_emu(path)
    assert status == FAILED
    assert rows[PASSED] == [{"config": "2x3", "group": RESNET_OP, "filter": "", "runner": "pytest"}]
    assert rows[FAILED] == [{"config": "1x3", "group": "unit_tests_legacy", "filter": "*Bmm", "runner": "gtest"}]


def test_quasar_emu_missing_stale_or_wrong_schema_is_inconclusive(tmp_path):
    assert parse_quasar_emu("") == (INCONCLUSIVE, {})
    assert parse_quasar_emu(str(tmp_path / "absent.json")) == (INCONCLUSIVE, {})
    stale = _quasar_emu(tmp_path, [_q("1x3", "unit_tests_legacy", "*Bmm", "passed")], ts="2020-01-01T00:00:00Z")
    assert parse_quasar_emu(stale) == (INCONCLUSIVE, {})
    horizon = _quasar_emu(
        tmp_path, [_q("1x3", "unit_tests_legacy", "*Bmm", "passed")], schema="horizon-test-results/v1"
    )
    assert parse_quasar_emu(horizon) == (INCONCLUSIVE, {}), "Horizon results must not pass for Quasar ones"


def test_emulator_documents_that_are_narrowed_or_trial_give_no_evidence(tmp_path):
    """`only` narrows the run to a few tests and `trial` marks a dry run: neither speaks for the release."""
    ok = [_q("1x3", "unit_tests_legacy", "*Bmm", "passed")]
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok))[0] == PASSED, "the default document is accepted"
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok, only=["unit_tests_legacy"])) == (INCONCLUSIVE, {})
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok, trial=True)) == (INCONCLUSIVE, {})
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok, trial="smoke")) == (INCONCLUSIVE, {})

    horizon_ok = [{"name": "a", "result": "passed", "group": RESNET_OP, "filter": "", "runner": "pytest"}]
    assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok))[0] == PASSED
    assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok, only=["x"])) == (INCONCLUSIVE, {})
    assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok, trial=True)) == (INCONCLUSIVE, {})


def test_emulator_documents_must_be_built_for_their_platform(tmp_path):
    """A Horizon-variant build posted as Quasar results (or the reverse) is not evidence for either."""
    ok = [_q("1x3", "unit_tests_legacy", "*Bmm", "passed")]
    for variant in ("", "quasar"):
        assert parse_quasar_emu(_quasar_emu(tmp_path, ok, tt_metal_quasar_variant=variant))[0] == PASSED
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok, tt_metal_quasar_variant="horizon")) == (INCONCLUSIVE, {})
    assert parse_quasar_emu(_quasar_emu(tmp_path, ok, platform="emu-horizon")) == (INCONCLUSIVE, {})

    horizon_ok = [{"name": "a", "result": "passed", "group": RESNET_OP, "filter": "", "runner": "pytest"}]
    assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok))[0] == PASSED
    for variant in ("", "quasar"):
        assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok, tt_metal_quasar_variant=variant)) == (
            INCONCLUSIVE,
            {},
        )
    assert parse_horizon_emu(_horizon_emu(tmp_path, horizon_ok, platform="emu-quasar")) == (INCONCLUSIVE, {})


def test_documents_whose_tests_are_not_a_list_of_objects_give_no_evidence(tmp_path):
    """A malformed `tests` must make only that source inconclusive, not crash the report."""
    for bad in ("oops", [1, 2], [{"name": "a", "result": "passed"}, "b"], None):
        path = _quasar_emu(tmp_path, [])
        doc = json.loads(Path(path).read_text())
        doc["tests"] = bad
        Path(path).write_text(json.dumps(doc))
        assert parse_quasar_emu(path) == (INCONCLUSIVE, {}), bad
        umd = _horizon(tmp_path, [])
        doc = json.loads(Path(umd).read_text())
        doc["tests"] = bad
        Path(umd).write_text(json.dumps(doc))
        assert parse_horizon(umd) == (INCONCLUSIVE, {}), bad


def test_tt_umd_horizon_results_are_not_held_to_the_emulator_checks(tmp_path):
    """tt-umd-horizon's file has no platform or variant fields; AIIPSW-15 keeps reading it as before."""
    path = _horizon(tmp_path, [{"name": "test_horizon_cluster", "result": "passed"}])
    assert parse_horizon(path)[0] == PASSED


def test_quasar_emu_rows_are_credited_through_the_map(mapping, tmp_path):
    """Emulator rows reach the requirements the map names, without touching the sim verdict."""
    conv = "models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer_conv2_modelcfg.py"
    e2e = "models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py"
    profiler = "tests/tt_metal/tools/profiler/test_device_profiler.py"
    path = _quasar_emu(
        tmp_path,
        [
            _q("2x3", RESNET_OP, "", "passed", "pytest"),
            _q("2x3", conv, "", "passed", "pytest"),
            _q("2x3", e2e, "test_resnet50_e2e[pretrained-device_params0]", "passed", "pytest"),
            _q("2x3_DISPATCH", profiler, "test_full_buffer", "failed", "pytest"),
            _q("2x3_DISPATCH", "unit_tests_legacy", "*EventQuery", "passed"),
        ],
    )
    _status, emu_rows = parse_quasar_emu(path)
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED, mapped_evidence=emu_rows)

    by_key = {r["key"]: r for r in report["requirements"]}
    assert {r["group"] for r in by_key["AIIPSW-4"]["passed"]} == {RESNET_OP, e2e}
    assert [r["group"] for r in by_key["AIIPSW-16"]["passed"]] == [conv]
    assert [r["filter"] for r in by_key["AIIPSW-13"]["failed"]] == ["test_full_buffer"]
    assert "*EventQuery" in [r["filter"] for r in by_key["AIIPSW-6"]["passed"]], "2x3_DISPATCH wildcard"
    assert report["verdict"] == PASSED and report["passed"] == rows, "the sim verdict and counts are the sim's"


def test_emulator_sources_are_named_in_both_renderers(mapping):
    rows = load_expected(SIM_YAML, "1x3")
    report = build(mapping, rows, rows, [], PASSED)
    meta = {**META, "emulators": [("Quasar emulator", "110 passed, 0 failed on tt-metal fedcba987654")]}
    for text in (render_markdown(report, meta), render_plain(report, meta)):
        assert "Quasar emulator" in text and "fedcba987654" in text
        assert "Quasar Release" in text, "the scope note names the source check"


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
