#!/usr/bin/env python3
"""Build the release test-evidence report and file it as a Jira issue.

Two parts, kept separate: AIIPSW requirement evidence from the Quasar RTL sim
gate and the tt-umd-simulators emulator checks, and a summary of the model e2e
suites from their JUnit XML (not Quasar, so they map to no requirement).

Passes come from the sim CI's rtl-sim-results/v1 block when present. Otherwise
they are derived as (tests the gate runs) - (tests reported failed), which only
holds if the run completed: a red check with no per-test detail, a timeout, or a
missing manifest is reported INCONCLUSIVE with no passes claimed.

Environment:
  RTL_SIM_CONCLUSION  success | failure | timed_out | ...              (required)
  RTL_SIM_DETAIL      check output.summary (+ text)                    (optional)
  RTL_SIM_SHA / RTL_SIM_URL / RTL_SIM_RUN_URL                          (optional)
  HORIZON_RESULTS_FILE  tt-umd-horizon results (horizon-test-results/v1) (optional)
  HORIZON_EMU_RESULTS_FILE  Horizon emulator results, same schema, from the
                      tt-umd-simulators "Horizon Release" check        (optional)
  QUASAR_EMU_RESULTS_FILE  Quasar emulator results (quasar-test-results/v1),
                      from the tt-umd-simulators "Quasar Release" check (optional)
  HORIZON_MAX_AGE_DAYS  staleness threshold for the above (default 7)   (optional)
  RELEASE_VERSION     used in the summary and the dedup label          (optional)
  RTL_SIM_MAP         relevance mapping   (default: ./ai_ip_tests.json)
  QUASAR_SIM_YAML     the yaml the gating job runs
  SIM_CI_CONFIG       config the gating job selects              (default: 1x3)
  TEST_REPORTS_DIR    JUnit XML from release-demo-tests                (optional)
  REPORT_MD_OUT       write the markdown report here                   (optional)
  JIRA_*              as jira_client.py; JIRA_ISSUE_TYPE default Task
  JIRA_SKIP           build the report but do not file it

A Jira issue is filed only when something needs attention: sim failures, an
inconclusive sim check, or e2e suite failures. A fully green run is recorded in
the markdown artifact and step summary only.

Exits 0 whether or not tests failed -- this reports, it does not gate.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jira_client import _commit_link, _env, _truthy, file_issue
from create_jira import format_test, match_entry, parse_failed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SIM_YAML = REPO / "tests/scripts/quasar/quasar_sim_regresion_tests.yaml"

PASSED, FAILED, INCONCLUSIVE = "passed", "failed", "inconclusive"

# <!-- rtl-sim-results/v1\n{...}\n--> in the check summary.
RESULTS_RE = re.compile(r"<!--\s*rtl-sim-results/v1\s*\n(\{.*?\})\s*\n-->", re.DOTALL)


def parse_results_block(detail):
    """Return the sim CI's full result set, or None if it did not send one."""
    m = RESULTS_RE.search(detail or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        print("warning: rtl-sim-results block present but not valid JSON; ignoring")
        return None

    def rows(key):
        return [
            {
                "config": r.get("config", ""),
                "group": r.get("group", ""),
                "filter": r.get("filter", ""),
                "runner": r.get("runner", "gtest"),
            }
            for r in payload.get(key, [])
        ]

    # A truncated block omits the passed list: fall back rather than report zero.
    if payload.get("truncated"):
        print(f"warning: rtl-sim-results block truncated ({payload['truncated']}); falling back")
        return None
    return {"passed": rows("passed"), "failed": rows("failed")}


def parse_junit_dir(path):
    """Per-suite counts from the JUnit XML in the test_reports_* artifacts."""
    # Guard the empty string: Path("") is ".", which would scan the whole repo.
    if not path:
        return []
    root = Path(path)
    if not root.is_dir():
        return []

    suites = {}
    for xml in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml)
        except ET.ParseError:
            print(f"warning: {xml.name} is not parseable XML; skipping")
            continue
        for suite in tree.iter("testsuite"):
            # pytest names every suite "pytest"; the filename is the label.
            name = suite.get("name") or xml.stem
            if name in ("pytest", "", None):
                name = xml.stem
            acc = suites.setdefault(name, {"name": name, "passed": 0, "failed": 0, "skipped": 0, "failures": []})
            for case in suite.iter("testcase"):
                if case.find("failure") is not None or case.find("error") is not None:
                    acc["failed"] += 1
                    cls, nm = case.get("classname", ""), case.get("name", "")
                    acc["failures"].append(f"{cls}::{nm}" if cls else nm)
                elif case.find("skipped") is not None:
                    acc["skipped"] += 1
                else:
                    acc["passed"] += 1
    return sorted(suites.values(), key=lambda s: s["name"])


def suite_totals(suites):
    return {
        "suites": len(suites),
        "passed": sum(s["passed"] for s in suites),
        "failed": sum(s["failed"] for s in suites),
        "skipped": sum(s["skipped"] for s in suites),
    }


def load_expected(yaml_path, config):
    """Rows the gating job runs: the sim yaml filtered to one config."""
    data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    rows = []
    for group, entries in data.items():
        for entry in entries or []:
            configs = [c.strip() for c in str(entry.get("config", "")).split(",") if c.strip()]
            if config not in configs:
                continue
            rows.append(
                {
                    "config": config,
                    "group": group,
                    "filter": str(entry.get("filter") or ""),
                    "runner": str(entry.get("runner") or "gtest"),
                }
            )
    return rows


def classify(expected, failed_rows, conclusion, detail):
    """Split expected rows into passed/failed, or declare the run inconclusive.

    `failed_rows` are (config, group, filter, runner) tuples from the check.
    """
    # Authoritative when present.
    reported = parse_results_block(detail)
    if reported is not None:
        verdict = FAILED if reported["failed"] else PASSED
        return verdict, reported["passed"], reported["failed"]

    if conclusion == "success":
        return PASSED, list(expected), []

    # Red with no visible per-test detail: claim no passes. A truncated failure
    # list is the same situation -- the hidden rows would be counted as passes.
    low = (detail or "").lower()
    if not failed_rows or "manifest missing" in low or "(truncated)" in low:
        return INCONCLUSIVE, [], []

    failed_keys = {(g, f) for _c, g, f, _r in failed_rows}

    def is_failed(row):
        # A back2back batch arrives as one ':'-joined filter, so a row counts as
        # failed when its filter is any component of a reported batch.
        for group, filt in failed_keys:
            if group == row["group"] and row["filter"] in filt.split(":"):
                return True
        return False

    passed = [r for r in expected if not is_failed(r)]
    failed = [r for r in expected if is_failed(r)]
    # Failures the expected set does not explain: surface, do not drop.
    extra = [
        {"config": c, "group": g, "filter": f, "runner": r}
        for c, g, f, r in failed_rows
        if not any(e["group"] == g and e["filter"] in f.split(":") for e in expected)
    ]
    return FAILED, passed, failed + extra


# The requirement whose evidence comes from the tt-umd-horizon suite, not the
# Quasar sim gate. Horizon is a separate repo/CI: it publishes its own results
# file (horizon-test-results/v1) which the release job pulls and passes here.
HORIZON_REQUIREMENT = "AIIPSW-15"

# Horizon emulation: tt-umd-simulators' Horizon Release pipeline runs tt-metal's
# Quasar regression lists on the Horizon Zebu emulator and posts the results,
# in the same schema, as a "Horizon Release" check on the commit it tested.
HORIZON_EMU_REQUIREMENT = "AIIPSW-9"


# Horizon emulation (AIIPSW-9) is the ResNet LLK API on the Horizon emulator, so
# only the ResNet tests of the Horizon Release check credit it. Its other tests
# (data movement, dispatch, watcher) show tt-metal runs on Horizon at all, and
# are reported as executed tests that map to no requirement.
HORIZON_EMU_SCOPE = "models/demos/vision/classification/resnet50/"

# Quasar emulation: tt-umd-simulators' Quasar Release pipeline runs every row of
# tests/scripts/quasar/quasar_regression_tests.yaml and quasar_local_tests.yaml on
# the emulator config each row names, and posts the results as a "Quasar
# Release" check. Each row keeps its config, group, filter and runner, so it is
# credited through the relevance map exactly like a sim row.
QUASAR_EMU_SCHEMA = "quasar-test-results/v1"


def _load_results(path, schema, label, what, max_age_days):
    """The results document at `path`, or None (saying why) if missing, unreadable, off-schema or stale."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"{label}: results file '{path}' not present; {what} inconclusive")
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"{label}: results file unreadable ({e}); {what} inconclusive")
        return None
    if not isinstance(data, dict):
        print(f"{label}: results root is {type(data).__name__}, not an object; {what} inconclusive")
        return None
    if data.get("schema") != schema:
        print(f"{label}: unexpected schema {data.get('schema')!r}; {what} inconclusive")
        return None

    ts = data.get("timestamp", "")
    try:
        when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - when).total_seconds() / 86400
    except (ValueError, AttributeError):
        print(f"{label}: unparseable timestamp {ts!r}; {what} inconclusive")
        return None
    if age_days > max_age_days:
        print(f"{label}: results are {age_days:.0f}d old (> {max_age_days}); {what} inconclusive")
        return None
    return data


def _result_row(test, default_config):
    """A results test as a (config, group, filter, runner) row.

    tt-umd-horizon names only a tests/axi binary; the emulator checks also carry
    each tt-metal entry's group, filter and runner (an empty filter is a whole
    gtest binary or pytest file).
    """
    if test.get("group"):
        filt = str(test.get("filter") or "")
    else:
        filt = str(test.get("name", "?"))
    return {
        "config": str(test.get("config") or default_config),
        "group": str(test.get("group") or "tests/axi"),
        "filter": filt,
        "runner": str(test.get("runner") or "gtest"),
    }


def parse_horizon(path, req_key=HORIZON_REQUIREMENT, max_age_days=7):
    """Read the tt-umd-horizon results file into evidence for one requirement.

    Returns (status, extra_evidence). `status` is PASSED / FAILED / INCONCLUSIVE
    and `extra_evidence` is {req_key: {PASSED: [...], FAILED: [...]}} of test rows,
    empty when inconclusive. Missing / unreadable / wrong-schema / stale input all
    yield INCONCLUSIVE with no rows -- the same conservative stance the sim path
    takes, so the requirement simply shows "no passing evidence" until Horizon
    publishes a fresh green result.
    """
    data = _load_results(path, "horizon-test-results/v1", "horizon", req_key, max_age_days)
    if data is None:
        return INCONCLUSIVE, {}
    tests = data.get("tests", [])
    passed = [_result_row(t, "horizon") for t in tests if t.get("result") == "passed"]
    failed = [_result_row(t, "horizon") for t in tests if t.get("result") == "failed"]
    status = FAILED if failed else PASSED
    print(f"horizon: {len(passed)} passed, {len(failed)} failed (commit {str(data.get('tested_sha',''))[:12]})")
    return status, {req_key: {PASSED: passed, FAILED: failed}}


def parse_horizon_emu(path, max_age_days=7):
    """Evidence from the Horizon Release check: its ResNet tests credit AIIPSW-9.

    Same return shape as parse_horizon; the other tests sit under the None key,
    which build() reports as executed tests that map to no requirement.
    """
    status, evidence = parse_horizon(path, req_key=HORIZON_EMU_REQUIREMENT, max_age_days=max_age_days)
    if not evidence:
        return status, evidence
    hits = evidence[HORIZON_EMU_REQUIREMENT]
    resnet = {k: [r for r in rows if r["group"].startswith(HORIZON_EMU_SCOPE)] for k, rows in hits.items()}
    other = {k: [r for r in rows if not r["group"].startswith(HORIZON_EMU_SCOPE)] for k, rows in hits.items()}
    return status, {HORIZON_EMU_REQUIREMENT: resnet, None: other}


def parse_quasar_emu(path, max_age_days=7):
    """Read the Quasar Release results into test rows for the relevance map.

    Returns (status, {PASSED: [...], FAILED: [...]}), or (INCONCLUSIVE, {}) when
    the file is missing, unreadable, off-schema or stale.
    """
    data = _load_results(path, QUASAR_EMU_SCHEMA, "quasar-emu", "Quasar emulator evidence", max_age_days)
    if data is None:
        return INCONCLUSIVE, {}
    tests = data.get("tests", [])
    passed = [_result_row(t, "") for t in tests if t.get("result") == "passed"]
    failed = [_result_row(t, "") for t in tests if t.get("result") == "failed"]
    status = FAILED if failed else PASSED
    print(f"quasar-emu: {len(passed)} passed, {len(failed)} failed (commit {str(data.get('tested_sha',''))[:12]})")
    return status, {PASSED: passed, FAILED: failed}


def results_source(path):
    """One line naming where an emulator results file came from, or None if there is none."""
    try:
        data = json.loads(Path(path).read_text()) if path and Path(path).is_file() else None
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    t = data.get("totals") or {}
    return (
        f"{t.get('passed', '?')} passed, {t.get('failed', '?')} failed on tt-metal "
        f"{str(data.get('tested_sha', ''))[:12] or '?'}, {data.get('timestamp') or '?'}: {data.get('run_url') or '-'}"
    )


def build(mapping, expected, passed, failed, verdict, suites=None, extra_evidence=None, mapped_evidence=None):
    """Group the run's tests under the requirement each one serves.

    `extra_evidence` ({req_key: {PASSED: [...], FAILED: [...]}}) injects evidence
    from sources other than the sim gate (e.g. Horizon), attributed directly to a
    requirement rather than through the (config, group, filter) map.
    `mapped_evidence` ({PASSED: [...], FAILED: [...]}) adds test rows from another
    Quasar source (the emulator) that are credited through the map like sim
    rows, without touching the sim verdict or counts.
    """

    def req_of(row):
        entry = match_entry(row["config"], row["group"], row["filter"], row["runner"], mapping)
        return (entry or {}).get("requirement")

    mapped = mapped_evidence or {}
    covered = {}
    for row, outcome in (
        [(r, PASSED) for r in passed + mapped.get(PASSED, [])] + [(r, FAILED) for r in failed + mapped.get(FAILED, [])]
    ):
        key = req_of(row)
        covered.setdefault(key, {PASSED: [], FAILED: []})[outcome].append(row)

    for key, hits in (extra_evidence or {}).items():
        acc = covered.setdefault(key, {PASSED: [], FAILED: []})
        acc[PASSED] += hits.get(PASSED, [])
        acc[FAILED] += hits.get(FAILED, [])

    requirements = []
    for req in mapping.get("requirements", []):
        hits = covered.get(req["key"], {PASSED: [], FAILED: []})
        requirements.append({**req, "passed": hits[PASSED], "failed": hits[FAILED]})

    # Tests that ran but map to no requirement (or to a key not in the inventory).
    unattributed = covered.get(None, {PASSED: [], FAILED: []})
    known = {r["key"] for r in mapping.get("requirements", [])}
    for key, hits in covered.items():
        if key is not None and key not in known:
            unattributed[PASSED] += hits[PASSED]
            unattributed[FAILED] += hits[FAILED]

    return {
        "verdict": verdict,
        "suites": suites,
        "expected": expected,
        "passed": passed,
        "failed": failed,
        "requirements": requirements,
        "unattributed": unattributed,
    }


def _in_scope(requirements):
    """Requirements the Quasar gate could in principle evidence."""
    return [r for r in requirements if r.get("in_scope", True)]


def _out_of_scope(requirements):
    return [r for r in requirements if not r.get("in_scope", True)]


def _lines(rows):
    return [f"  - {format_test(r['config'], r['group'], r['filter'], r['runner'])}" for r in rows]


def render_plain(report, meta):
    """Plain text for the Jira description (ADF renders one paragraph per line)."""
    verdict = report["verdict"]
    scoped = _in_scope(report["requirements"])
    with_evidence = [r for r in scoped if r["passed"]]
    without = [r for r in scoped if not r["passed"]]

    out = [f"Release test evidence for {meta['version']}", ""]
    if verdict == PASSED:
        out.append(f"RESULT: all {len(report['passed'])} gating RTL sim test(s) passed.")
    elif verdict == FAILED:
        out.append(f"RESULT: {len(report['passed'])} test(s) passed, {len(report['failed'])} failed.")
    else:
        out.append(
            "RESULT: INCONCLUSIVE -- the RTL sim check was not green and carried no "
            "per-test detail, so no test can be recorded as having passed."
        )
    out += [
        f"Requirements with passing evidence: {len(with_evidence)} of {len(scoped)}.",
        "",
        f"Commit:      {_commit_link(meta['sha'])}",
        f"Sim results: {meta['url']}",
        f"Release run: {meta['run_url']}",
    ]
    out += [f"{name}: {text}" for name, text in meta.get("emulators", [])]
    out += [
        "",
        "--- Requirements with passing test evidence ---",
    ]
    if with_evidence:
        for req in with_evidence:
            out.append(f"{req['key']} ({req['milestone']}) -- {req['summary']} [{req['owner']}]")
            out += _lines(req["passed"])
            if req["failed"]:
                out.append("  FAILED in this run:")
                out += _lines(req["failed"])
    else:
        out.append("(none)")

    out += [
        "",
        "--- Requirements with no passing evidence in this release ---",
        "Evidence means executed and passed in this release. A test that exists, or that CI",
        "only compiles, is not evidence.",
    ]
    for req in without:
        why = req.get("evidence") or "no test executed by the release gate"
        note = " FAILED this run." if req["failed"] else ""
        out.append(f"{req['key']} ({req['milestone']}) -- {req['summary']}: {why}.{note}")
        out += _lines(req["failed"])

    if report["unattributed"][PASSED] or report["unattributed"][FAILED]:
        out += ["", "--- Tests executed that map to no requirement ---"]
        out += _lines(report["unattributed"][PASSED] + report["unattributed"][FAILED])

    oos = _out_of_scope(report["requirements"])
    if oos:
        out += [
            "",
            "Out of scope for this gate -- a different platform, so neither covered nor missing: "
            + ", ".join(f"{r['key']} ({r.get('team', '?')})" for r in oos),
        ]

    suites = report.get("suites") or []
    if suites:
        t = suite_totals(suites)
        out += [
            "",
            "--- Other release testing (model e2e suites) ---",
            f"{t['suites']} suite(s): {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped.",
        ]
        for s_ in suites:
            line = f"{s_['name']}: {s_['passed']} passed, {s_['failed']} failed, {s_['skipped']} skipped"
            out.append(line)
            for f_ in s_["failures"][:10]:
                out.append(f"  FAILED {f_}")
            if len(s_["failures"]) > 10:
                out.append(f"  ... and {len(s_['failures']) - 10} more")
        out.append("These suites are not Quasar and do not map to an AIIPSW requirement.")

    out += [
        "",
        "Scope: the requirement evidence above covers the RTL sim tests run by the release gate "
        f"({meta['sim_yaml_name']}, config {meta['config']}), plus the Quasar and Horizon "
        "emulator runs of quasar_regression_tests.yaml and quasar_local_tests.yaml, from the "
        "tt-umd-simulators \"Quasar Release\" and \"Horizon Release\" checks. "
        "Full inventory: the coverage doc in this artifact.",
    ]
    return "\n".join(out)


def render_markdown(report, meta):
    verdict = report["verdict"]
    badge = {PASSED: "✅ all gating tests passed", FAILED: "❌ failures present", INCONCLUSIVE: "⚠️ inconclusive"}[
        verdict
    ]
    scoped = _in_scope(report["requirements"])
    with_evidence = [r for r in scoped if r["passed"]]

    out = [
        f"# Release test evidence — {meta['version']}",
        "",
        f"**{badge}** — {len(report['passed'])} passed, {len(report['failed'])} failed, "
        f"{len(with_evidence)} of {len(scoped)} requirements with passing evidence.",
        "",
        f"| | |",
        f"|---|---|",
        f"| Commit | `{meta['sha']}` |",
        f"| Sim results | {meta['url']} |",
        f"| Release run | {meta['run_url']} |",
        f"| Scope | `{meta['sim_yaml_name']}` @ `{meta['config']}` |",
    ]
    out += [f"| {name} | {text} |" for name, text in meta.get("emulators", [])]
    out += [
        "",
    ]
    if verdict == INCONCLUSIVE:
        out += [
            "> The RTL sim check was not green and carried no per-test detail, so no "
            "test can be recorded as having passed. Nothing below is claimed as evidence.",
            "",
        ]

    out += ["## Requirements with passing test evidence", ""]
    if with_evidence:
        out += [
            "| Requirement | Milestone | Owner | Tests passed | Tests failed |",
            "|---|---|---|---|---|",
        ]
        for req in with_evidence:

            def cell(rows):
                return (
                    "<br>".join(f"`{format_test(r['config'], r['group'], r['filter'], r['runner'])}`" for r in rows)
                    or "—"
                )

            out.append(
                f"| **{req['key']}** — {req['summary']} | {req['milestone']} | {req['owner']} "
                f"| {cell(req['passed'])} | {cell(req['failed'])} |"
            )
    else:
        out.append("_None._")
    out.append("")

    out += [
        "## Requirements with no passing evidence in this release",
        "",
        "_Evidence means executed and passed in this release. A test that merely exists, or "
        "that CI only compiles, is not evidence._",
        "",
    ]
    out += ["| Requirement | Milestone | Owner | Why |", "|---|---|---|---|"]
    for req in [r for r in scoped if not r["passed"]]:
        why = req.get("evidence") or "no test executed by the release gate"
        if req["failed"]:
            why = "**failed this run**: " + ", ".join(
                f"`{format_test(r['config'], r['group'], r['filter'], r['runner'])}`" for r in req["failed"]
            )
        out.append(f"| {req['key']} — {req['summary']} | {req['milestone']} | {req['owner']} | {why} |")
    out.append("")

    extras = report["unattributed"][PASSED] + report["unattributed"][FAILED]
    if extras:
        out += ["## Tests executed that map to no requirement", ""]
        out += [f"- `{format_test(r['config'], r['group'], r['filter'], r['runner'])}`" for r in extras]
        out.append("")

    oos = _out_of_scope(report["requirements"])
    if oos:
        out += [
            "_Out of scope for this gate — a different platform, so neither covered nor missing: "
            + ", ".join(f"**{r['key']}** ({r.get('team', '?')})" for r in oos)
            + "._",
            "",
        ]

    suites = report.get("suites") or []
    if suites:
        t = suite_totals(suites)
        out += [
            "## Other release testing (model e2e suites)",
            "",
            f"**{t['suites']} suite(s)** — {t['passed']} passed, {t['failed']} failed, "
            f"{t['skipped']} skipped. Not Quasar, so these map to no AIIPSW requirement, "
            "but they are the bulk of what this release exercised.",
            "",
            "| Suite | Passed | Failed | Skipped |",
            "|---|---|---|---|",
        ]
        for s_ in suites:
            out.append(f"| `{s_['name']}` | {s_['passed']} | {s_['failed']} | {s_['skipped']} |")
        out.append("")
        failing = [s_ for s_ in suites if s_["failures"]]
        if failing:
            out += ["<details><summary>Failed tests</summary>", ""]
            for s_ in failing:
                out.append(f"**{s_['name']}**")
                out += [f"- `{f_}`" for f_ in s_["failures"][:25]]
                if len(s_["failures"]) > 25:
                    out.append(f"- … and {len(s_['failures']) - 25} more")
            out += ["", "</details>", ""]

    out += [
        "---",
        "",
        "Scope note: the requirement evidence above covers the RTL sim tests the release gate runs "
        f"(`{meta['sim_yaml_name']}`, config `{meta['config']}`), plus the Quasar and Horizon "
        "emulator runs of `quasar_regression_tests.yaml` and `quasar_local_tests.yaml`, from "
        "the tt-umd-simulators “Quasar Release” and “Horizon Release” checks. Full inventory: "
        "the coverage inventory attached to this same artifact.",
    ]
    return "\n".join(out)


def main():
    conclusion = (_env("RTL_SIM_CONCLUSION", "") or "").strip().lower()
    if not conclusion:
        sys.exit("error: RTL_SIM_CONCLUSION is required")
    detail = _env("RTL_SIM_DETAIL", "")
    version = _env("RELEASE_VERSION", "") or _env("RTL_SIM_SHA", "unknown")[:12]
    config = _env("SIM_CI_CONFIG", "1x3")
    sim_yaml = Path(_env("QUASAR_SIM_YAML", str(DEFAULT_SIM_YAML)))
    map_path = Path(_env("RTL_SIM_MAP", str(HERE / "ai_ip_tests.json")))

    mapping = json.loads(map_path.read_text())
    expected = load_expected(sim_yaml, config)
    failed_rows = parse_failed(detail)
    verdict, passed, failed = classify(expected, failed_rows, conclusion, detail)
    suites = parse_junit_dir(_env("TEST_REPORTS_DIR", ""))
    if suites:
        t = suite_totals(suites)
        print(
            f"read {t['suites']} test suite(s) from TEST_REPORTS_DIR: "
            f"{t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped"
        )
    max_age = int(_env("HORIZON_MAX_AGE_DAYS", "7") or "7")
    _horizon_status, horizon_evidence = parse_horizon(_env("HORIZON_RESULTS_FILE", ""), max_age_days=max_age)
    horizon_emu_file = _env("HORIZON_EMU_RESULTS_FILE", "")
    quasar_emu_file = _env("QUASAR_EMU_RESULTS_FILE", "")
    _emu_status, emu_evidence = parse_horizon_emu(horizon_emu_file, max_age_days=max_age)
    _quasar_status, quasar_rows = parse_quasar_emu(quasar_emu_file, max_age_days=max_age)
    report = build(
        mapping,
        expected,
        passed,
        failed,
        verdict,
        suites,
        extra_evidence={**horizon_evidence, **emu_evidence},
        mapped_evidence=quasar_rows,
    )

    meta = {
        "version": version,
        "sha": _env("RTL_SIM_SHA", "unknown"),
        "url": _env("RTL_SIM_URL", "-"),
        "run_url": _env("RTL_SIM_RUN_URL", "-"),
        "config": config,
        "sim_yaml_name": sim_yaml.name,
        # Only sources that were read: a stale or invalid file is not evidence.
        "emulators": [
            (name, results_source(path))
            for name, path, read in (
                ("Quasar emulator", quasar_emu_file, quasar_rows),
                ("Horizon emulator", horizon_emu_file, emu_evidence),
            )
            if read and results_source(path)
        ],
    }

    markdown = render_markdown(report, meta)
    out_path = _env("REPORT_MD_OUT", "")
    if out_path:
        Path(out_path).write_text(markdown + "\n")
        print(f"wrote {out_path}")
    else:
        print(markdown)

    if _truthy(_env("JIRA_SKIP")):
        print("JIRA_SKIP set; report not filed")
        return

    # A fully green run needs no ticket: the report is already in the step
    # summary and the release artifact. File only when there is something to
    # act on -- sim failures, an inconclusive check, or e2e suite failures.
    suite_failures = suite_totals(suites)["failed"] if suites else 0
    if verdict == PASSED and not suite_failures:
        print("all green; report kept in the artifact and step summary, no Jira issue filed")
        return

    scoped = _in_scope(report["requirements"])
    with_evidence = sum(1 for r in scoped if r["passed"])
    status = {
        PASSED: "all gating tests passed",
        FAILED: "sim failures present",
        INCONCLUSIVE: "sim check inconclusive",
    }[verdict]
    if suite_failures:
        status += f", {suite_failures} e2e suite test(s) failed"
    print(
        file_issue(
            base=_env("JIRA_BASE_URL", required=True),
            email=_env("JIRA_USER_EMAIL", required=True),
            token=_env("JIRA_API_TOKEN", required=True),
            project=_env("JIRA_PROJECT_KEY", required=True),
            summary=(
                f"Release test evidence {version}: {status} " f"({with_evidence}/{len(scoped)} requirements covered)"
            ),
            issue_type=_env("JIRA_ISSUE_TYPE", "Task"),
            assignee=_env("JIRA_ASSIGNEE_ACCOUNT_ID", "") or None,
            description=render_plain(report, meta) + "\n",
            labels=["release", "test-evidence", f"release-{version}"]
            + sorted({r["key"] for r in report["requirements"] if r["passed"]}),
            # One report issue per release; a re-run updates it instead of piling up.
            dedup_label=f"release-test-report:{version}",
            dry_run=_truthy(_env("JIRA_DRY_RUN")),
        )
    )


if __name__ == "__main__":
    main()
