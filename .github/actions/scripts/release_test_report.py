#!/usr/bin/env python3
"""Build the release test-evidence report and file it as a Jira issue.

Two parts, kept separate: AIIPSW requirement evidence from the Quasar RTL sim
gate, and a summary of the model e2e suites from their JUnit XML (not Quasar, so
they map to no requirement).

Passes come from the sim CI's rtl-sim-results/v1 block when present. Otherwise
they are derived as (tests the gate runs) - (tests reported failed), which only
holds if the run completed: a red check with no per-test detail, a timeout, or a
missing manifest is reported INCONCLUSIVE with no passes claimed.

Environment:
  RTL_SIM_CONCLUSION  success | failure | timed_out | ...              (required)
  RTL_SIM_DETAIL      check output.summary (+ text)                    (optional)
  RTL_SIM_SHA / RTL_SIM_URL / RTL_SIM_RUN_URL                          (optional)
  RELEASE_VERSION     used in the summary and the dedup label          (optional)
  RTL_SIM_MAP         relevance mapping   (default: ./ai_ip_tests.json)
  QUASAR_SIM_YAML     the yaml the gating job runs
  SIM_CI_CONFIG       config the gating job selects              (default: 1x3)
  TEST_REPORTS_DIR    JUnit XML from release-demo-tests                (optional)
  REPORT_MD_OUT       write the markdown report here                   (optional)
  JIRA_*              as jira_client.py; JIRA_ISSUE_TYPE default Task
  JIRA_SKIP           build the report but do not file it

Exits 0 whether or not tests failed -- this reports, it does not gate.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from jira_client import _env, _truthy, file_issue
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


def build(mapping, expected, passed, failed, verdict, suites=None):
    """Group the run's tests under the requirement each one serves."""

    def req_of(row):
        entry = match_entry(row["config"], row["group"], row["filter"], row["runner"], mapping)
        return (entry or {}).get("requirement")

    covered = {}
    for row, outcome in [(r, PASSED) for r in passed] + [(r, FAILED) for r in failed]:
        key = req_of(row)
        covered.setdefault(key, {PASSED: [], FAILED: []})[outcome].append(row)

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
        f"Commit:      {meta['sha']}",
        f"Sim results: {meta['url']}",
        f"Release run: {meta['run_url']}",
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
        f"({meta['sim_yaml_name']}, config {meta['config']}). Quasar tests that run "
        "only in the emulator job are not included -- that job reports to Slack and "
        "does not feed this check. Full inventory: the coverage doc in this artifact.",
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
        "Scope note: the requirement evidence above covers only the RTL sim tests the release gate runs "
        f"(`{meta['sim_yaml_name']}`, config `{meta['config']}`). Quasar tests that run "
        "only in the emulator job are not included — that job reports to Slack and does "
        "not feed this check. Full inventory: "
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
    report = build(mapping, expected, passed, failed, verdict, suites)

    meta = {
        "version": version,
        "sha": _env("RTL_SIM_SHA", "unknown"),
        "url": _env("RTL_SIM_URL", "-"),
        "run_url": _env("RTL_SIM_RUN_URL", "-"),
        "config": config,
        "sim_yaml_name": sim_yaml.name,
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

    scoped = _in_scope(report["requirements"])
    with_evidence = sum(1 for r in scoped if r["passed"])
    status = {PASSED: "all gating tests passed", FAILED: "failures present", INCONCLUSIVE: "inconclusive"}[verdict]
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
