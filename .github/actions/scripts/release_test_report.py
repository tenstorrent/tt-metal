#!/usr/bin/env python3
"""Build the release test-evidence report and file it as a Jira issue.

The failure path (file_rtl_sim_jira.py) opens a ticket per failed test. This is
the other half: after the release gate runs, produce a shareable record of which
AIIPSW requirements actually have passing test evidence in this release, and
which have none.

How "executed successfully" is determined
-----------------------------------------
The sim CI reports only FAILURES -- `failed_tests.tsv` lists failed tests and
nothing else, so there is no positive list of passes to read. Passes are
therefore derived:

    executed = the rows of the sim yaml the gating job runs (config SIM_CI_CONFIG)
    failed   = the per-test rows parsed out of the check detail
    passed   = executed - failed

That derivation is only sound when the job ran to completion. When the check is
red but carries no per-test detail, or timed out, or the sim reporter says the
manifest was missing, the run is marked INCONCLUSIVE and NOTHING is reported as
passing -- a report that guesses at passes is worse than no report.

Environment:
  RTL_SIM_CONCLUSION  check conclusion: success | failure | timed_out | ...  (required)
  RTL_SIM_DETAIL      check output.summary (+ text); failure bullets    (optional)
  RTL_SIM_SHA         commit the check ran on                           (optional)
  RTL_SIM_URL         link to the sim results                           (optional)
  RTL_SIM_RUN_URL     link to the release workflow run                  (optional)
  RELEASE_VERSION     release tag/version, used in the summary + dedup  (optional)
  RTL_SIM_MAP         relevance mapping JSON   (default: ./ai_ip_tests.json)
  QUASAR_SIM_YAML     the yaml the gating job runs
                      (default: tests/scripts/quasar/quasar_sim_regresion_tests.yaml)
  SIM_CI_CONFIG       config the gating job selects                     (default: 1x3)
  REPORT_MD_OUT       write the markdown report here                    (optional)
  JIRA_*              as create_jira_issue.py; JIRA_ISSUE_TYPE defaults to Task
  JIRA_SKIP           when truthy, build the report but do not file it

Exit status is 0 whether or not tests failed -- this reports, it does not gate.
"""
import json
import os
import sys
from pathlib import Path

import yaml

from create_jira_issue import _env, _truthy, file_issue
from file_rtl_sim_jira import format_test, match_entry, parse_failed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SIM_YAML = REPO / "tests/scripts/quasar/quasar_sim_regresion_tests.yaml"

PASSED, FAILED, INCONCLUSIVE = "passed", "failed", "inconclusive"


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
    if conclusion == "success":
        return PASSED, list(expected), []

    # Red, but we cannot see which tests failed -- infra failure, timeout, or the
    # sim pipeline did not forward per-test detail. Claim no passes.
    if not failed_rows or "manifest missing" in (detail or "").lower():
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
    # Failures the expected set does not explain (the yaml moved, or the sim ran
    # something else): surface them rather than dropping them.
    extra = [
        {"config": c, "group": g, "filter": f, "runner": r}
        for c, g, f, r in failed_rows
        if not any(e["group"] == g and e["filter"] in f.split(":") for e in expected)
    ]
    return FAILED, passed, failed + extra


def build(mapping, expected, passed, failed, verdict):
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
        "expected": expected,
        "passed": passed,
        "failed": failed,
        "requirements": requirements,
        "unattributed": unattributed,
    }


def _lines(rows):
    return [f"  - {format_test(r['config'], r['group'], r['filter'], r['runner'])}" for r in rows]


def render_plain(report, meta):
    """Plain text for the Jira description (ADF renders one paragraph per line)."""
    verdict = report["verdict"]
    with_evidence = [r for r in report["requirements"] if r["passed"]]
    without = [r for r in report["requirements"] if not r["passed"]]

    out = [f"Release test evidence for {meta['version']}", ""]
    if verdict == PASSED:
        out.append(f"RESULT: all {len(report['passed'])} gating RTL sim test(s) passed.")
    elif verdict == FAILED:
        out.append(
            f"RESULT: {len(report['passed'])} test(s) passed, {len(report['failed'])} failed."
        )
    else:
        out.append(
            "RESULT: INCONCLUSIVE -- the RTL sim check was not green and carried no "
            "per-test detail, so no test can be recorded as having passed."
        )
    out += [
        f"Requirements with passing evidence: {len(with_evidence)} of {len(report['requirements'])}.",
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

    out += ["", "--- Requirements with no passing evidence in this release ---"]
    for req in without:
        why = req.get("_evidence") or "no test executed by the release gate"
        note = " FAILED this run." if req["failed"] else ""
        out.append(f"{req['key']} ({req['milestone']}) -- {req['summary']}: {why}.{note}")
        out += _lines(req["failed"])

    if report["unattributed"][PASSED] or report["unattributed"][FAILED]:
        out += ["", "--- Tests executed that map to no requirement ---"]
        out += _lines(report["unattributed"][PASSED] + report["unattributed"][FAILED])

    out += [
        "",
        "Scope: this covers the RTL sim tests run by the release gate "
        f"({meta['sim_yaml_name']}, config {meta['config']}). Quasar tests that run "
        "only in the emulator job are not included -- that job reports to Slack and "
        "does not feed this check. See tests/scripts/quasar/QUASAR_TEST_COVERAGE.md.",
    ]
    return "\n".join(out)


def render_markdown(report, meta):
    verdict = report["verdict"]
    badge = {PASSED: "✅ all gating tests passed", FAILED: "❌ failures present", INCONCLUSIVE: "⚠️ inconclusive"}[verdict]
    with_evidence = [r for r in report["requirements"] if r["passed"]]

    out = [
        f"# Release test evidence — {meta['version']}",
        "",
        f"**{badge}** — {len(report['passed'])} passed, {len(report['failed'])} failed, "
        f"{len(with_evidence)} of {len(report['requirements'])} requirements with passing evidence.",
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
                    "<br>".join(
                        f"`{format_test(r['config'], r['group'], r['filter'], r['runner'])}`"
                        for r in rows
                    )
                    or "—"
                )

            out.append(
                f"| **{req['key']}** — {req['summary']} | {req['milestone']} | {req['owner']} "
                f"| {cell(req['passed'])} | {cell(req['failed'])} |"
            )
    else:
        out.append("_None._")
    out.append("")

    out += ["## Requirements with no passing evidence in this release", ""]
    out += ["| Requirement | Milestone | Owner | Why |", "|---|---|---|---|"]
    for req in [r for r in report["requirements"] if not r["passed"]]:
        why = req.get("_evidence") or "no test executed by the release gate"
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

    out += [
        "---",
        "",
        "Scope note: this covers only the RTL sim tests the release gate runs "
        f"(`{meta['sim_yaml_name']}`, config `{meta['config']}`). Quasar tests that run "
        "only in the emulator job are not included — that job reports to Slack and does "
        "not feed this check. Full inventory: "
        "[`tests/scripts/quasar/QUASAR_TEST_COVERAGE.md`](../../../tests/scripts/quasar/QUASAR_TEST_COVERAGE.md).",
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
    report = build(mapping, expected, passed, failed, verdict)

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

    with_evidence = sum(1 for r in report["requirements"] if r["passed"])
    status = {PASSED: "all gating tests passed", FAILED: "failures present", INCONCLUSIVE: "inconclusive"}[verdict]
    print(
        file_issue(
            base=_env("JIRA_BASE_URL", required=True),
            email=_env("JIRA_USER_EMAIL", required=True),
            token=_env("JIRA_API_TOKEN", required=True),
            project=_env("JIRA_PROJECT_KEY", required=True),
            summary=(
                f"Release test evidence {version}: {status} "
                f"({with_evidence}/{len(report['requirements'])} requirements covered)"
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
