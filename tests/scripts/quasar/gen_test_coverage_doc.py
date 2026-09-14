#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the Quasar test coverage inventory from the quasar test yamls.

Run by the package-and-release workflow, which attaches the result as a release
artifact. The output is deliberately not committed: it is a point-in-time view
of the yamls and ai_ip_tests.json, and a checked-in copy goes stale silently.

    gen_test_coverage_doc.py [-o PATH]
"""
import argparse
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAP_PATH = REPO / ".github/actions/scripts/ai_ip_tests.json"
OUT_PATH = HERE / "QUASAR_TEST_COVERAGE.md"

# Where each yaml is consumed -- the one thing not derivable from the repo.
# Verified 2026-08-12 against tensix/tt-umd-simulators main.
YAML_SOURCES = [
    {
        "file": "quasar_sim_regresion_tests.yaml",
        "job": "metal_unit_test_vcs_qsr (VCS)",
        "configs_run": "1x3 only (hardcoded)",
        "feeds_release_jira": True,
        "note": (
            "The only list wired to the release gate. Writes `test_results.tsv` "
            "(every outcome), which `report_rtl_sim_failures.py` turns into the "
            '"RTL Sim CI test" check output. Rows are labelled `1x3` unconditionally.'
        ),
    },
    {
        "file": "quasar_regression_tests.yaml",
        "job": "metal_unit_test_emu_quasar (Zebu/Aether emulator)",
        "configs_run": "all configs present in the yaml",
        "feeds_release_jira": False,
        "note": (
            "Nightly emulator run, still on GitLab branch "
            "`kstevens/emu-quasar-1x3-testing`. Reports to Slack `#tt-qsr-emu-ci`; "
            "writes no manifest, so none of these reach Jira."
        ),
    },
    {
        "file": "quasar_local_tests.yaml",
        "job": "metal_unit_test_emu_quasar (Zebu/Aether emulator)",
        "configs_run": "all configs present in the yaml",
        "feeds_release_jira": False,
        "note": (
            "pytest-only; wired on GitLab branch `kstevens/pytest_ci`, also "
            "Slack-reported. Entries move to the regression yaml as support lands."
        ),
    },
]


CASE_RE = re.compile(r"\bTEST(?:_F|_P)?\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)")


def defined_gtests():
    """Every gtest case declared under tests/tt_metal, as Fixture.Test."""
    cases = set()
    for src in (REPO / "tests/tt_metal").rglob("*.cpp"):
        for fixture, name in CASE_RE.findall(src.read_text(errors="ignore")):
            cases.add(f"{fixture}.{name}")
    return cases


def _core(gtest_filter):
    """Prefix/Fixture.Test/Param -> Fixture.Test (gtest instantiation naming)."""
    s = gtest_filter
    if "/" in s:
        _, _, rest = s.partition("/")
        if "." in rest:
            s = rest
    return s.split("/")[0]


def selected_gtests(rows, cases):
    hit = set()
    for row in rows:
        if row["runner"] != "gtest" or not row["filter"]:
            continue
        for part in row["filter"].split(":"):
            core = _core(part)
            pattern = core if any(c in core for c in "*?") else f"*{core}*"
            hit |= {c for c in cases if fnmatch.fnmatch(c, pattern)}
    return hit


def load_rows(path):
    """Flatten a quasar tests yaml into one row per (group, filter, config)."""
    data = yaml.safe_load(path.read_text()) or {}
    rows = []
    for group, entries in data.items():
        for entry in entries or []:
            configs = [c.strip() for c in str(entry.get("config", "")).split(",") if c.strip()]
            for config in configs:
                rows.append(
                    {
                        "group": group,
                        "filter": str(entry.get("filter") or ""),
                        "config": config,
                        "runner": str(entry.get("runner") or "gtest"),
                        "env": entry.get("env") or {},
                        "back2back": bool(entry.get("back2back")),
                    }
                )
    return rows


def load_map():
    return json.loads(MAP_PATH.read_text()).get("relevant_tests", [])


def load_map_requirements():
    """The AIIPSW requirement inventory the release evidence report reports on."""
    return json.loads(MAP_PATH.read_text()).get("requirements", [])


def match(row, entries):
    """The relevance-map entry watching this row, or None (same rules as the filer)."""
    for e in entries:
        if (
            e.get("config", row["config"]) == row["config"]
            and e.get("group", row["group"]) == row["group"]
            and e.get("filter", row["filter"]) == row["filter"]
            and e.get("runner", row["runner"]) == row["runner"]
        ):
            return e
    return None


def md_escape(text):
    return text.replace("|", "\\|") or "_(all tests in file)_"


def render():
    entries = load_map()
    out = [
        "# Quasar test coverage",
        "",
        "<!-- GENERATED by tests/scripts/quasar/gen_test_coverage_doc.py -- do not edit by hand. -->",
        "",
        "What the Quasar test yamls contain, where each list actually runs, and "
        "which rows are watched by the release Jira automation "
        "(`.github/actions/scripts/ai_ip_tests.json`).",
        "",
        "Regenerate with `python3 tests/scripts/quasar/gen_test_coverage_doc.py`.",
        "",
        "## Summary",
        "",
        "| Test list | Rows | Distinct tests | Configs | Runners | Consumed by | Feeds release Jira |",
        "|---|---|---|---|---|---|---|",
    ]

    all_rows = {}
    for src in YAML_SOURCES:
        path = HERE / src["file"]
        if not path.is_file():
            continue
        rows = load_rows(path)
        all_rows[src["file"]] = rows
        configs = sorted({r["config"] for r in rows})
        runners = sorted({r["runner"] for r in rows})
        distinct = len({(r["group"], r["filter"]) for r in rows})
        out.append(
            f"| `{src['file']}` | {len(rows)} | {distinct} | {', '.join(configs)} | "
            f"{', '.join(runners)} | {src['job']} | "
            f"{'**yes**' if src['feeds_release_jira'] else 'no'} |"
        )

    watched = sum(1 for rows in all_rows.values() for r in rows if match(r, entries))
    total = sum(len(rows) for rows in all_rows.values())
    gated = sum(len(all_rows.get(s["file"], [])) for s in YAML_SOURCES if s["feeds_release_jira"])
    out += [
        "",
        f"**{total}** test rows across all lists; **{gated}** of them run in the job that "
        f"gates the release; **{watched}** are matched by the relevance map.",
        "",
    ]

    for src in YAML_SOURCES:
        rows = all_rows.get(src["file"])
        if not rows:
            continue
        out += [f"## `{src['file']}`", "", src["note"], ""]
        out += [
            f"Runs: {src['job']} — {src['configs_run']}.",
            "",
            "| Group | Filter | Config | Runner | Watched (requirement) |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r["group"], r["filter"], r["config"])):
            e = match(r, entries)
            flag = f"✅ {e.get('requirement', 'yes')}" if e else "—"
            b2b = " _(back2back)_" if r["back2back"] else ""
            out.append(
                f"| `{r['group']}` | `{md_escape(r['filter'])}`{b2b} | {r['config']} | " f"{r['runner']} | {flag} |"
            )
        out.append("")

    # What each AIIPSW ticket is covered by.
    by_req = defaultdict(list)
    for fname, rows in all_rows.items():
        for r in rows:
            e = match(r, entries)
            if e and e.get("requirement"):
                by_req[e["requirement"]].append((fname, r))
    inventory = load_map_requirements()
    out += [
        "## Coverage by AIIPSW requirement",
        "",
        "Requirements from `ai_ip_tests.json`. *Wired rows* are yaml rows the "
        "relevance map watches; *evidence* names the Quasar tests that exist for "
        "the requirement whether or not any yaml runs them.",
        "",
        "| Requirement | Milestone | Owner | Wired rows | Gates release | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    gating = {s["file"] for s in YAML_SOURCES if s["feeds_release_jira"]}
    known = {r["key"] for r in inventory}
    rows_out = list(inventory) + [{"key": k, "milestone": "?", "owner": "?"} for k in sorted(by_req) if k not in known]
    for req in rows_out:
        hits = by_req.get(req["key"], [])
        live = any(f in gating for f, _ in hits)
        evidence = req.get("evidence", "wired into the gating yaml")
        out.append(
            f"| **{req['key']}** — {req.get('summary', '')} | {req.get('milestone', '?')} "
            f"| {req.get('owner', '?')} | {len(hits)} | {'yes' if live else 'no'} "
            f"| {evidence} |"
        )
    out.append("")

    # Never win a match: no such row, or shadowed by an earlier entry.
    unmatched = []
    for e in entries:
        hit = any(match(r, entries) is e for rows in all_rows.values() for r in rows)
        if not hit:
            unmatched.append(e)
    if unmatched:
        out += [
            "## Relevance-map entries that never win a match",
            "",
            "No yaml row matches, or an earlier entry claims every row they would.",
            "",
            "| Config | Group | Filter | Requirement |",
            "|---|---|---|---|",
        ]
        for e in unmatched:
            out.append(
                f"| {e.get('config', '*')} | `{e.get('group', '*')}` | "
                f"`{md_escape(e.get('filter', '*'))}` | {e.get('requirement', '—')} |"
            )
        out.append("")

    # Skipped rather than reported as zero when tests/tt_metal is absent, e.g. a
    # sparse checkout that did not ask for it.
    cases = defined_gtests()
    quasar_cases = {c for c in cases if "quasar" in c.lower()}
    picked = selected_gtests([r for rows in all_rows.values() for r in rows], cases)
    gate_rows = [r for f, rows in all_rows.items() if f in gating for r in rows]
    out += (
        []
        if not cases
        else [
            "## tt-metal gtests: defined vs selected",
            "",
            "The yamls select by gtest filter from binaries that hold far more than "
            "they run. Approximate — `TEST_P` instantiation names are normalised.",
            "",
            "| | Count |",
            "|---|---|",
            f"| gtest cases defined under `tests/tt_metal` | {len(cases)} |",
            f"| of those, Quasar-named | {len(quasar_cases)} |",
            f"| selected by any quasar yaml | {len(picked)} |",
            f"| selected by the gating sim yaml | {len(selected_gtests(gate_rows, cases))} |",
            f"| Quasar-named, selected by no yaml | {len(quasar_cases - picked)} |",
            "",
        ]
    )

    counts = Counter((r["config"], r["runner"]) for rows in all_rows.values() for r in rows)
    out += ["## Rows by config and runner", "", "| Config | Runner | Rows |", "|---|---|---|"]
    for (config, runner), n in sorted(counts.items()):
        out.append(f"| {config} | {runner} | {n} |")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(OUT_PATH), help=f"output path (default: {OUT_PATH.name})")
    args = ap.parse_args()

    out = Path(args.out)
    out.write_text(render())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
