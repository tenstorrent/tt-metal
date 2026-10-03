#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Render the sweep log as report.md and junit.xml.

Usage:
  python3 report.py --jsonl /path/to/findings.jsonl --report-md /path/to/report.md \
      --junit /path/to/junit.xml
"""

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree

_ILLEGAL = re.compile(r"[^\x09\x0a\x0d\x20-퟿-�]")


def _clean(text) -> str:
    return _ILLEGAL.sub("", str(text or ""))


def load(jsonl_path: Path) -> list:
    records = []
    if not jsonl_path.exists():
        return records
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a killed sweep can cut off the final append
    return records


def _is_escape(r) -> bool:
    return r["verdict"] != r["victim_baseline"] and r.get("verified") is not False


def _table(headers, rows) -> list:
    return [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
        *(f"| {' | '.join(str(v) for v in row)} |" for row in rows),
    ]


def render_markdown(records: list) -> str:
    escapes = [r for r in records if _is_escape(r)]
    out = ["# Reconfig sweep results", ""]
    out.append(f"{len(records)} trial(s), {len(escapes)} escape(s)")
    if not records:
        return "\n".join(out) + "\n"

    if escapes:
        out += ["", "## Escapes", ""]
        out += _table(
            ("polluter (X)", "victim (K)", "K baseline", "K after X", "mode"),
            (
                (
                    f"`{e['polluter']}`",
                    f"`{e['victim']}`",
                    e["victim_baseline"],
                    e["verdict"],
                    e["mode"],
                )
                for e in escapes
            ),
        )
    else:
        out += [
            "",
            "No escapes: every victim that passes on its own still passes after "
            "every polluter in the catalog.",
        ]

    fullreset = [r for r in records if r["mode"] == "fullreset"]
    if fullreset:
        out += [
            "",
            "## Full-reset fallback rows",
            *_table(
                ("polluter (X)", "trials"),
                sorted(
                    (
                        (x, sum(1 for r in fullreset if r["polluter"] == x))
                        for x in {r["polluter"] for r in fullreset}
                    ),
                ),
            ),
        ]
    return "\n".join(out) + "\n"


def write_markdown(records: list, path: Path) -> Path:
    path.write_text(render_markdown(records))
    return path


def render_junit(records: list, path: Path) -> Path:
    cases = []
    failed = 0
    for r in records:
        name = f"{r['polluter']}__then__{r['victim']}"
        element = ElementTree.Element(
            "testcase", classname="reconfig_escape.pair_sweep", name=_clean(name)
        )
        if _is_escape(r):
            failed += 1
            failure = ElementTree.SubElement(
                element,
                "failure",
                type="ReconfigEscape",
                message=_clean(
                    f"{r['victim']} was {r['verdict']} after {r['polluter']} "
                    f"(baseline {r['victim_baseline']})"
                )[:200],
            )
            failure.text = _clean(json.dumps(r))
        cases.append(element)

    suite = ElementTree.Element(
        "testsuite",
        name="reconfig_escape",
        tests=str(len(cases)),
        failures=str(failed),
        errors="0",
        skipped="0",
    )
    suite.extend(cases)
    root = ElementTree.Element("testsuites")
    root.append(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--report-md", required=True)
    p.add_argument("--junit", required=True)
    args = p.parse_args()

    records = load(Path(args.jsonl))
    write_markdown(records, Path(args.report_md))
    render_junit(records, Path(args.junit))
    escapes = sum(1 for r in records if _is_escape(r))
    print(f"{len(records)} trials, {escapes} escapes -> {args.report_md}, {args.junit}")
    raise SystemExit(1 if escapes else 0)


if __name__ == "__main__":
    main()
