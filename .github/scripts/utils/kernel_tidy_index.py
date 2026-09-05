#!/usr/bin/env python3
"""Assemble the per-leg kernel clang-tidy CodeChecker reports into one site.

CodeChecker's HTML export has to run inside each test leg's container, where
the analyzed sources still exist (exporting from plists alone yields an index
with no report pages). So each leg uploads its own CodeChecker report and this
script only stages them side by side and writes a navigation index linking to
each leg's own CodeChecker index.html, with counts read from the leg's
`reports.json` (CodeChecker's own JSON export).

Usage:
    kernel_tidy_index.py --reports-dir <downloaded artifacts> \
        --site-dir <output> [--summary-file <GITHUB_STEP_SUMMARY>]
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import pathlib
import re
import shutil
import sys

ARTIFACT_PREFIX = "kernel-clang-tidy-html-"


def leg_name(directory: pathlib.Path) -> str:
    name = directory.name
    if name.startswith(ARTIFACT_PREFIX):
        name = name[len(ARTIFACT_PREFIX) :]
    return name


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "leg"


def read_reports(report_dir: pathlib.Path) -> list[dict]:
    """Return CodeChecker's JSON reports for a leg ([] when absent/unparseable)."""
    path = report_dir / "reports.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return []
    reports = data.get("reports") if isinstance(data, dict) else data
    return reports if isinstance(reports, list) else []


def checker_counts(reports: list[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for report in reports:
        counts[report.get("checker_name") or "(unknown)"] += 1
    return counts


def render_index(legs: list[dict], totals: collections.Counter) -> str:
    if legs:
        rows = "".join(
            "<tr>"
            f'<td><a href="{leg["slug"]}/index.html">{html.escape(leg["name"])}</a></td>'
            f'<td>{leg["count"]}</td>'
            f'<td>{html.escape(leg["top"])}</td>'
            "</tr>"
            for leg in legs
        )
        table = (
            "<table><thead><tr><th>Test group</th><th>Findings</th>"
            f"<th>Most frequent check</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        table = "<p><em>No kernel clang-tidy reports were produced by this run.</em></p>"

    if totals:
        checker_rows = "".join(
            f"<tr><td><code>{html.escape(checker)}</code></td><td>{count}</td></tr>"
            for checker, count in totals.most_common()
        )
        checkers = (
            "<h2>All checks</h2><table><thead><tr><th>Check</th><th>Findings</th>"
            f"</tr></thead><tbody>{checker_rows}</tbody></table>"
        )
    else:
        checkers = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tt-metal kernel clang-tidy</title>
<style>
 body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
 th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
 th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>tt-metal kernel clang-tidy</h1>
<p>clang-tidy over the kernels each ttnn test group JIT-compiled at runtime.
Each row links to that group's CodeChecker report.</p>
{table}
{checkers}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--site-dir", required=True)
    ap.add_argument("--summary-file")
    args = ap.parse_args()

    reports_root = pathlib.Path(args.reports_dir)
    site = pathlib.Path(args.site_dir)
    site.mkdir(parents=True, exist_ok=True)

    legs: list[dict] = []
    totals: collections.Counter = collections.Counter()

    for directory in sorted(p for p in reports_root.glob("*") if p.is_dir()) if reports_root.is_dir() else []:
        if not (directory / "index.html").is_file():
            print(f"skipping '{directory.name}': no CodeChecker index.html", file=sys.stderr)
            continue
        name = leg_name(directory)
        slug = slugify(name)
        shutil.copytree(directory, site / slug, dirs_exist_ok=True)

        reports = read_reports(directory)
        counts = checker_counts(reports)
        totals.update(counts)
        legs.append(
            {
                "name": name,
                "slug": slug,
                "count": len(reports),
                "top": counts.most_common(1)[0][0] if counts else "-",
            }
        )

    (site / "index.html").write_text(render_index(legs, totals))
    print(f"staged {len(legs)} leg report(s), {sum(totals.values())} finding(s) total")

    if args.summary_file:
        with open(args.summary_file, "a") as out:
            out.write("## Kernel clang-tidy (prototype, non-blocking)\n\n")
            if legs:
                out.write("| Test group | Findings | Most frequent check |\n|---|---|---|\n")
                for leg in legs:
                    out.write(f'| {leg["name"]} | {leg["count"]} | {leg["top"]} |\n')
                out.write(f"\n**Total: {sum(totals.values())} findings across {len(legs)} test groups.**\n")
            else:
                out.write("_No kernel clang-tidy reports were produced by this run._\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
