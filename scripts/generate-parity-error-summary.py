#!/usr/bin/env python3
"""Build ERRORS.md from parity sweep summary.tsv and per-suite logs."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def tail_errors(log_path: Path, max_lines: int = 40) -> str:
    if not log_path.is_file():
        return "(log file missing)\n"
    lines = log_path.read_text(errors="replace").splitlines()
    interesting = [
        ln
        for ln in lines
        if re.search(
            r"FAILED|Failure|ERROR|error:|Segmentation|Aborted|TIMEOUT|"
            r"Fatal|FATAL|what\(\)|Exception|SKIPPED.*SetUp",
            ln,
            re.I,
        )
    ]
    if interesting:
        snippet = interesting[-max_lines:]
    else:
        snippet = lines[-max_lines:] if lines else ["(empty log)"]
    return "\n".join(f"    {ln}" for ln in snippet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--summary", default="summary.tsv")
    parser.add_argument("--output", default="ERRORS.md")
    args = parser.parse_args()

    results = args.results_dir
    summary_path = results / args.summary
    out_path = results / args.output

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    if summary_path.is_file():
        with summary_path.open() as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                suite, status, rc, dur, cmd = parts[0], parts[1], parts[2], parts[3], parts[4]
                rows.append(
                    {
                        "suite": suite,
                        "status": status,
                        "rc": rc,
                        "dur": dur,
                        "cmd": cmd,
                    }
                )
                counts[status] += 1

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Parity Test Error Summary",
        "",
        f"Generated: {now}",
        f"Results directory: `{results}`",
        "",
        "## Scorecard",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status in sorted(counts):
        lines.append(f"| {status} | {counts[status]} |")
    if not counts:
        lines.append("| (no results yet) | 0 |")

    failures = [r for r in rows if r["status"] in ("FAIL", "TIMEOUT", "MISSING")]
    passes = [r for r in rows if r["status"] == "PASS"]
    skips = [r for r in rows if r["status"] == "SKIP"]

    lines.extend(
        [
            "",
            f"**Total suites:** {len(rows)} | **PASS:** {len(passes)} | "
            f"**FAIL/TIMEOUT/MISSING:** {len(failures)} | **SKIP:** {len(skips)}",
            "",
            "## Failures and timeouts",
            "",
        ]
    )

    if not failures:
        lines.append("_No failures recorded._")
    else:
        for r in failures:
            log_name = r["suite"].replace("/", "__").replace(" ", "__") + ".log"
            log_path = results / log_name
            lines.extend(
                [
                    f"### `{r['suite']}` — {r['status']} (rc={r['rc']}, {r['dur']}s)",
                    "",
                    f"```bash",
                    r["cmd"],
                    "```",
                    "",
                    "<details><summary>Log excerpt</summary>",
                    "",
                    "```text",
                    tail_errors(log_path),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

    lines.extend(["## Skipped suites", ""])
    if not skips:
        lines.append("_None._")
    else:
        for r in skips:
            lines.append(f"- `{r['suite']}` — {r['cmd']}")

    lines.extend(
        [
            "",
            "## Tail live output",
            "",
            "```bash",
            f"tail -f {results / 'run.log'}",
            "```",
            "",
        ]
    )

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
