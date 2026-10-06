# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Combine compatible observations into a self-contained HTML/JSON report."""

import json
from collections import defaultdict
from pathlib import Path

from .baseline import source_baseline
from .gcov_json import observation_key
from .html_report import render_html
from .sweep import finite_sweeps


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[observation_key(record)].append(record)
    functions = []
    for _, observations in sorted(groups.items()):
        first = observations[0]
        executed = [record for record in observations if record["execution_count"] > 0]
        branches = {}
        for line in (line for record in observations for line in record["lines"]):
            for index, branch in enumerate(line.get("branches", [])):
                key = (line["line_number"], index)
                branches[key] = branches.get(key, False) or branch["count"] > 0
        functions.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "arch",
                        "trisc",
                        "source",
                        "symbol",
                        "signature",
                        "function",
                        "arguments",
                        "graph",
                        "blocks",
                    )
                },
                "executed": bool(executed),
                "measured": any(
                    record.get("measurement") != "build_only" for record in observations
                ),
                "build_axes": first.get("build_axes", {}),
                "definition": first.get("definition"),
                "tests": sorted(
                    {
                        record.get("run", {}).get("test") or record["test"]
                        for record in executed
                    }
                ),
                "blocks_executed_lower_bound": max(
                    record["blocks_executed"] for record in observations
                ),
                "blocks_executed_upper_bound": min(
                    first["blocks"],
                    sum(record["blocks_executed"] for record in observations),
                ),
                "branches": len(branches),
                "branches_taken": sum(branches.values()),
            }
        )
    return functions


def write_report(
    records: list[dict],
    output: Path,
    *,
    root: Path | None = None,
    architectures: list[str] | None = None,
    scans: list[dict] = (),
):
    root = root or Path(__file__).resolve().parents[4]
    architectures = (
        architectures
        if architectures is not None
        else sorted({record["arch"] for record in records})
    )
    baseline = source_baseline(records, root, architectures, scans)
    report = {
        "schema_version": 3,
        "instantiations": summarize(records),
        "baseline": baseline,
        "finite_sweeps": finite_sweeps(baseline, records),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "instantiations.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "index.html").write_text(render_html(report, records))
