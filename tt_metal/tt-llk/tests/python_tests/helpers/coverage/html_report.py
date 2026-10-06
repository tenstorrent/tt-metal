# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A self-contained dashboard over observed function and branch coverage."""

import html
import json
from collections import defaultdict
from pathlib import Path
from string import Template

from .gcov_json import observation_key
from .run_table import run_inputs, test_summary
from .sweep_html import sweep_section


def status(row: dict) -> str:
    if not row.get("measured", True):
        return "unmeasured"
    if not row["executed"]:
        return "missed"
    if row["branches_taken"] < row["branches"]:
        return "gaps"
    return "ran"


LABELS = {
    "ran": "Ran",
    "gaps": "Ran · branch gaps",
    "missed": "Measured · zero executions",
    "absent": "No emitted coverage record",
    "unmeasured": "Emitted · no runtime data",
    "stale": "Source snapshot changed",
}


def badge(state: str) -> str:
    return f'<span class="badge {state}">{LABELS[state]}</span>'


def source_table(observations: list[dict]) -> str:
    lines = {}
    for observation in observations:
        for line in observation["lines"]:
            number = line["line_number"]
            entry = lines.setdefault(
                number, {"hit": False, "source": "", "branches": {}}
            )
            entry["hit"] |= line["count"] > 0
            entry["source"] = line.get("source_text", entry["source"])
            for index, branch in enumerate(line.get("branches", [])):
                entry["branches"][index] = (
                    entry["branches"].get(index, False) or branch["count"] > 0
                )
    rows = []
    for number, line in sorted(lines.items()):
        hit = line["hit"]
        branches = line["branches"]
        gaps = bool(branches) and not all(branches.values())
        state = "missed" if not hit else "gaps" if gaps else "ran"
        branch_label = f"{sum(branches.values())}/{len(branches)}" if branches else "—"
        rows.append(
            f'<tr class="{state}"><td>{number}</td><td>{"Hit" if hit else "Miss"}</td>'
            f'<td>{branch_label}</td><td><code>{html.escape(line["source"])}</code></td></tr>'
        )
    if not rows:
        return '<p class="muted">No per-line records emitted for this function.</p>'
    return (
        '<div class="source-scroll"><table class="source"><thead><tr>'
        "<th>Line</th><th>Executed</th><th>Branches hit</th><th>Source</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def instantiation_card(row: dict, observations: list[dict]) -> str:
    state = status(row)
    tests = test_summary(row["tests"])
    low, high = row["blocks_executed_lower_bound"], row["blocks_executed_upper_bound"]
    blocks = str(low) if low == high else f"{low}–{high}"
    variants = sorted({r["variant"] for r in observations})
    variant_list = "".join(
        f"<li><code>{html.escape(value)}</code></li>" for value in variants
    )
    args = ", ".join(row["arguments"]) or "No template arguments"
    return (
        f'<details class="instantiation" data-status="{state}">'
        f"<summary>{badge(state)} <code>{html.escape(args)}</code>"
        f'<span class="counts">{row["branches_taken"]}/{row["branches"]} branches · '
        f"{len(variants)} compiled variants</span></summary>"
        f'<div class="instance-body"><p><code>{html.escape(row["signature"])}</code></p>'
        f"<p><strong>Run by:</strong> {html.escape(tests)}</p>"
        f'<p class="muted">Executed blocks: {blocks}/{row["blocks"]}. '
        "A range means the exact union across captures is unknown. "
        "Lines below combine this instantiation’s matching instrumentation layout only.</p>"
        + source_table(observations)
        + run_inputs(
            [record for record in observations if record["execution_count"] > 0]
        )
        + '<details class="technical"><summary>Build details</summary>'
        f'<p>Build axes: <code>{html.escape(json.dumps(row["build_axes"], sort_keys=True))}</code></p>'
        f'<p>Instrumentation layout: <code>{html.escape(row["graph"])}</code></p>'
        f"<ul>{variant_list}</ul></details></div></details>"
    )


def definition_card(definition: dict) -> str:
    state = definition["status"]
    source = "\n".join(
        f"{definition['start_line'] + offset:5}  {line}"
        for offset, line in enumerate(definition["source_lines"])
    )
    search = f"{definition['arch']} {definition['header']} {definition['function']} {' '.join(definition['template_parameters'])}"
    threads = " ".join(definition["threads"])
    count = f"{definition['executed_instantiations']}/{definition['emitted_instantiations']} emitted instantiations ran"
    inspect = (
        f'<button type="button" data-inspect="{html.escape(definition["function"])}">View captured instances</button>'
        if definition["emitted_instantiations"]
        else ""
    )
    runtime = ", ".join(
        parameter["declaration"] for parameter in definition["runtime_parameters"]
    )
    runtime_domains = "".join(
        f'<li><code>{html.escape(parameter["name"])}</code>: '
        f'{html.escape(", ".join(" / ".join(value["names"]) for value in parameter["values"]))} '
        f'({html.escape(parameter.get("domain_source", "named values"))})</li>'
        for parameter in definition["runtime_parameters"]
        if parameter["values"]
    )
    assertions = "".join(
        f'<li>{html.escape(assertion["kind"])} at line {assertion["line"]}: <code>{html.escape(assertion["condition"])}</code></li>'
        for assertion in definition.get("assertions", [])
    )
    decisions = "".join(
        f'<li>Line {point["line"]}:{point["column"]}: {"constexpr " if point["compile_time"] else ""}{html.escape(point["kind"])} '
        f'<code>{html.escape(point["condition"])}</code></li>'
        for point in definition["decisions"]
    )
    return (
        f'<details class="function" data-thread="{threads}" data-states="{state}" data-search="{html.escape(search.lower())}">'
        f'<summary>{badge(state)} <code>{html.escape(definition["function"])}</code><span class="counts">{count}</span></summary>'
        f'<div class="instance-body"><p>{html.escape(definition["header"])}:{definition["line"]}</p>'
        f'<p><strong>Template parameters:</strong> <code>{html.escape(", ".join(definition["template_parameters"]) or "none")}</code></p>'
        f'<p><strong>Runtime parameters:</strong> <code>{html.escape(runtime or "none")}</code></p>'
        f"<ul>{runtime_domains}</ul>"
        f'<details><summary>{len(definition.get("assertions", []))} assertion constraints</summary><ul>{assertions}</ul></details>'
        f'<p>Thread: {html.escape(definition["trisc"])} · Build axes: <code>{html.escape(json.dumps(definition["build_axes"], sort_keys=True))}</code></p>'
        '<p class="muted">AST definition in this build context, including uninstantiated templates. '
        "Decision inventory is static; runtime hits come from each captured instantiation’s gcov branches. "
        "An absent record may mean unused code or compiler optimization.</p>"
        f'<details><summary>{len(definition["decisions"])} AST decisions</summary><ul>{decisions}</ul></details>'
        f'{inspect}<pre class="definition-source">{html.escape(source)}</pre></div></details>'
    )


def baseline_section(baseline: dict) -> str:
    headers = defaultdict(list)
    for definition in baseline.get("definitions", []):
        headers[(definition["arch"], definition["header"])].append(
            definition_card(definition)
        )
    sections = [
        '<p class="guide">AST baseline: includes definitions with no gcov record. '
        "Active preprocessor branches are collected per build context; this is not a percentage of legal instantiations.</p>"
    ]
    sections.append(
        f'<p class="muted">Scope: {html.escape(baseline.get("scope", "no source census loaded"))}. '
        f'{html.escape(baseline["method"])}.</p>'
    )
    if not baseline["complete"]:
        sections.append(
            "<p><strong>Incomplete AST baseline.</strong> Unparsed headers, failed scans or unmatched observations remain.</p>"
        )
    for key, title in (
        ("unparsed_headers", "Unparsed headers by architecture"),
        ("incomplete_scans", "Incomplete AST scans"),
    ):
        sections.append(
            f"<details><summary>{title}</summary><pre>{html.escape(json.dumps(baseline[key], indent=2))}</pre></details>"
        )
    for (arch, header), cards in sorted(headers.items()):
        sections.append(
            f'<section class="header-group"><h2>{html.escape(arch)} / {html.escape(header)}</h2>'
            + "\n".join(cards)
            + "</section>"
        )
    if not headers:
        sections.append(
            "<p>No complete AST scans loaded. Rebuild with coverage to capture definitions.</p>"
        )
    if baseline.get("unmatched_observations"):
        sections.append(
            "<details><summary>Observations outside the census or without an unambiguous source match</summary>"
            f'<pre>{html.escape(json.dumps(baseline["unmatched_observations"], indent=2))}</pre></details>'
        )
    return "\n".join(sections)


def function_card(rows: list[dict], observations: dict) -> str:
    first = rows[0]
    ran = sum(row["executed"] for row in rows)
    branches = sum(row["branches"] for row in rows)
    hit = sum(row["branches_taken"] for row in rows)
    gaps = any(status(row) == "gaps" for row in rows)
    states = " ".join(sorted({status(row) for row in rows}))
    state = "missed" if not ran else "gaps" if gaps or ran < len(rows) else "ran"
    search = " ".join(
        [
            first["function"],
            first["source"],
            first["arch"],
            first["trisc"],
            *[row["signature"] for row in rows],
            *[test for row in rows for test in row["tests"]],
        ]
    )
    cards = "".join(
        instantiation_card(row, observations[observation_key(row)]) for row in rows
    )
    return (
        f'<details class="function" data-thread="{html.escape(first["trisc"])}" '
        f'data-states="{states}" data-search="{html.escape(search.lower())}">'
        f'<summary><span class="dot {state}" aria-hidden="true"></span>'
        f'<code>{html.escape(first["function"])}</code><span class="counts">'
        f"{ran}/{len(rows)} entries ran · {hit}/{branches} branches</span></summary>"
        f'<div class="function-body">{cards}</div></details>'
    )


def render_html(report: dict, records: list[dict]) -> str:
    observations = defaultdict(list)
    for record in records:
        observations[observation_key(record)].append(record)
    functions = defaultdict(list)
    for row in report["instantiations"]:
        functions[(row["arch"], row["trisc"], row["source"], row["function"])].append(
            row
        )
    headers = defaultdict(list)
    for key, rows in sorted(functions.items()):
        headers[key[:3]].append(function_card(rows, observations))
    sections = []
    for (arch, thread, source), cards in sorted(headers.items()):
        sections.append(
            '<section class="header-group">'
            f'<h2><span class="thread">{html.escape(arch)} / {html.escape(thread)}</span>'
            f"{html.escape(Path(source).name)}</h2>"
            f'<p class="path">{html.escape(source)}</p>' + "".join(cards) + "</section>"
        )
    rows = report["instantiations"]
    ran_functions = sum(
        any(row["executed"] for row in group) for group in functions.values()
    )
    stats = [
        (ran_functions, f"of {len(functions)} observed functions ran"),
        (
            sum(row["executed"] for row in rows),
            f"of {len(rows)} instrumented entries ran",
        ),
        (
            sum(status(row) == "gaps" for row in rows),
            "executed entries with branch gaps",
        ),
        (
            sum(not row["executed"] for row in rows),
            "instrumented entries without observed execution",
        ),
    ]
    census = report.get("baseline", {})
    if census:
        stats.extend(
            [
                (
                    len(census["definitions"]),
                    "independently discovered source definitions",
                ),
                (
                    census["counts"]["absent"],
                    "source definitions with no emitted coverage record",
                ),
            ]
        )
    tiles = "".join(
        f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in stats
    )
    threads = "".join(
        f'<option value="{html.escape(thread)}">{html.escape(thread)}</option>'
        for thread in sorted({row["trisc"] for row in rows})
    )
    template = Template(Path(__file__).with_name("dashboard.html").read_text())
    return template.substitute(
        stats=tiles,
        threads=threads,
        sections="\n".join(sections),
        baseline=baseline_section(census),
        runs=run_inputs(records),
        sweeps=sweep_section(report.get("finite_sweeps", {})),
    )
