# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A self-contained dashboard over observed function and branch coverage."""

import html
import json
from collections import defaultdict
from pathlib import Path
from string import Template

from . import sweep_html
from .ast_scan import llk_ast
from .gcov_json import observation_key
from .run_table import run_inputs, test_summary
from .sweep_html import sweep_section


def spelled_arguments(
    arguments: list[str], definition: dict | None
) -> list[tuple[str, str, str]]:
    """(name, label, C++ spelling) per template argument; `(ckernel::DstSync)0` becomes SyncHalf / ckernel::DstSync::SyncHalf."""
    if not definition or len(definition["parameters"]) != len(arguments):
        return [("", argument, argument) for argument in arguments]
    function = llk_ast.function_from_json(json.dumps(definition))
    spelled = []
    for parameter, argument in zip(function.parameters, arguments):
        value = llk_ast.argument_value(parameter, argument)
        names = next((v.names for v in parameter.values if v.value == value), None)
        if value is None:
            spelled.append((parameter.name, argument, argument))
        elif names and parameter.kind == "enum":
            spelled.append(
                (
                    parameter.name,
                    " / ".join(names),
                    f"{parameter.enum_type}::{names[0]}",
                )
            )
        elif names:
            spelled.append((parameter.name, names[0], names[0]))
        else:
            spelled.append((parameter.name, str(value), str(value)))
    return spelled


def argument_labels(arguments: list[str], definition: dict | None) -> str:
    return ", ".join(
        f"{name}={label}" if name else label
        for name, label, _ in spelled_arguments(arguments, definition)
    )


def signature(row: dict, definition: dict | None) -> str:
    """The instantiation as it would be written in C++, from the AST definition when it is known."""
    if not definition:
        return row["signature"]
    arguments = ", ".join(
        spelling for _, _, spelling in spelled_arguments(row["arguments"], definition)
    )
    runtime = ", ".join(
        parameter["declaration"] for parameter in definition["runtime_parameters"]
    )
    name = definition.get("qualified_name") or definition["function"]
    prefix = f'{definition["return_type"]} ' if definition.get("return_type") else ""
    return (
        f"{prefix}{name}<{arguments}>({runtime})"
        if arguments
        else f"{prefix}{name}({runtime})"
    )


def status(row: dict) -> str:
    if not row.get("measured", True):
        return "unmeasured"
    if not row["executed"]:
        return "missed"
    if row["branches_taken"] < row["branches"]:
        return "gaps"
    return "ran"


LABELS = {
    "ran": "Ran in tests",
    "gaps": "Ran, branches missed",
    "missed": "Compiled, never ran",
    "absent": "Not compiled by any test",
    "unmeasured": "Compiled, no run data",
    "stale": "Source changed since scan",
}


def axes_text(axes: dict) -> str:
    """Build axes as words; empty values are the default variant."""
    items = [f"{key}: {value or 'default'}" for key, value in sorted(axes.items())]
    return ", ".join(items) if items else "default"


def canonical_name(source: str) -> str:
    while source.startswith("../"):
        source = source[3:]
    return source


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


def instantiation_card(
    row: dict, observations: list[dict], definition: dict | None
) -> str:
    state = status(row)
    tests = test_summary(row["tests"])
    low, high = row["blocks_executed_lower_bound"], row["blocks_executed_upper_bound"]
    blocks = str(low) if low == high else f"{low}–{high}"
    variants = sorted({r["variant"] for r in observations})
    variant_list = "".join(
        f"<li><code>{html.escape(value)}</code></li>" for value in variants
    )
    args = argument_labels(row["arguments"], definition) or "No template arguments"
    counts = (
        f'{row["branches_taken"]}/{row["branches"]} branches hit'
        if row["branches"]
        else ""
    )
    return (
        f'<details class="instantiation" data-status="{state}">'
        f"<summary>{badge(state)} <code>{html.escape(args)}</code>"
        f'<span class="counts">{counts}</span></summary>'
        f'<div class="instance-body"><p><code>{html.escape(signature(row, definition))}</code></p>'
        f"<p><strong>Run by:</strong> {html.escape(tests)}</p>"
        f'<p class="muted">Executed blocks: {blocks}/{row["blocks"]}. '
        "A range means the exact union across captures is unknown. "
        "Lines below combine this instantiation’s matching instrumentation layout only.</p>"
        + source_table(observations)
        + run_inputs(
            [record for record in observations if record["execution_count"] > 0]
        )
        + f'<details class="technical"><summary>Build details · {len(variants)} compiled variants</summary>'
        f'<p>Build axes: {html.escape(axes_text(row["build_axes"]))}</p>'
        f'<p>Instrumentation layout: <code>{html.escape(row["graph"])}</code></p>'
        f"<ul>{variant_list}</ul></details></div></details>"
    )


STATUS_ORDER = ("stale", "ran", "missed", "unmeasured", "absent")


def merged_definition(copies: list[dict]) -> dict:
    """One card per function: the per-thread scans of the same definition folded together."""
    statuses = {copy["status"] for copy in copies}
    return {
        **copies[0],
        "status": next(state for state in STATUS_ORDER if state in statuses),
        "states": sorted(statuses),
        "threads": sorted({copy["trisc"] for copy in copies}),
        "emitted_instantiations": sum(
            copy["emitted_instantiations"] for copy in copies
        ),
        "executed_instantiations": sum(
            copy["executed_instantiations"] for copy in copies
        ),
        "tests": sorted({test for copy in copies for test in copy["tests"]}),
    }


def definition_title(definition: dict) -> str:
    runtime = ", ".join(
        parameter["declaration"] for parameter in definition["runtime_parameters"]
    )
    template = ", ".join(definition["template_parameters"])
    return (
        f'{definition["function"]}<{template}>({runtime})'
        if template
        else f'{definition["function"]}({runtime})'
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
        f'<details class="function" data-thread="{threads}" data-states="{" ".join(definition.get("states", [state]))}" data-search="{html.escape(search.lower())}">'
        f'<summary>{badge(state)} <code>{html.escape(definition_title(definition))}</code><span class="counts">{count}</span></summary>'
        f'<div class="instance-body"><p>{html.escape(definition["header"])}:{definition["line"]}</p>'
        f'<p><strong>Template parameters:</strong> <code>{html.escape(", ".join(definition["template_parameters"]) or "none")}</code></p>'
        f'<p><strong>Runtime parameters:</strong> <code>{html.escape(runtime or "none")}</code></p>'
        f"<ul>{runtime_domains}</ul>"
        f'<details><summary>{len(definition.get("assertions", []))} assertion constraints</summary><ul>{assertions}</ul></details>'
        f'<p>Threads: {html.escape(", ".join(definition["threads"]))} · Build axes: {html.escape(axes_text(definition["build_axes"]))}</p>'
        '<p class="muted">AST definition in this build context, including uninstantiated templates. '
        "Decision inventory is static; runtime hits come from each captured instantiation’s gcov branches. "
        "An absent record may mean unused code or compiler optimization.</p>"
        f'<details><summary>{len(definition["decisions"])} AST decisions</summary><ul>{decisions}</ul></details>'
        f'{inspect}<pre class="definition-source">{html.escape(source)}</pre></div></details>'
    )


def baseline_section(baseline: dict) -> str:
    grouped = defaultdict(list)
    for definition in baseline.get("definitions", []):
        grouped[
            (
                definition["arch"],
                definition["header"],
                definition["line"],
                definition["id"],
            )
        ].append(definition)
    headers = defaultdict(list)
    for (arch, header, _, _), copies in sorted(grouped.items()):
        headers[(arch, header)].append(definition_card(merged_definition(copies)))
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


def missed_combinations(sweeps: list[dict]) -> str:
    """What the sweep knows about this function: combinations ran, missed, and the first missed ones spelled out."""
    if not sweeps:
        return ""
    ran = sum(row["executed_combinations"] for row in sweeps)
    candidates = sum(row["candidate_combinations"] for row in sweeps)
    missing = [
        c
        for row in sweeps
        for c in row["combinations"]
        if c["status"] in ("missing", "compiled")
    ]
    link = f'<a href="#{sweep_html.card_id(sweeps[0])}" class="sweep-link">open the template sweep</a>'
    if not candidates:
        return ""
    if not missing:
        return f'<p class="sweep-summary covered">All {candidates} template combinations ran · {link}</p>'
    chips = "".join(
        f'<li><code>{html.escape(", ".join(f"{name}={value}" for name, value in c["values"].items()))}</code></li>'
        for c in missing[:12]
    )
    more = f"<li>… {len(missing) - 12} more</li>" if len(missing) > 12 else ""
    return (
        f'<div class="sweep-summary gap"><p><strong>{len(missing)} of {candidates} template combinations never ran</strong> '
        f'({ran} ran) · {link}</p><ul class="chips">{chips}{more}</ul></div>'
    )


def function_card(
    rows: list[dict], observations: dict, definitions: dict, sweeps: list[dict] = ()
) -> str:
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
        instantiation_card(
            row,
            observations[observation_key(row)],
            definitions.get(row.get("definition")),
        )
        for row in rows
    )
    missing = sum(row["missing_combinations"] for row in sweeps)
    counts = f"{ran}/{len(rows)} instantiations ran"
    if branches:
        counts += f" · {hit}/{branches} branches hit"
    if missing:
        counts += (
            f' · <span class="gap-count">{missing} template combinations missed</span>'
        )
    return (
        f'<details class="function" data-thread="{html.escape(first["trisc"])}" data-rank="{STATE_RANK[state]}" '
        f'data-states="{states}" data-search="{html.escape(search.lower())}">'
        f'<summary><span class="dot {state}" aria-hidden="true"></span>'
        f'<code>{html.escape(first["function"])}</code><span class="counts">{counts}</span></summary>'
        f'<div class="function-body">{missed_combinations(sweeps)}{cards}</div></details>'
    )


STATE_RANK = {"missed": 0, "gaps": 1, "ran": 2}


def render_html(report: dict, records: list[dict]) -> str:
    observations = defaultdict(list)
    for record in records:
        observations[observation_key(record)].append(record)
    functions = defaultdict(list)
    for row in report["instantiations"]:
        functions[(row["arch"], row["trisc"], row["source"], row["function"])].append(
            row
        )
    definitions = {
        definition["key"]: definition
        for definition in report.get("baseline", {}).get("definitions", [])
    }
    sweeps = defaultdict(list)
    for row in report.get("finite_sweeps", {}).get("rows", []):
        if row["observed"]:
            sweeps[
                (row["arch"], row["trisc"], Path(row["source"]).name, row["function"])
            ].append(row)
    headers = defaultdict(list)
    for key, rows in sorted(functions.items()):
        arch, thread, source, function = key
        card = function_card(
            rows,
            observations,
            definitions,
            sweeps.get((arch, thread, Path(source).name, function), []),
        )
        rank = int(card.split('data-rank="', 1)[1][0])
        headers[key[:3]].append((rank, card))
    sections = []
    for (arch, thread, source), cards in sorted(headers.items()):
        sections.append(
            '<section class="header-group">'
            f'<h2><span class="thread">{html.escape(arch)} / {html.escape(thread)}</span>'
            f"{html.escape(canonical_name(source))}</h2>"
            + "".join(card for _, card in sorted(cards, key=lambda item: item[0]))
            + "</section>"
        )
    rows = report["instantiations"]
    ran_functions = sum(
        any(row["executed"] for row in group) for group in functions.values()
    )
    missed_combinations_total = sum(
        row["missing_combinations"] for group in sweeps.values() for row in group
    )
    stats = [
        (ran_functions, f"of {len(functions)} compiled functions ran"),
        (sum(row["executed"] for row in rows), f"of {len(rows)} instantiations ran"),
        (missed_combinations_total, "template combinations never ran"),
        (
            sum(status(row) == "gaps" for row in rows),
            "instantiations with branches never taken",
        ),
    ]
    census = report.get("baseline", {})
    if census:
        merged = defaultdict(list)
        for definition in census["definitions"]:
            merged[(definition["header"], definition["line"], definition["id"])].append(
                definition
            )
        stats.extend(
            [
                (len(merged), "function definitions found in the headers"),
                (
                    sum(
                        all(d["status"] == "absent" for d in copies)
                        for copies in merged.values()
                    ),
                    "definitions no test compiled",
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
