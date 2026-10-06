# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Finite parameter checklist and joint-combination gaps."""

import html
import json


def values_text(values: list[str]) -> str:
    return html.escape(", ".join(values)) if values else "—"


def combinations_table(combinations: list[dict]) -> str:
    if not combinations:
        return ""
    names = list(combinations[0])
    header = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(combination[name])}</td>" for name in names)
        + "</tr>"
        for combination in combinations
    )
    return f'<div class="configuration-scroll"><table class="configurations"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'


def sweep_card(row: dict) -> str:
    state = row["status"]
    titles = {
        "sweep_gap": "Candidate sweep gaps",
        "sweep_complete": "Finite candidates exercised",
        "unresolved": "Domain unresolved",
        "absent": "No emitted record",
    }
    colour = (
        "ran"
        if state == "sweep_complete"
        else "gaps" if state == "sweep_gap" else "absent"
    )
    count = f'{row["executed_combinations"]}/{row["candidate_combinations"]} candidate combinations executed · {row["assertion_excluded_combinations"]} excluded by assertions'
    if not row["parameters"]:
        count = "No automatically inferred finite domain"
    parameters = []
    for parameter in row["parameters"]:
        expected = [" / ".join(item["names"]) for item in parameter["values"]]
        missing = parameter["missing"]
        missing_text = values_text(missing)
        if not missing and parameter["seen"]:
            missing_text = "Covered"
        parameters.append(
            f'<tr><td><code>{html.escape(parameter["name"])}</code></td><td>{html.escape(parameter["type"])}</td>'
            f'<td>{values_text(expected)}</td><td>{values_text(parameter["excluded"])}</td><td>{values_text(parameter["seen"])}</td>'
            f'<td class="{"parameter-gap" if missing else "parameter-covered"}">{missing_text}</td>'
            "</tr>"
        )
    details = (
        '<p class="muted">Individual values and joint combinations are checked separately. '
        "Seeing both values of every bool does not prove all their combinations ran. "
        "Enums use distinct named values (aliases share coverage); unnamed casts/flag combinations are not inferred.</p>"
        f'<p>{html.escape(row["source"])} · build axes <code>{html.escape(json.dumps(row["build_axes"]))}</code>'
        f' · other template arguments <code>{html.escape(json.dumps(row["other_arguments"]))}</code></p>'
        '<div class="configuration-scroll"><table class="configurations"><thead><tr><th>Parameter</th><th>Type</th>'
        "<th>Declared/inferred values</th><th>Excluded values</th><th>Executed values</th><th>Untested candidate values</th>"
        "</tr></thead><tbody>" + "".join(parameters) + "</tbody></table></div>"
    )
    if row["untracked_parameters"]:
        notes = "".join(
            f'<li><code>{html.escape(p["name"])}</code> ({html.escape(p["type"])}): {html.escape(p["reason"])}</li>'
            for p in row["untracked_parameters"]
        )
        details += f"<p>Parameters without an inferred finite domain (not claimed covered):</p><ul>{notes}</ul>"
    if row["unmapped_instantiations"]:
        details += (
            "<p>Some compiler arguments could not be mapped to named domain values:</p><pre>"
            + html.escape("\n".join(row["unmapped_instantiations"]))
            + "</pre>"
        )
    if row["assertions"]:
        assertions = "".join(
            f'<li>{html.escape(a["kind"])} at line {a["line"]}: <code>{html.escape(a["condition"])}</code></li>'
            for a in row["assertions"]
        )
        details += f'<details><summary>Assertion constraints</summary><ol start="0">{assertions}</ol></details>'
    if row["constraint_unknown_combinations"]:
        details += f'<p>{row["constraint_unknown_combinations"]} combinations have unresolved constraints; they remain candidates.</p>'
    if row["excluded_preview"]:
        details += (
            "<details><summary>Excluded combinations and assertion indices (up to 32)</summary><pre>"
            + html.escape(json.dumps(row["excluded_preview"], indent=2))
            + "</pre></details>"
        )
    if row["constraint_conflicts"]:
        details += (
            "<p>Constraint filtering conflicts with compiler/execution evidence; these combinations were retained:</p><pre>"
            + html.escape(json.dumps(row["constraint_conflicts"], indent=2))
            + "</pre>"
        )
    if row["missing_combinations"]:
        details += (
            f'<h4>Missing candidate combinations: {row["missing_combinations"]}</h4>'
            "<p>Provably false local/class assertions are excluded, respecting conditional guards. "
            "Unresolved predicates and callee constraints still require compiler/runtime validation. Showing up to 32 candidates.</p>"
            + combinations_table(row["missing_preview"])
        )
    search = " ".join(
        [
            row["arch"],
            row["source"],
            row["function"],
            *(p["name"] for p in row["parameters"]),
        ]
    )
    return (
        f'<details class="function" data-thread="{html.escape(row["trisc"])}" data-unobserved="{str(not row["observed"]).lower()}" '
        f'data-states="{state}" data-search="{html.escape(search.lower())}"><summary><span class="badge {colour}">{titles[state]}</span>'
        f'<code>{html.escape(row["function"])}</code><span class="counts">{count}</span></summary>'
        f'<div class="instance-body">{details}</div></details>'
    )


def sweep_section(sweeps: dict) -> str:
    rows = sweeps.get("rows", [])
    if not rows:
        return (
            "<p>No template parameter declarations were available for this report.</p>"
        )
    counts = sweeps["counts"]
    intro = (
        f'<p class="guide"><strong>Emitted template groups:</strong> {counts["sweep_complete"]} complete, '
        f'{counts["sweep_gap"]} with gaps, {counts["unresolved"]} unresolved. '
        "Checked separately per thread, build axes, and fixed non-finite arguments. "
        "Candidates exclude combinations rejected by evaluable local/class static_assert and LLK_ASSERT predicates. "
        "Integer domains come from bounded assertion predicates where possible. Runtime values are not inferred from test-level metadata. "
        "Unobserved library templates are hidden by default.</p>"
    )
    cards = "\n".join(sweep_card(row) for row in rows)
    unresolved = sweeps.get("unresolved_definitions", [])
    notes = (
        "<details><summary>Declarations that could not be audited</summary><pre>"
        + html.escape(json.dumps(unresolved, indent=2))
        + "</pre></details>"
        if unresolved
        else ""
    )
    return (
        intro
        + f'<section class="header-group"><h2>Template parameter sweep checklist</h2>{cards}</section>'
        + notes
    )
