# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Template parameter sweeps: an overview of every template group and one combination matrix per group."""

import html
import json

STATES = {
    "executed": "Executed",
    "compiled": "Compiled, not executed",
    "missing": "Not instantiated",
    "excluded": "Excluded by assertion",
    "conflict": "Assertion conflict",
}

TITLES = {
    "sweep_gap": "Candidate sweep gaps",
    "sweep_complete": "Finite candidates exercised",
    "unresolved": "Domain unresolved",
    "absent": "No emitted record",
}


def values_text(values: list[str]) -> str:
    return html.escape(", ".join(values)) if values else "—"


def card_id(row: dict) -> str:
    return "sweep-" + html.escape(
        "-".join(
            [row["trisc"], row["function"], *sorted(row["other_arguments"].values())]
        ).replace(" ", "_")
    )


def colour(state: str) -> str:
    return (
        "ran"
        if state == "sweep_complete"
        else "gaps" if state == "sweep_gap" else "absent"
    )


def progress(done: int, total: int) -> str:
    percent = round(100 * done / total) if total else 0
    return (
        f'<span class="progress" title="{done} of {total}"><span style="width:{percent}%"></span></span>'
        f'<span class="progress-label">{done}/{total}</span>'
    )


def parameter_table(row: dict) -> str:
    lines = []
    for parameter in row["parameters"]:
        declared = [" / ".join(item["names"]) for item in parameter["values"]]
        candidates = [value for value in declared if value not in parameter["excluded"]]
        lines.append(
            f'<tr><td><code>{html.escape(parameter["name"])}</code><br><span class="muted">{html.escape(parameter["type"])}</span></td>'
            f'<td>{progress(len(parameter["seen"]), len(candidates))}</td>'
            f'<td class="parameter-covered">{values_text(parameter["seen"])}</td>'
            f'<td class="{"parameter-gap" if parameter["missing"] else ""}">{values_text(parameter["missing"])}</td>'
            f'<td class="muted">{values_text(parameter["excluded"])}</td></tr>'
        )
    for parameter in row["untracked_parameters"]:
        lines.append(
            f'<tr class="muted"><td><code>{html.escape(parameter["name"])}</code><br><span class="muted">{html.escape(parameter["type"])}</span></td>'
            f'<td colspan="4">No finite domain: {html.escape(parameter["reason"])}</td></tr>'
        )
    return (
        '<div class="configuration-scroll"><table class="configurations parameters"><thead><tr>'
        "<th>Parameter</th><th>Values executed</th><th>Executed</th><th>Not yet executed</th><th>Excluded</th>"
        "</tr></thead><tbody>" + "".join(lines) + "</tbody></table></div>"
    )


def combination_table(row: dict) -> str:
    combinations = row["combinations"]
    if not combinations:
        return ""
    assertions = row["assertions"]
    names = list(combinations[0]["values"])
    header = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    lines = []
    for combination in combinations:
        state = combination["status"]
        detail = ""
        if state == "excluded":
            detail = "; ".join(
                html.escape(assertions[i]["condition"])
                for i in combination["assertions"]
            )
        elif state == "executed":
            detail = ", ".join(
                html.escape(test.split("[", 1)[0].rsplit("::", 1)[-1])
                for test in combination["tests"]
            )
        cells = "".join(
            f"<td>{html.escape(combination['values'][name])}</td>" for name in names
        )
        lines.append(
            f'<tr class="combo {state}" data-state="{state}">{cells}'
            f'<td><span class="badge {state}">{STATES[state]}</span></td><td class="detail">{detail}</td></tr>'
        )
    counts = {
        state: sum(c["status"] == state for c in combinations) for state in STATES
    }
    # a group with gaps opens on what is missing; "All" is one click away
    initial = "missing" if counts["missing"] else ""
    buttons = (
        f'<button type="button" class="combo-filter{" active" if not initial else ""}" data-state="">All</button>'
        + "".join(
            f'<button type="button" class="combo-filter{" active" if state == initial else ""}" data-state="{state}">{STATES[state]} <b>{count}</b></button>'
            for state, count in counts.items()
            if count
        )
    )
    if initial:
        lines = [
            (
                line
                if f'data-state="{initial}"' in line
                else line.replace("<tr ", "<tr hidden ", 1)
            )
            for line in lines
        ]
    note = (
        ""
        if row["combinations_listed"]
        else f'<p class="muted">Showing the first {len(combinations)} of {row["declared_combinations"]} declared combinations.</p>'
    )
    return (
        f'<h4>All template combinations</h4><div class="combo-filters">{buttons}</div>{note}'
        '<div class="configuration-scroll combo-scroll"><table class="configurations combinations"><thead><tr>'
        f"{header}<th>Status</th><th>Detail</th></tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div>"
    )


def sweep_card(row: dict) -> str:
    state = row["status"]
    count = (
        f'{row["executed_combinations"]}/{row["candidate_combinations"]} combinations executed'
        + (
            f' · {row["assertion_excluded_combinations"]} excluded by assertions'
            if row["assertion_excluded_combinations"]
            else ""
        )
    )
    if not row["parameters"]:
        count = "No automatically inferred finite domain"
    context = f'{html.escape(row["source"])} · thread {html.escape(row["trisc"])}'
    if row["build_axes"]:
        context += (
            f' · build axes <code>{html.escape(json.dumps(row["build_axes"]))}</code>'
        )
    if row["other_arguments"]:
        context += f' · fixed arguments <code>{html.escape(json.dumps(row["other_arguments"]))}</code>'
    details = (
        f'<p class="path">{context}</p>' + parameter_table(row) + combination_table(row)
    )
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
        details += f"<details><summary>Assertion constraints</summary><ol>{assertions}</ol></details>"
    if row["constraint_unknown_combinations"]:
        details += f'<p class="muted">{row["constraint_unknown_combinations"]} combinations have unresolved constraints; they remain candidates.</p>'
    if row["constraint_conflicts"]:
        details += (
            "<p>Constraint filtering conflicts with compiler/execution evidence; these combinations were retained:</p><pre>"
            + html.escape(json.dumps(row["constraint_conflicts"], indent=2))
            + "</pre>"
        )
    details += (
        '<p class="muted">Executed means a test ran that instantiation. Combinations are checked jointly: seeing every value of a '
        "parameter does not prove every combination ran. Excluded combinations fail an evaluable assertion; unresolved predicates "
        "and callee constraints still need compiler or runtime validation.</p>"
    )
    search = " ".join(
        [
            row["arch"],
            row["source"],
            row["function"],
            row["trisc"],
            *(p["name"] for p in row["parameters"]),
        ]
    )
    return (
        f'<details class="function" id="{card_id(row)}" data-thread="{html.escape(row["trisc"])}" data-unobserved="{str(not row["observed"]).lower()}" '
        f'data-states="{state}" data-search="{html.escape(search.lower())}"><summary><span class="badge {colour(state)}">{TITLES[state]}</span>'
        f'<code>{html.escape(row["function"])}</code><span class="counts">{count}</span></summary>'
        f'<div class="instance-body">{details}</div></details>'
    )


def overview_table(rows: list[dict]) -> str:
    lines = []
    for row in sorted(
        rows, key=lambda r: (-r["missing_combinations"], r["function"], r["trisc"])
    ):
        if not row["parameters"]:
            continue
        axes = ", ".join(f'{p["name"]} ({len(p["values"])})' for p in row["parameters"])
        lines.append(
            f'<tr class="overview-row" data-unobserved="{str(not row["observed"]).lower()}" data-thread="{html.escape(row["trisc"])}">'
            f'<td><a href="#{card_id(row)}"><code>{html.escape(row["function"])}</code></a></td>'
            f'<td>{html.escape(row["trisc"])}</td><td>{html.escape(axes)}</td>'
            f'<td>{progress(row["executed_combinations"], row["candidate_combinations"])}</td>'
            f'<td>{row["missing_combinations"]}</td><td>{row["assertion_excluded_combinations"]}</td>'
            f'<td><span class="badge {colour(row["status"])}">{TITLES[row["status"]]}</span></td></tr>'
        )
    return (
        '<details class="overview" open><summary>All template groups at a glance</summary>'
        '<div class="configuration-scroll"><table class="configurations overview"><thead><tr>'
        "<th>Function</th><th>Thread</th><th>Finite parameters (values)</th><th>Combinations executed</th>"
        "<th>Missing</th><th>Excluded</th><th>Status</th></tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div></details>"
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
        "Each group is one template in one thread and build context with its non-finite arguments fixed. "
        "Open a group to see every combination of its bool, enum and bounded integer parameters with what ran, "
        "what only compiled, what no test instantiated and what an assertion rules out. "
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
        + overview_table(rows)
        + f'<section class="header-group"><h2>Template parameter sweep checklist</h2>{cards}</section>'
        + notes
    )
