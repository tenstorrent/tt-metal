# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readable test configurations, retaining raw provenance behind details."""

import html
import json
import re
from collections import defaultdict


def test_name(node_id: str) -> str:
    return node_id.split("[", 1)[0].rsplit("::", 1)[-1].rsplit("/", 1)[-1]


def test_summary(names: list[str]) -> str:
    if not names:
        return "No execution recorded"
    tests = sorted({test_name(name) for name in names})
    return f"{len(names)} recorded case(s) · " + ", ".join(tests)


def display_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parameters(run: dict) -> dict:
    values = defaultdict(list)
    for group in (run.get("runtime_parameters"), run.get("template_configuration")):
        for item in group if isinstance(group, list) else [group]:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if value not in values[key]:
                    values[key].append(value)
    return {
        key: ", ".join(display_value(value) for value in items)
        for key, items in values.items()
    }


def input_size(run: dict) -> str:
    size = run.get("input_dimensions")
    if isinstance(size, (list, tuple)) and all(isinstance(n, int) for n in size):
        return " × ".join(map(str, size))
    # Older metadata kept dimensions only in the explicitly named pytest axis.
    node = run.get("test", "")
    if not re.search(r"\[(?:\w*_dims|input_dimensions|dimensions):", node):
        return "—"
    match = re.search(r"\[([0-9]+(?:,\s*[0-9]+)+)\]\)?\]$", node)
    return " × ".join(part.strip() for part in match[1].split(",")) if match else "—"


def format_value(run: dict, field: str) -> str:
    formats = run.get("formats") or []
    if isinstance(formats, dict):
        formats = [formats]
    values = [display_value(step.get(field)) for step in formats]
    if not values:
        return "—"
    if len(values) == 1:
        return values[0]
    return "; ".join(f"Step {index}: {value}" for index, value in enumerate(values, 1))


def configuration_row(record: dict, number: int) -> str:
    run = record["run"]
    fields = parameters(run)
    cells = [
        number,
        format_value(run, "unpack_A_src"),
        format_value(run, "unpack_B_src"),
        format_value(run, "pack_dst"),
        input_size(run),
        fields.get("tile_cnt", "—"),
        fields.get("dest_sync", "—"),
        fields.get("implied_math_format", "—"),
    ]
    rendered = "".join(f"<td>{html.escape(str(value))}</td>" for value in cells)
    formats = run.get("formats") or []
    formats = [formats] if isinstance(formats, dict) else formats
    for key in sorted({key for step in formats for key in step}):
        fields[f"format: {key}"] = format_value(run, key)
    for key in ("compile_time_formats", "speed_of_light"):
        if key in run:
            fields[key] = display_value(run[key])
    settings = "".join(
        f"<dt>{html.escape(key.replace('_', ' '))}</dt><dd>{html.escape(value)}</dd>"
        for key, value in fields.items()
    )
    raw = html.escape(json.dumps(run, indent=2))
    provenance = html.escape(run.get("test") or record["test"])
    historical = (
        "<p>Runtime values were not recorded for this historical stream.</p>"
        if run.get("schema_version") != 1
        else ""
    )
    return (
        f'<tr>{rendered}<td><details class="case-details"><summary>Details</summary>'
        f'{historical}<dl class="settings">{settings}</dl>'
        f'<details class="technical"><summary>Full pytest ID and raw metadata</summary>'
        f'<p class="case-id">{provenance}</p><p>Build: <code>{html.escape(record["variant"])}</code>'
        f'<br>Capture: <code>{html.escape(record["run_id"])}</code></p>'
        f'<pre class="input-values">{raw}</pre></details></details></td></tr>'
    )


def run_inputs(records: list[dict]) -> str:
    runs = {}
    for record in records:
        if record.get("run_id"):
            runs[(record["arch"], record["variant"], record["run_id"])] = record
    if not runs:
        return '<p class="muted">No per-run input metadata is available for these records.</p>'
    groups = defaultdict(list)
    for record in runs.values():
        node = record["run"].get("test") or record["test"]
        groups[(record["arch"], node.split("[", 1)[0])].append(record)
    tables = []
    for (arch, test), cases in sorted(groups.items()):
        cases.sort(
            key=lambda record: (
                record["run"].get("test", ""),
                record["variant"],
                record["run_id"],
            )
        )
        rows = "\n".join(
            configuration_row(record, index) for index, record in enumerate(cases, 1)
        )
        tables.append(
            f'<h4>{html.escape(test_name(test))} <span class="counts">{html.escape(arch)} · {len(cases)} captures</span></h4>'
            '<div class="configuration-scroll"><table class="configurations"><thead><tr>'
            "<th>Case</th><th>Input A</th><th>Input B</th><th>Output</th>"
            '<th title="Recorded dimensions, or dimensions from an explicitly named pytest dimension axis">Input size</th>'
            "<th>Tiles</th><th>Dest sync</th><th>Implied math</th><th>Other settings</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return (
        f'<details class="run-inputs"><summary>Input configurations · {len(runs)} captures</summary>'
        '<p class="muted">Test-run inputs, not values sampled at each LLK call. '
        "“—” means not recorded. Expand Details for additional parameters.</p>"
        + "\n".join(tables)
        + "</details>"
    )
