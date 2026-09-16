# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only KV accuracy reports and explicit, configuration-specific baselines."""

import ast
import hashlib
import html
import json
import math
import pathlib
import subprocess

BASELINE_DIR = pathlib.Path(__file__).resolve().parents[1] / "tests" / "kv_pcc_baselines"
REPORT_DIR = pathlib.Path(__file__).resolve().parents[1] / "tests" / "kv_pcc_reports"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_reference_blocks(index, n_layers, context_len):
    """Select a contiguous reference prefix, retaining source block boundaries."""
    streams = []
    for layer in range(n_layers):
        blocks = sorted(
            index["tensor_streams"][f"kv_post_transform_layer_{layer}"]["chunks"], key=lambda b: b["row_start"]
        )
        selected, next_row = [], 0
        for block in blocks:
            if next_row == context_len:
                break
            if block["row_start"] != next_row or block["row_end"] <= next_row:
                raise ValueError(f"layer {layer}: reference blocks must be contiguous from zero")
            selected.append(block)
            next_row = min(block["row_end"], context_len)
        if next_row != context_len:
            raise ValueError(f"layer {layer}: reference does not cover {context_len} tokens")
        streams.append(selected)
    return streams


def make_report(configuration, records):
    """Emit chunk-by-layer tables first, then the user's per-layer summary columns."""
    tables = {part: {metric: {} for metric in ("pcc", "rel_l2")} for part in ("K", "V")}
    grouped, seen = {}, set()
    for record in records:
        layer, part, chunk = record["layer"], record["part"], record["chunk"]
        key = (layer, part, chunk)
        if key in seen:
            raise ValueError(f"duplicate measurement: {key}")
        seen.add(key)
        grouped.setdefault((layer, part), []).append(record)
        for metric in ("pcc", "rel_l2"):
            value = record[metric]
            row = tables[part][metric].setdefault(
                chunk, {"chunk": chunk, "row_start": record["row_start"], "row_end": record["row_end"]}
            )
            if (row["row_start"], row["row_end"]) != (record["row_start"], record["row_end"]):
                raise ValueError("reference block ranges must match across layers")
            row[f"L{layer}"] = value if math.isfinite(value) else None
    summary = []
    for layer, layer_type in enumerate(configuration["layer_types"]):
        row = {"layer": layer, "type": layer_type}
        for part in ("K", "V"):
            entries = grouped.get((layer, part), [])
            if not entries:
                raise ValueError(f"missing measurements: layer {layer} {part}")
            row["n"] = len(entries)
            for metric, label, operations in (
                ("pcc", "pcc", ("mean", "min", "max")),
                ("rel_l2", "relL2", ("mean", "max")),
            ):
                values = [entry[metric] for entry in entries]
                finite = all(math.isfinite(value) for value in values)
                for operation in operations:
                    result = {
                        "mean": lambda: sum(values) / len(values),
                        "min": lambda: min(values),
                        "max": lambda: max(values),
                    }
                    row[f"{part} {label} {operation}"] = result[operation]() if finite else None
        summary.append(row)
    for part in tables.values():
        for metric, rows in part.items():
            part[metric] = [rows[chunk] for chunk in sorted(rows)]
    return {"schema_version": 1, "configuration": configuration, "per_chunk": tables, "per_layer": summary}


def measurements(report):
    """Validate table coverage and return scores keyed by layer/component/token range."""
    if report.get("schema_version") != 1:
        raise ValueError("unsupported KV PCC report schema")
    layers = {f"L{i}" for i in range(len(report["configuration"]["layer_types"]))}
    result, expected_ranges = {}, None
    for part in ("K", "V"):
        for metric in ("pcc", "rel_l2"):
            rows = report["per_chunk"][part][metric]
            next_row, ranges = 0, []
            for chunk, row in enumerate(rows):
                start, end = row["row_start"], row["row_end"]
                if row["chunk"] != chunk or start != next_row or end <= start:
                    raise ValueError("invalid or missing reference block in report")
                if set(row) - {"chunk", "row_start", "row_end"} != layers:
                    raise ValueError("missing or unexpected layers in report")
                ranges.append((start, end))
                for layer in sorted(layers, key=lambda value: int(value[1:])):
                    value = row[layer]
                    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError("invalid metric in report")
                    if value is not None and (not -1 <= value <= 1 if metric == "pcc" else value < 0):
                        raise ValueError("metric outside its valid range")
                    result[(int(layer[1:]), part, start, end, metric)] = value
                next_row = end
            if next_row != report["configuration"]["context_len"]:
                raise ValueError("report does not cover the complete context")
            if expected_ranges is not None and ranges != expected_ranges:
                raise ValueError("reference block ranges differ across report tables")
            expected_ranges = ranges
    return result


def compare_report(report, baseline, tolerances):
    current = measurements(report)
    if baseline is not None:
        if baseline["configuration"] != report["configuration"]:
            raise ValueError("KV PCC baseline configuration/reference differs from this run")
        expected = measurements(baseline)
        if current.keys() != expected.keys():
            raise ValueError("KV PCC baseline measurement coverage differs from this run")
        if any(value is None for value in expected.values()):
            raise ValueError("KV PCC baseline contains non-finite metrics")
    failures = []
    for (layer, part, start, end, metric), value in current.items():
        old = expected[(layer, part, start, end, metric)] if baseline is not None else None
        degradation = None if old is None or value is None else (old - value if metric == "pcc" else value - old)
        tolerance = tolerances[metric]
        if value is None or (degradation is not None and degradation > tolerance):
            failures.append(
                {
                    "layer": layer,
                    "part": part,
                    "row_start": start,
                    "row_end": end,
                    "metric": metric,
                    "baseline": old,
                    "current": value,
                    "degradation": degradation,
                    "tolerance": tolerance,
                }
            )
    return sorted(
        failures, key=lambda row: float("inf") if row["degradation"] is None else row["degradation"], reverse=True
    )


def read_markdown(path):
    """Load exact regression data from the Markdown baseline's expanded tables."""
    text = pathlib.Path(path).read_text(encoding="utf-8")

    def rows(name, headers):
        begin, end = f"<!-- kv-pcc-{name} -->", f"<!-- /kv-pcc-{name} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f"Missing or duplicate Markdown {name} table")
        body = text.split(begin, 1)[1].split(end, 1)[0]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not all(line.startswith("|") and line.endswith("|") for line in lines):
            raise ValueError(f"Invalid Markdown {name} table")
        cells = [[html.unescape(cell.strip()) for cell in line[1:-1].split("|")] for line in lines]
        if len(cells) < 2 or cells[0] != headers or cells[1] != ["---"] * len(headers):
            raise ValueError(f"Invalid Markdown {name} headers")
        if any(len(row) != len(headers) for row in cells[2:]):
            raise ValueError(f"Invalid Markdown {name} row")
        return cells[2:]

    metadata = {}
    for key, value in rows("metadata", ["field", "value"]):
        if key in metadata:
            raise ValueError(f"Duplicate Markdown metadata: {key}")
        try:
            metadata[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid Markdown metadata: {key}") from error
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported Markdown KV report schema")
    records = []
    headers = ["layer", "part", "chunk", "row_start", "row_end", "pcc", "rel_l2"]
    for row in rows("measurements", headers):
        entry = dict(zip(headers, row))
        for key in ("layer", "chunk", "row_start", "row_end"):
            entry[key] = int(entry[key])
        if entry["part"] not in ("K", "V"):
            raise ValueError("Invalid Markdown KV component")
        for key in ("pcc", "rel_l2"):
            entry[key] = float("nan") if entry[key] == "N/A" else float(entry[key])
        records.append(entry)
    report = make_report(metadata["configuration"], records)
    report["tolerances"] = metadata["tolerances"]
    report["provenance"] = metadata.get("provenance", {})
    report["comparison"] = metadata.get("comparison", {})
    measurements(report)
    return report


def write_markdown(path, report, *, exclusive=False):
    """Write the three KV tables, with comparison status and failure details."""
    lines = []

    def cell(value):
        return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", " ")

    def number(value, precision):
        return "N/A" if value is None or not math.isfinite(value) else f"{value:.{precision}f}"

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
        lines.append("")

    comparison = report.get("comparison", {})
    status = comparison.get("status", "uncompared")
    lines.extend([f"**Result: {cell(status.upper())}**", ""])
    if comparison.get("error"):
        lines.extend([f"**Error:** {cell(comparison['error'])}", ""])
    if comparison.get("baseline"):
        lines.extend([f"Baseline: `{cell(comparison['baseline'])}`", ""])
    provenance = report.get("provenance", {})
    if provenance.get("recovered_from_multiple_runs"):
        lines.extend(["Recovered from two runs; earlier measurements were rounded to six decimal places.", ""])
    if provenance.get("revision"):
        lines.extend([f"Revision: `{cell(provenance['revision'])}`", ""])
    if report.get("tolerances"):
        lines.extend(
            [
                f"Allowed degradation: PCC {report['tolerances']['pcc']}; relative L2 {report['tolerances']['rel_l2']}.",
                "",
            ]
        )

    layers = [f"L{i}" for i in range(len(report["configuration"]["layer_types"]))]
    for part, label in (("K", "K / K_effective"), ("V", "V")):
        lines.extend([f"## Per-chunk detail, KV {label} (pcc)", ""])
        table(
            ["chunk"] + layers,
            [
                [row["chunk"]] + [number(row.get(layer), 5) for layer in layers]
                for row in report["per_chunk"][part]["pcc"]
            ],
        )
    lines.extend(
        [
            "## kv_post_transform_layer_{i} — K / K_effective and V scored separately",
            "",
            "Aggregated over chunks. `pcc min`/`relL2 max` expose per-chunk outliers a mean would bury.",
            "",
        ]
    )
    metrics = ("pcc mean", "pcc min", "pcc max", "relL2 mean", "relL2 max")
    table(
        ["layer", "type", "n"] + [f"{part} {metric}" for part in ("K / K_effective", "V") for metric in metrics],
        [
            [row["layer"], row["type"], row["n"]]
            + [
                number(row[f"{part} {metric}"], 6 if metric.startswith("pcc") else 4)
                for part in ("K", "V")
                for metric in metrics
            ]
            for row in report["per_layer"]
        ],
    )
    failures = comparison.get("failures", [])
    if failures:
        lines.extend([f"<details><summary>{len(failures)} accuracy regressions</summary>", ""])
        headers = ["layer", "part", "row_start", "row_end", "metric", "baseline", "current", "degradation", "tolerance"]
        table(headers, [[failure.get(key) for key in headers] for failure in failures])
        lines.extend(["</details>", ""])
    # Human-facing tables are rounded; this Markdown table preserves exact scores.
    lines.extend(["<details><summary>Full-precision regression data</summary>", "", "<!-- kv-pcc-metadata -->"])
    metadata = {key: report[key] for key in ("schema_version", "configuration", "tolerances")}
    metadata["provenance"] = provenance
    metadata["comparison"] = {key: value for key, value in comparison.items() if key != "failures"}
    table(["field", "value"], [[key, repr(value)] for key, value in metadata.items()])
    lines.extend(["<!-- /kv-pcc-metadata -->", "", "<!-- kv-pcc-measurements -->"])
    exact_rows = []
    for part in ("K", "V"):
        l2_rows = {row["chunk"]: row for row in report["per_chunk"][part]["rel_l2"]}
        for row in report["per_chunk"][part]["pcc"]:
            for layer in layers:
                values = (row.get(layer), l2_rows.get(row["chunk"], {}).get(layer))
                exact_rows.append(
                    [int(layer[1:]), part, row["chunk"], row["row_start"], row["row_end"]]
                    + ["N/A" if value is None else repr(value) for value in values]
                )
    table(["layer", "part", "chunk", "row_start", "row_end", "pcc", "rel_l2"], exact_rows)
    lines.extend(["<!-- /kv-pcc-measurements -->", "", "</details>", ""])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


class KvPccRun:
    """Load the baseline before device work; always save measured regressions before failing."""

    def __init__(self, configuration, nodeid, baseline_path=None):
        self.configuration = configuration
        self.baseline_path = pathlib.Path(baseline_path) if baseline_path else BASELINE_DIR / "baseline.md"
        self.report_path = REPORT_DIR / f"{digest(nodeid)[:16]}.md"
        if self.report_path.resolve() == self.baseline_path.resolve():
            raise ValueError("report and baseline paths must be different")
        if not self.baseline_path.is_file():
            raise ValueError(f"Missing KV PCC baseline: {self.baseline_path}. Provide an approved Markdown baseline.")
        self.baseline = read_markdown(self.baseline_path)
        if self.baseline["configuration"] != configuration:
            raise ValueError("KV PCC baseline configuration/reference differs from this run")
        self.tolerances = self.baseline["tolerances"]
        if set(self.tolerances) != {"pcc", "rel_l2"} or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in self.tolerances.values()
        ):
            raise ValueError("invalid baseline tolerances")
        if any(value is None for value in measurements(self.baseline).values()):
            raise ValueError("baseline contains non-finite metrics")
        self.nodeid = nodeid

    def finish(self, records):
        report = make_report(self.configuration, records)
        repo = pathlib.Path(__file__).resolve().parents[4]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
        report["provenance"] = {
            "revision": revision.stdout.strip() or None,
            "test": self.nodeid,
        }
        report["tolerances"] = self.tolerances
        try:
            failures = compare_report(report, self.baseline, self.tolerances)
            report["comparison"] = {
                "status": "failed" if failures else "passed",
                "baseline": str(self.baseline_path),
                "failures": failures,
            }
        except ValueError as error:
            report["comparison"] = {"status": "invalid", "error": str(error)}
            write_markdown(self.report_path, report)
            raise
        write_markdown(self.report_path, report)
        if failures:
            lines = [
                f"L{f['layer']} {f['part']} rows [{f['row_start']},{f['row_end']}) {f['metric']}: "
                f"{f['baseline']} -> {f['current']} (allowed {f['tolerance']})"
                for f in failures[:20]
            ]
            raise AssertionError(
                f"{len(failures)} KV accuracy regressions; report: {self.report_path}\n" + "\n".join(lines)
            )
        return self.report_path
