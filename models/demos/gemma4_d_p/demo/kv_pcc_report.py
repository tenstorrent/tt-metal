# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only KV accuracy reports and explicit, configuration-specific baselines."""

import hashlib
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


def write_json(path, report, *, exclusive=False):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w") as output:
        output.write(payload)


class KvPccRun:
    """Load the baseline before device work; always save measured regressions before failing."""

    def __init__(self, configuration, nodeid, baseline_path=None):
        self.configuration = configuration
        self.baseline_path = (
            pathlib.Path(baseline_path) if baseline_path else BASELINE_DIR / f"{digest(configuration)}.json"
        )
        self.report_path = REPORT_DIR / f"{digest(nodeid)[:16]}.json"
        if self.report_path.resolve() == self.baseline_path.resolve():
            raise ValueError("report and baseline paths must be different")
        if not self.baseline_path.is_file():
            raise ValueError(f"Missing KV PCC baseline: {self.baseline_path}. Provide an approved baseline JSON.")
        self.baseline = json.loads(self.baseline_path.read_text())
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
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True, capture_output=True
        )
        report["provenance"] = {
            "revision": revision.stdout.strip() or None,
            "dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
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
            write_json(self.report_path, report)
            raise
        write_json(self.report_path, report)
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
