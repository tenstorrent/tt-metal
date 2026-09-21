# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only KV accuracy reports compared with the selected dataset baseline."""

import hashlib
import json
import math
import pathlib
import subprocess

REPORT_DIR = pathlib.Path(__file__).resolve().parents[1] / "kv_pcc_reports"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_report(configuration, records):
    """Emit per-chunk K/V PCC and relative-L2 tables."""
    tables = {part: {metric: {} for metric in ("pcc", "rel_l2")} for part in ("K", "V")}
    seen = set()
    for record in records:
        layer, part, chunk = record["layer"], record["part"], record["chunk"]
        key = (layer, part, chunk)
        if key in seen:
            raise ValueError(f"duplicate measurement: {key}")
        seen.add(key)
        for metric in ("pcc", "rel_l2"):
            value = record[metric]
            row = tables[part][metric].setdefault(
                chunk, {"chunk": chunk, "row_start": record["row_start"], "row_end": record["row_end"]}
            )
            if (row["row_start"], row["row_end"]) != (record["row_start"], record["row_end"]):
                raise ValueError("reference block ranges must match across layers")
            row[f"L{layer}"] = value if math.isfinite(value) else None
    for part in tables.values():
        for metric, rows in part.items():
            part[metric] = [rows[chunk] for chunk in sorted(rows)]
    return {"schema_version": 1, "configuration": configuration, "per_chunk": tables}


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
    if baseline is None:
        raise ValueError("KV PCC comparison requires an approved baseline")
    if baseline["configuration"] != report["configuration"]:
        raise ValueError("KV PCC baseline configuration/reference differs from this run")
    expected = measurements(baseline)
    if current.keys() != expected.keys():
        raise ValueError("KV PCC baseline measurement coverage differs from this run")
    if any(value is None for value in expected.values()):
        raise ValueError("KV PCC baseline contains non-finite metrics")
    failures = []
    for (layer, part, start, end, metric), value in current.items():
        old = expected[(layer, part, start, end, metric)]
        degradation = None if value is None else (old - value if metric == "pcc" else value - old)
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


def write_json(path, report):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.write_text(payload)


class KvPccRun:
    """Validate the dataset baseline and save comparison results before reporting regressions."""

    def __init__(self, configuration, nodeid, baseline_path):
        self.configuration = configuration
        self.baseline_path = pathlib.Path(baseline_path)
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
        repo = pathlib.Path(__file__).resolve().parents[5]
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
