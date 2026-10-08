# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attribute warm eager device operations without summing parallel TP ranks.

Only P0 signpost windows are selected; model loading, prefill, trace capture and
warmup stay outside the report. Nested stages have both inclusive and exclusive
totals. Device firmware time is a sum of op durations, not end-to-end TPOT.
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

PROFILE_CASES = [(8192, 1), (8192, 16), (131072, 1), (131072, 8), (262016, 1), (262016, 4)]
SIGNPOST = re.compile(r"^(P0_S(\d+)_B(\d+)_(?:MODEL|L\d+_.+))_(BEGIN|END)$")
DURATIONS = {
    "firmware_ns": "DEVICE FW DURATION [ns]",
    "kernel_ns": "DEVICE KERNEL DURATION [ns]",
    "reader_ns": "DEVICE BRISC KERNEL DURATION [ns]",
    "writer_ns": "DEVICE NCRISC KERNEL DURATION [ns]",
    "compute_ns": "DEVICE TRISC1 KERNEL DURATION [ns]",
}


def require_storage_headroom(paths, *, minimum_free_bytes=16 * 1024**3):
    """Abort diagnostics before exhausting either the report or JIT filesystem."""
    for path in paths:
        free = shutil.disk_usage(path).free
        if free < minimum_free_bytes:
            raise RuntimeError(
                f"Profiler storage guard: {path} has {free} free bytes; " f"requires at least {minimum_free_bytes}"
            )


def drain_after_call(method, drain, *, storage_guard=lambda: None):
    """Drain diagnostic records after each completed prefill chunk, not a batch."""

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        storage_guard()
        result = method(*args, **kwargs)
        drain()
        storage_guard()
        return result

    return wrapped


def metric(row, name):
    value = row.get(name)
    if value in (None, "", "-", "N/A"):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {name}: {value}")
    return number


def accumulate(bucket, row):
    bucket["device_op_rows"] += 1
    for key, name in DURATIONS.items():
        value = metric(row, name)
        if value is None:
            bucket[f"missing_{key}_rows"] += 1
        else:
            bucket[key] += value


def analyze(rows, receipt, *, expected_cases=PROFILE_CASES):
    if receipt.get("passed") is not True or receipt.get("state") != "completed":
        raise ValueError("A completed passing diagnostic receipt is required")
    expected = {(cell["input_tokens"], cell["batch"]) for cell in receipt["cells"]}
    if not expected_cases or expected != set(expected_cases) or len(receipt["cells"]) != len(expected):
        raise ValueError("Diagnostic receipt is missing required long-context cases")
    devices = set(receipt["device_ids"])
    if len(devices) != 4:
        raise ValueError("Diagnostic must contain exactly four TP devices")
    stack, roots, seen = [], set(), set()
    totals = defaultdict(lambda: defaultdict(float))
    inclusive = defaultdict(lambda: defaultdict(float))
    exclusive = defaultdict(lambda: defaultdict(float))
    operations = defaultdict(lambda: defaultdict(float))
    for row in rows:
        op = row.get("OP CODE", "")
        if row.get("OP TYPE") == "signpost":
            marker = SIGNPOST.fullmatch(op)
            if marker is None:
                if op.startswith("P0_"):
                    raise ValueError(f"Unrecognized P0 signpost: {op}")
                continue
            label, length, batch, boundary = marker.groups()
            case = (int(length), int(batch))
            if case not in expected:
                raise ValueError(f"Unexpected diagnostic case {case}")
            if boundary == "BEGIN":
                if not stack:
                    if not label.endswith("_MODEL") or case in roots:
                        raise ValueError(f"Missing or repeated model window: {label}")
                    roots.add(case)
                elif case != stack[0][0] or label.endswith("_MODEL"):
                    raise ValueError(f"Overlapping model windows: {label}")
                stack.append((case, label))
            else:
                if not stack or stack[-1] != (case, label):
                    raise ValueError(f"Unbalanced signposts: {label}")
                stack.pop()
            continue
        if not stack:
            continue
        # Host-only ops have no device id. A device op with missing time is
        # counted explicitly; it must never disappear as a zero-cost op.
        raw_device = row.get("DEVICE ID")
        if raw_device in (None, "", "-"):
            continue
        device = int(raw_device)
        if device not in devices:
            raise ValueError(f"Unqualified device in diagnostic: {device}")
        case = stack[0][0]
        key = (*case, device)
        identity = (*key, row.get("GLOBAL CALL COUNT"), row.get("METAL TRACE REPLAY SESSION ID", ""))
        if identity in seen:
            raise ValueError(f"Duplicate device operation: {identity}")
        seen.add(identity)
        if row.get("METAL TRACE ID") not in (None, "", "-"):
            raise ValueError("Trace replay found inside eager diagnostic window")
        accumulate(totals[key], row)
        for _, stage in stack:
            accumulate(inclusive[(*key, stage)], row)
        accumulate(exclusive[(*key, stack[-1][1])], row)
        accumulate(operations[(*key, stack[-1][1], op)], row)
    if stack or roots != expected:
        raise ValueError("Incomplete diagnostic signpost windows")
    if set(totals) != {(*case, device) for case in expected for device in devices}:
        raise ValueError("Missing device measurements for a diagnostic case")

    def flatten(table, fields):
        return [dict(zip(fields, key), **dict(values)) for key, values in sorted(table.items())]

    fields = ["input_tokens", "batch", "device"]
    device_totals = flatten(totals, fields)
    complete = all(
        not row.get("missing_firmware_ns_rows") and not row.get("missing_kernel_ns_rows") for row in device_totals
    )
    return dict(
        state="completed",
        measurements_complete=complete,
        p0_gate_passed=False,
        scope="Warm eager two-layer device-op attribution; not full-model traced TPOT or accuracy qualification",
        accounting=(
            "Durations remain separate per device. Inclusive stages overlap; exclusive stages partition each "
            "device total. Firmware sums may include overlapping execution. Per-RISC durations include waits "
            "and alone cannot establish a bandwidth or compute bottleneck. Device-op rows are not program counts."
        ),
        remaining_p0=["Reconcile full-model traced op time with TPOT within 5%", "Measure TP8 collective costs"],
        device_totals=device_totals,
        inclusive_stages=flatten(inclusive, fields + ["stage"]),
        exclusive_stages=flatten(exclusive, fields + ["stage"]),
        operations=flatten(operations, fields + ["stage", "op"]),
    )


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "profile-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for table in ("device_totals", "inclusive_stages", "exclusive_stages", "operations"):
        rows = report[table]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{table}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Qwen long-context layer profile",
        "",
        report["scope"] + ".",
        "",
        report["accounting"],
        "",
        f"All device timings present: {report['measurements_complete']}. Full P0 gate: still pending.",
        "",
        "| Input tokens | Batch | Device | Device-op rows | Sum firmware (ms) | Sum kernel (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["device_totals"]:
        lines.append(
            f"| {row['input_tokens']} | {row['batch']} | {row['device']} | {int(row['device_op_rows'])} "
            f"| {row.get('firmware_ns', 0)/1e6:.3f} | {row.get('kernel_ns', 0)/1e6:.3f} |"
        )
    lines.extend(["", "## Most expensive device ops per geometry", ""])
    for length, batch in sorted({(r["input_tokens"], r["batch"]) for r in report["device_totals"]}):
        rows = [r for r in report["device_totals"] if r["input_tokens"] == length and r["batch"] == batch]
        device = max(rows, key=lambda r: r.get("firmware_ns", 0))["device"]
        lines.extend(
            [
                f"### Input {length}, batch {batch}, device {device}",
                "",
                "Device with the largest firmware-duration sum; other ranks remain in the CSV/JSON.",
                "",
                "| Exclusive stage | Op | Calls | Sum firmware (ms) | Sum kernel (ms) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        rows = [
            r for r in report["operations"] if (r["input_tokens"], r["batch"], r["device"]) == (length, batch, device)
        ]
        for row in sorted(rows, key=lambda r: r.get("firmware_ns", 0), reverse=True)[:20]:
            op = row["op"].replace("|", "\\|")
            lines.append(
                f"| {row['stage']} | {op} | {int(row['device_op_rows'])} | {row.get('firmware_ns', 0)/1e6:.3f} | {row.get('kernel_ns', 0)/1e6:.3f} |"
            )
        lines.append("")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"OP CODE", "OP TYPE", "DEVICE ID", "GLOBAL CALL COUNT", *DURATIONS.values()}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing profiler columns: {required - set(reader.fieldnames or [])}")
        report = analyze(reader, json.loads(args.receipt.read_text()))
    report["sources"] = {
        path.name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (args.csv, args.receipt)
    }
    write_report(report, args.output)
    print(json.dumps(dict(output=str(args.output), measurements_complete=report["measurements_complete"])))


if __name__ == "__main__":
    main()
