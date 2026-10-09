# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile complete model/sampler trace replays with their fenced host time.

Trace rows keep capture-time HOST START TS although CSV ordering moves them to
replay time. Attribute them using capture signpost timestamps, not CSV position.
Clock conversion is inferred from the export's own firmware cycles/duration;
cycles from different chips are never compared directly.
"""

import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CASES = ((32768, 32), (8192, 1), (8192, 16))
SCOPE = (
    "Full 64-layer model plus split device sampler, real weights and synthetic populated caches; "
    "three restored-state trace replays, not natural-prompt accuracy or unprofiled serving throughput"
)


def number(row, key, *, integer=False):
    value = row.get(key)
    if value in (None, "", "-", "N/A"):
        raise ValueError(f"Missing {key}")
    result = int(value) if integer else float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {key}")
    return result


def intervals_by_stage(intervals):
    """Disjoint elapsed-time accounting, retaining overlaps and uncovered gaps."""
    events = defaultdict(list)
    for start, end, stage in intervals:
        if end <= start:
            raise ValueError("Invalid firmware interval")
        events[start].append((stage, 1))
        events[end].append((stage, -1))
    active, totals = defaultdict(int), defaultdict(int)
    previous = None
    for cycle in sorted(events):
        if previous is not None:
            labels = [stage for stage, count in active.items() if count]
            label = labels[0] if len(labels) == 1 else "overlap" if labels else "uncovered_gap"
            totals[label] += cycle - previous
        for stage, change in events[cycle]:
            active[stage] += change
            if active[stage] < 0:
                raise ValueError("Unbalanced firmware interval")
        previous = cycle
    return dict(totals)


def capture_windows(rows):
    boundaries = defaultdict(dict)
    for row in rows:
        if row.get("OP TYPE") != "signpost" or not row.get("OP CODE", "").startswith("FULLTRACE_"):
            continue
        label, boundary = row["OP CODE"].rsplit("_", 1)
        if boundary not in ("BEGIN", "END") or boundary in boundaries[label]:
            raise ValueError("Duplicate or unknown capture boundary")
        boundaries[label][boundary] = number(row, "HOST START TS", integer=True)
    windows = {}
    for label, pair in boundaries.items():
        if set(pair) != {"BEGIN", "END"} or pair["BEGIN"] >= pair["END"]:
            raise ValueError("Incomplete capture window")
        windows[label.removeprefix("FULLTRACE_")] = (pair["BEGIN"], pair["END"])
    expected = {"MODEL", "SAMPLER", *(f"L{i}" for i in range(64))}
    if set(windows) != expected:
        raise ValueError("Capture must identify all 64 layers, model and sampler")
    layers = [windows[f"L{i}"] for i in range(64)]
    lo, hi = windows["MODEL"]
    if any(not lo <= begin < end <= hi for begin, end in layers):
        raise ValueError("Layer capture lies outside model")
    if any(layers[i][1] > layers[i + 1][0] for i in range(63)) or hi > windows["SAMPLER"][0]:
        raise ValueError("Overlapping or unordered layer captures")
    return windows


def analyze(rows, receipt):
    if (
        receipt.get("state") != "completed"
        or receipt.get("passed") is not True
        or receipt.get("cleanup_completed") is not True
        or receipt.get("scope") != SCOPE
        or receipt.get("layer_indices") != list(range(64))
        or receipt.get("prefill_calls") != 0
        or len(receipt.get("replays", [])) != 3
    ):
        raise ValueError("Require a clean complete full-model trace receipt")
    if len(set(receipt.get("output_hashes", []))) != 1 or len(receipt.get("output_hashes", [])) != 5:
        raise ValueError("Warm and replay logits must match exactly")
    if len(set(receipt.get("token_hashes", []))) != 1 or len(receipt.get("token_hashes", [])) != 5:
        raise ValueError("Warm and replay sampled tokens must match exactly")
    devices = set(receipt["device_ids"])
    if len(devices) != 4:
        raise ValueError("Require four distinct TP ranks")
    model_id, sample_id = receipt["model_trace_id"], receipt["sample_trace_id"]
    if model_id == sample_id:
        raise ValueError("Model and sampling traces must be distinct")
    rows = list(rows)
    windows = capture_windows(rows)
    per_session, seen, clock_ratios = defaultdict(list), set(), defaultdict(list)
    for row in rows:
        trace = row.get("METAL TRACE ID")
        if trace in (None, "", "-") or int(trace) not in (model_id, sample_id):
            continue
        session = row.get("METAL TRACE REPLAY SESSION ID")
        if session in (None, "", "-"):
            continue  # A host capture record without executed device timing.
        trace, session = int(trace), int(session)
        device = number(row, "DEVICE ID", integer=True)
        if device not in devices:
            raise ValueError("Unqualified device in full trace")
        call = number(row, "GLOBAL CALL COUNT", integer=True)
        identity = (device, trace, session, call)
        if identity in seen:
            raise ValueError("Duplicate replay operation")
        seen.add(identity)
        captured = number(row, "HOST START TS", integer=True)
        root = "MODEL" if trace == model_id else "SAMPLER"
        if not windows[root][0] <= captured < windows[root][1]:
            raise ValueError("Replay metadata lies outside its capture window")
        stage = "model_other" if root == "MODEL" else "sampler"
        if root == "MODEL":
            for index in range(64):
                begin, end = windows[f"L{index}"]
                if begin <= captured < end:
                    stage = f"L{index}"
                    break
        start = number(row, "DEVICE FW START CYCLE", integer=True)
        end = number(row, "DEVICE FW END CYCLE", integer=True)
        duration = number(row, "DEVICE FW DURATION [ns]")
        if end <= start or duration <= 0:
            raise ValueError("Invalid firmware clock conversion")
        clock_ratios[device].append((end - start) / duration)
        per_session[device, trace, session].append(
            dict(
                call=call,
                start=start,
                end=end,
                stage=stage,
                op=row.get("OP CODE", ""),
                kernel_ns=number(row, "DEVICE KERNEL DURATION [ns]"),
            )
        )
    sessions = {}
    for device in devices:
        for trace in (model_id, sample_id):
            ids = sorted(session for d, t, session in per_session if (d, t) == (device, trace))
            if len(ids) != 3:
                raise ValueError("Every rank must contain exactly three model and sampler replays")
            sessions[device, trace] = ids
    for trace in (model_id, sample_id):
        if len({tuple(sessions[d, trace]) for d in devices}) != 1:
            raise ValueError("Trace session IDs differ between ranks")
    required = {f"L{i}" for i in range(64)}
    results = []
    for device in sorted(devices):
        ratio = statistics.median(clock_ratios[device])
        # The rounded nanosecond column must imply a consistent clock. Tiny
        # durations have rounding noise; tolerate 1%, never silently mix clocks.
        if any(abs(value / ratio - 1) > 0.01 for value in clock_ratios[device]):
            raise ValueError("Inconsistent profiler clock conversion")
        expected_calls = None
        for index in range(3):
            model = per_session[device, model_id, sessions[device, model_id][index]]
            sample = per_session[device, sample_id, sessions[device, sample_id][index]]
            if not required.issubset(row["stage"] for row in model):
                raise ValueError("A replay is missing one or more decoder layers")
            calls = {(model_id, row["call"]) for row in model} | {(sample_id, row["call"]) for row in sample}
            if expected_calls is not None and calls != expected_calls:
                raise ValueError("Replay operation coverage changed")
            expected_calls = calls
            if min(r["start"] for r in sample) < min(r["start"] for r in model):
                raise ValueError("Sampler executes before model")
            operations = model + sample
            start, end = min(r["start"] for r in operations), max(r["end"] for r in operations)
            accounting = intervals_by_stage([(r["start"], r["end"], r["stage"]) for r in operations])
            assert sum(accounting.values()) == end - start
            op_types = defaultdict(lambda: dict(calls=0, kernel_ns=0.0))
            for row in operations:
                target = op_types[row["op"]]
                target["calls"] += 1
                target["kernel_ns"] += row["kernel_ns"]
            results.append(
                dict(
                    device=device,
                    replay=index,
                    cycles_per_ns=ratio,
                    span_ns=(end - start) / ratio,
                    disjoint_stage_ns={key: value / ratio for key, value in accounting.items()},
                    device_op_rows=len(operations),
                    operations=dict(op_types),
                )
            )
    comparisons = []
    for index, replay in enumerate(receipt["replays"]):
        host_ns = replay["host_step_s"] * 1e9
        if not math.isfinite(host_ns) or host_ns <= 0:
            raise ValueError("Invalid fenced host duration")
        rank = max((r for r in results if r["replay"] == index), key=lambda row: row["span_ns"])
        gap = (host_ns - rank["span_ns"]) / host_ns
        comparisons.append(
            dict(
                replay=index,
                host_step_ns=host_ns,
                longest_rank=rank["device"],
                device_span_ns=rank["span_ns"],
                relative_unaccounted_time=gap,
                within_five_percent=abs(gap) <= 0.05,
            )
        )
    return dict(
        state="completed",
        measurements_complete=True,
        scope=SCOPE,
        ranks=results,
        comparisons=comparisons,
        full_trace_reconciliation_passed=all(row["within_five_percent"] for row in comparisons),
        p0_gate_passed=False,
        accounting="Each rank's firmware intervals form a disjoint timeline with explicit overlap and gaps. "
        "Firmware includes waits; this is not compute-active time. Longest rank span is compared to the same "
        "fenced two-trace host step. No cross-chip clock alignment or summed rank duration is used.",
        remaining=["Unprofiled overhead calibration", "Natural-prompt timing comparison", "TP8 collective costs"],
    )


def collect(root):
    root = Path(root)
    suites = ET.parse(root / "hardware.xml").getroot().findall(".//testsuite")
    if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
        int(s.get(key, 0)) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("Full trace hardware test did not pass")
    files = list((root / "tracy").rglob("ops_perf_results*.csv"))
    if len(files) != 1 or list((root / "tracy").rglob("profile_log_device.csv")):
        raise ValueError("Require one bounded ops export and no raw device dump")
    with files[0].open(newline="") as stream:
        report = analyze(csv.DictReader(stream), json.loads((root / "profile.json").read_text()))
    report["csv_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
