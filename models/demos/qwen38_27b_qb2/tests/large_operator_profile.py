# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact boundary/counter coverage for a bounded projection and SDPA capture."""

import collections
import csv
import json
import math
import re
import statistics

LABELS = ("OUTPUT_LOCAL", "OUTPUT_COMPLETE", "DOWN_LOCAL", "DOWN_COMPLETE", "SDPA_COMPLETE")


def selected_labels(selection):
    if selection == "all":
        return list(LABELS)
    if selection in LABELS:
        return [selection]
    raise ValueError("Unsupported operator boundary selection")


def capture_plan(passes, split_boundaries):
    boundaries = LABELS if split_boundaries else ("all",)
    return (
        [("before", [], "all")]
        + [
            (f"pass-{i:02d}-{boundary.lower()}", group, boundary)
            for i, group in enumerate(passes)
            for boundary in boundaries
        ]
        + [("after", [], "all")]
    )


def validate_receipt(receipt):
    if (
        receipt.get("state") != "completed"
        or receipt.get("passed") is not True
        or receipt.get("cleanup_completed") is not True
        or receipt.get("device_ids") != [0, 4, 12, 8]
        or receipt.get("batch") != 16
        or receipt.get("context") != 32768
        or not receipt.get("source_sha256")
        or not receipt.get("precision")
    ):
        raise ValueError("Incomplete physical operator receipt")
    labels = selected_labels(receipt.get("boundary_selection", "all"))
    if [case.get("label") for case in receipt.get("cases", [])] != labels:
        raise ValueError("Missing or repeated operator boundary")
    for case in receipt["cases"]:
        samples = case.get("samples", [])
        if case.get("passed") is not True or [s.get("repeat") for s in samples] != [0, 1, 2]:
            raise ValueError("Incomplete operator samples")
        for sample in samples:
            checks = sample.get("accuracy", [])
            hashes = sample.get("hashes", [])
            if (
                len(checks) != 4
                or any(c.get("passed") is not True for c in checks)
                or len(hashes) != 4
                or any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) for h in hashes)
                or hashes != samples[0]["hashes"]
            ):
                raise ValueError("Operator accuracy or repeatability failed")


def compare_outputs(before, after):
    for receipt in (before, after):
        validate_receipt(receipt)
    for name in ("source_sha256", "precision", "device_ids"):
        if before.get(name) != after.get(name):
            raise ValueError("Counter pass changed source, geometry or precision")
    reference = {c["label"]: c["samples"][0]["hashes"] for c in before["cases"]}
    for case in after["cases"]:
        if reference.get(case["label"]) != case["samples"][0]["hashes"]:
            raise ValueError("Counter pass changed output")
    if any(c["label"] == "SDPA_COMPLETE" for c in after["cases"]) and before.get("attention_geometry") != after.get(
        "attention_geometry"
    ):
        raise ValueError("Counter pass changed attention geometry")


def boundary_calls(rows, receipt):
    validate_receipt(receipt)
    active, seen, calls = None, set(), {}
    targets = collections.Counter()
    inventory = []
    for row in rows:
        if row["OP TYPE"] == "signpost":
            match = re.fullmatch(
                r"LARGE_OP_(OUTPUT_LOCAL|OUTPUT_COMPLETE|DOWN_LOCAL|DOWN_COMPLETE|SDPA_COMPLETE)_([012])_(BEGIN|END)",
                row["OP CODE"],
            )
            if not match:
                raise ValueError("Unexpected operator signpost")
            key = (match[1], int(match[2]))
            if match[3] == "BEGIN":
                if active is not None or key in seen:
                    raise ValueError("Repeated or nested operator boundary")
                active = key
                seen.add(key)
            else:
                if active != key or any(targets[key, d] != 1 for d in receipt["device_ids"]):
                    raise ValueError("Operator boundary lacks exactly one target on every rank")
                active = None
        elif active is not None and row.get("DEVICE ID", "") != "":
            device, run = int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"])
            if device not in receipt["device_ids"]:
                raise ValueError("Unexpected operator rank")
            duration = float(row["DEVICE KERNEL DURATION [ns]"]) / 1000
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Missing operator device timing")
            entry = dict(
                label=active[0],
                repeat=active[1],
                device=device,
                run=run,
                op=row["OP CODE"],
                duration_us=duration,
                cores=int(row["CORE COUNT"]),
                attributes=row.get("ATTRIBUTES", ""),
                kernels={k: v for k, v in row.items() if "KERNEL" in k and "PATH" in k},
            )
            inventory.append(entry)
            expected_op = "SdpaDecodeDeviceOperation" if active[0] == "SDPA_COMPLETE" else "MatmulDeviceOperation"
            if row["OP CODE"] == expected_op:
                if (device, run) in calls:
                    raise ValueError("Duplicate target operator")
                calls[device, run] = entry
                targets[active, device] += 1
    labels = selected_labels(receipt.get("boundary_selection", "all"))
    if (
        active is not None
        or seen != {(label, i) for label in labels for i in range(3)}
        or len(calls) != 12 * len(labels)
    ):
        raise ValueError("Incomplete operator boundary inventory")
    return calls, inventory


def collect_counters(rows, calls, expected, type_names):
    # Matmuls also have reader-only cores. Require counters on the independent
    # TRISC1 kernel inventory, rather than treating CORE COUNT as compute cores.
    starts, ends, counters = {}, {}, collections.defaultdict(dict)
    for row in rows:
        key = (int(row["PCIe slot"]), int(row["run host ID"]))
        if key not in calls:
            continue
        core = (*key, int(row["core_x"]), int(row["core_y"]))
        if row["RISC processor type"] == "TRISC_1" and row["zone name"] == "TRISC-KERNEL":
            destination = starts if row["type"] == "ZONE_START" else ends if row["type"] == "ZONE_END" else None
            if destination is not None:
                if core in destination:
                    raise ValueError("Repeated compute kernel marker")
                destination[core] = int(row["time[cycles since reset]"])
        if row["timer_id"].strip() != "9090":
            continue
        try:
            payload = json.loads(row["meta data"].replace(";", ",").replace("'", '"'))
            name = payload["counter type"]
            if not isinstance(name, str):
                name = type_names[name]
            value, reference = payload["value"], payload["ref cnt"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Invalid counter payload") from error
        if name not in expected or name in counters[core]:
            raise ValueError("Duplicate or unexpected counter")
        if type(value) is not int or type(reference) is not int or value < 0 or reference <= 0:
            raise ValueError("Invalid counter interval")
        counters[core][name] = (value, reference)
    if set(starts) != set(ends) or any(ends[k] <= v for k, v in starts.items()):
        raise ValueError("Unpaired compute kernel markers")
    if {k[:2] for k in starts} != set(calls) or any(
        sum(k[:2] == key for k in starts) > c["cores"] for key, c in calls.items()
    ):
        raise ValueError("Missing compute kernel coverage")
    if expected and (set(counters) != set(starts) or any(set(v) != expected for v in counters.values())):
        raise ValueError("Incomplete compute-core counter coverage")
    if not expected and counters:
        raise ValueError("Unexpected counters in zero-mask control")
    grouped = collections.defaultdict(list)
    for core, values in counters.items():
        call = calls[core[:2]]
        for name, pair in values.items():
            grouped[call["label"], core[0], name].append(pair)
    return dict(
        compute_core_calls=len(starts),
        counter_coverage_passed=True,
        summaries=[
            dict(
                label=label,
                device=device,
                counter=name,
                samples=len(values),
                value_sum=sum(v for v, _ in values),
                reference_sum=sum(r for _, r in values),
                ratio_median=statistics.median(v / r for v, r in values),
                ratio_min=min(v / r for v, r in values),
                ratio_max=max(v / r for v, r in values),
            )
            for (label, device, name), values in sorted(grouped.items())
        ],
    )


def analyze_pass(root, expected, type_names):
    receipt = json.loads((root / "operators.json").read_text())
    reports = list((root / "tracy/reports").glob("*/ops_perf_results*.csv"))
    raw = root / "tracy/.logs/profile_log_device.csv"
    if len(reports) != 1 or not raw.is_file():
        raise ValueError("Require one op report and canonical device log")
    with reports[0].open() as stream:
        calls, inventory = boundary_calls(csv.DictReader(stream), receipt)
    with raw.open() as stream:
        architecture = next(stream).strip()
        result = collect_counters(csv.DictReader(stream, skipinitialspace=True), calls, expected, type_names)
    timings = collections.defaultdict(list)
    for call in calls.values():
        timings[call["label"], call["device"]].append(call["duration_us"])
    result.update(
        architecture=architecture,
        target_calls=len(calls),
        inventory=inventory,
        kernels=[dict(label=l, device=d, median_us=statistics.median(v)) for (l, d), v in sorted(timings.items())],
    )
    return result
