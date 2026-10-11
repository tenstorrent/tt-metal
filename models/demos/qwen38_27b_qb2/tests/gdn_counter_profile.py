# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict counter coverage for the current pipeline; no device imports."""

import collections
import csv
import json
import re
import statistics

from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import validate_coverage


def counter_names(header, groups):
    """Use the pinned architecture's arrays, not a guessed counter inventory."""
    names = set()
    for group in groups:
        match = re.search(r"\b" + re.escape(group) + r"_counters\s*=\s*\{(.*?)\};", header, re.S)
        if not match:
            raise ValueError("Native counter array missing: " + group)
        found = re.findall(r"PerfCounterType::([A-Z0-9_]+)", match[1])
        if not found or len(set(found)) != len(found):
            raise ValueError("Native counter array is empty or ambiguous: " + group)
        names.update(found)
    return names


def pipeline_calls(rows, receipt):
    """Resolve exact run IDs from paired signposts on every physical rank."""
    validate_coverage(receipt, pipeline=True)
    calls, seen = {}, set()
    active, counts = None, None
    for row in rows:
        if row["OP TYPE"] == "signpost":
            match = re.fullmatch(r"GDN_PIPELINE_B(\d+)_(ZERO|SKIP)_ZONES([01])_STEP([012])_(BEGIN|END)", row["OP CODE"])
            if not match:
                raise ValueError("Unexpected pipeline signpost")
            config = (int(match[1]), match[2].lower(), int(match[3]), int(match[4]))
            if match[5] == "BEGIN":
                if active is not None or config in seen:
                    raise ValueError("Duplicate or nested pipeline signpost")
                active, counts = config, collections.Counter()
                seen.add(config)
            else:
                if active != config or set(counts) != set(receipt["device_ids"]) or set(counts.values()) != {2}:
                    raise ValueError("Incomplete two-kernel, four-rank pipeline")
                active = None
        elif active is not None:
            if row["OP CODE"] != "GenericOpDeviceOperation":
                raise ValueError("Unexpected operation inside pipeline")
            device, run = int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"])
            if (device, run) in calls or counts[device] > 1:
                raise ValueError("Duplicate pipeline operation")
            kind = ("recurrence", "epilogue")[counts[device]]
            counts[device] += 1
            duration = float(row["DEVICE KERNEL DURATION [ns]"]) / 1000
            cores = int(row["CORE COUNT"])
            if not (duration > 0 and cores > 0):
                raise ValueError("Missing pipeline timing or core count")
            calls[device, run] = dict(config=active, kind=kind, duration_us=duration, cores=cores)
    expected = {(b, p, z, s) for b in (16, 32) for p in ("zero", "skip") for z in (0, 1) for s in range(3)}
    if active is not None or seen != expected or len(calls) != 192:
        raise ValueError("Incomplete pipeline call inventory")
    return calls


def collect_counters(rows, calls, expected, type_names):
    """Require each selected counter on every active core and target operation."""
    per_core = collections.defaultdict(dict)
    summaries = collections.defaultdict(list)
    for row in rows:
        if row["timer_id"].strip() != "9090":
            continue
        key = (int(row["PCIe slot"]), int(row["run host ID"]))
        if key not in calls:
            continue
        core = (*key, int(row["core_x"]), int(row["core_y"]))
        try:
            payload = json.loads(row["meta data"].replace(";", ",").replace("'", '"'))
            name = payload["counter type"]
            if not isinstance(name, str):
                name = type_names[name]
            value, reference = payload["value"], payload["ref cnt"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Malformed native counter payload") from error
        if name not in expected or name in per_core[core]:
            raise ValueError("Unexpected or duplicate native counter")
        if type(value) is not int or type(reference) is not int or value < 0 or reference <= 0:
            raise ValueError("Counter requires a nonnegative value and positive reference interval")
        per_core[core][name] = (value, reference)
        call = calls[key]
        # Keep ranks separate; phase and processor intervals overlap.
        group = (*call["config"][:3], call["kind"], key[0], name)
        summaries[group].append((value, reference))
    observed = collections.Counter()
    for core, counters in per_core.items():
        if set(counters) != expected:
            raise ValueError("Incomplete counters on an active pipeline core")
        observed[core[:2]] += 1
    if dict(observed) != {key: call["cores"] for key, call in calls.items()}:
        raise ValueError("Counter coverage missing an operation, rank or active core")
    rows = []
    for (batch, padding, profiled, kind, device, name), values in sorted(summaries.items()):
        ratios = [value / ref for value, ref in values]
        rows.append(
            dict(
                batch=batch,
                padding=padding,
                profiled=profiled,
                kind=kind,
                device=device,
                counter=name,
                samples=len(values),
                value_sum=sum(value for value, _ in values),
                reference_sum=sum(ref for _, ref in values),
                ratio_median=statistics.median(ratios),
                ratio_min=min(ratios),
                ratio_max=max(ratios),
            )
        )
    return dict(active_core_calls=len(per_core), all_requested_counters_present=True, summaries=rows)


def compare_outputs(reference, observed):
    for report in (reference, observed):
        validate_coverage(report, pipeline=True)
    if reference["device_ids"] != observed["device_ids"]:
        raise ValueError("Counter pass changed physical ranks")
    for key in ("source_sha256", "instrumented_sha256"):
        if not reference.get(key) or reference[key] != observed.get(key):
            raise ValueError("Counter pass changed kernel source")
    if [case["hashes"] for case in reference["cases"]] != [case["hashes"] for case in observed["cases"]]:
        raise ValueError("Counter instrumentation changed state or output")


def analyze_pass(root, expected, type_names):
    receipt = json.loads((root / "phase.json").read_text())
    paths = list((root / "tracy/reports").glob("*/ops_perf_results*.csv"))
    # Tracy also copies this raw log into the report directory. Read its
    # canonical capture path, not both copies as independent observations.
    raw = root / "tracy/.logs/profile_log_device.csv"
    if len(paths) != 1 or not raw.is_file():
        raise ValueError("Require one native op report and the canonical raw device log per pass")
    with paths[0].open() as stream:
        calls = pipeline_calls(csv.DictReader(stream), receipt)
    result = dict(kernel_calls=len(calls), kernels=[])
    timings = collections.defaultdict(list)
    for (device, _), call in calls.items():
        timings[(*call["config"][:3], call["kind"], device)].append(call["duration_us"])
    for (batch, padding, profiled, kind, device), times in sorted(timings.items()):
        result["kernels"].append(
            dict(
                batch=batch,
                padding=padding,
                profiled=profiled,
                kind=kind,
                device=device,
                median_us=statistics.median(times),
            )
        )
    if expected:
        with raw.open() as stream:
            result["architecture"] = next(stream).strip()
            result.update(collect_counters(csv.DictReader(stream, skipinitialspace=True), calls, expected, type_names))
    return result
