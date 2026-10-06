# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile same-run profiler windows and host timings using saved artifacts only."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return {"path": str(path), "sha256": sha256(path)}


def profile_command(journal_path, runner_path, csv_path):
    matches = []
    for entry in json.loads(journal_path.read_text()):
        command = entry["command"]
        if "--output" not in command:
            continue
        if Path(command[command.index("--output") + 1]).resolve() == runner_path.resolve():
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("Expected one journal command for the supplied runner JSON")
    entry = matches[0]
    command = entry["command"]
    if entry["returncode"] != 0 or not all(flag in command for flag in ("--profile", "--timing", "-o")):
        raise ValueError("A successful profiled/timed command with a Tracy output directory is required")
    raw_dir = Path(command[command.index("-o") + 1])
    digest = sha256(csv_path)
    raw_matches = [path for path in raw_dir.rglob("ops_perf_results*.csv") if sha256(path) == digest]
    if len(raw_matches) != 1:
        raise ValueError("Summary CSV must match exactly one raw CSV from this journal command")
    return {"journal": source(journal_path), "command": command, "raw_csv": source(raw_matches[0])}


def decode_signposts(csv_path, steps):
    endpoints, sessions = {}, set()
    active, native_ops = False, 0
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            code = row["OP CODE"]
            if code in ("PERF_DECODE", "PERF_DECODE_END"):
                if code in endpoints:
                    raise ValueError("Expected one complete decode signpost pair")
                endpoints[code] = int(row["HOST START TS"])
                active = code == "PERF_DECODE"
            elif active and row["OP TYPE"] == "tt_dnn_device":
                sessions.add(row["METAL TRACE REPLAY SESSION ID"])
                native_ops += 1
    if set(endpoints) != {"PERF_DECODE", "PERF_DECODE_END"} or len(sessions) != steps:
        raise ValueError("Incomplete signpost pair or decode replay count")
    if sessions.intersection({"", "-", "nan"}):
        raise ValueError("Device rows lack decode replay identifiers")
    elapsed = endpoints["PERF_DECODE_END"] - endpoints["PERF_DECODE"]
    if elapsed <= 0:
        raise ValueError("Decode host signposts must be ordered")
    return dict(
        start_ns=endpoints["PERF_DECODE"],
        end_ns=endpoints["PERF_DECODE_END"],
        total_ns=elapsed,
        mean_host_us=elapsed / steps / 1000,
        replays=steps,
        native_ops=native_ops,
    )


def reconcile(summary_path, runner_path, journal_path, unprofiled_path=None):
    summary = json.loads(summary_path.read_text())
    runner = json.loads(runner_path.read_text())
    workload = summary["workload"]
    steps = workload["output_tokens"]
    if (
        runner["layer_type"] != summary["layer_type"]
        or runner["length"] != workload["input_tokens"]
        or runner["decode"]["steps"] != steps
        or summary["whole_layer_windows"]["decode"]["samples"] != steps
    ):
        raise ValueError("Summary and runner geometries disagree")
    csv_path = Path(summary["source"])
    command = profile_command(journal_path, runner_path, csv_path)
    loop = decode_signposts(csv_path, steps)
    if loop["native_ops"] != summary["whole_layer_windows"]["decode"]["native_ops"]:
        raise ValueError("Summary and CSV native-operation counts disagree")
    host_samples = runner["traced_decode_host_us"]
    if len(host_samples) != 5 or any(value <= 0 for value in host_samples):
        raise ValueError("Expected five positive fixed-position host timing samples")
    host_fixed = statistics.median(host_samples)
    device = summary["decode_device_us"]
    theoretical = summary["decode_dram_bytes"] / summary["peak_dram_bytes_per_s"] * 1e6
    result = dict(
        layer_type=summary["layer_type"],
        workload=workload,
        sources={"summary": source(summary_path), "runner": source(runner_path), "csv": source(csv_path)},
        same_run_provenance=command,
        decode_estimated_dram_bytes=summary["decode_dram_bytes"],
        theoretical_dram_bytes_per_s=summary["peak_dram_bytes_per_s"],
        decode_theoretical_transfer_us=theoretical,
        decode_whole_device_window_mean_us=device,
        decode_profile_loop_host_mean_us=loop["mean_host_us"],
        decode_fixed_position_host_median_us=host_fixed,
        profile_loop=loop,
        fixed_position_host=dict(
            samples_us=host_samples,
            samples=5,
            replays_per_sample=30,
            absolute_position=runner["decode"]["positions"][-1],
            input_refresh=False,
        ),
        gaps_us=dict(
            device_window_minus_theoretical_transfer=device - theoretical,
            profile_loop_host_minus_device_window=loop["mean_host_us"] - device,
            fixed_position_host_minus_device_window=host_fixed - device,
            profile_loop_host_minus_fixed_position_host=loop["mean_host_us"] - host_fixed,
        ),
        scopes=[
            "Theoretical transfer time is estimated operand bytes divided by the stated peak DRAM bandwidth. It is a traffic-only roofline estimate, not measured memory-controller time or a prediction of complete-layer latency.",
            "Device time is the mean of complete first-firmware-start to last-firmware-end windows for all successive-position replays. All layer operations and intra-layer gaps remain included; inter-replay gaps and input refresh are outside these individual windows.",
            "Profile-loop host time is the Tracy host-clock difference between PERF_DECODE and PERF_DECODE_END divided by replay count. The runner synchronizes before the first signpost and before the final signpost. The span includes all per-step input/position preparation, copies, trace submissions, waits, inter-replay intervals, final synchronization and signpost-call boundary costs; HF preparation and output readback are outside it.",
            "Fixed-position host time is from the same profiled process, after the successive-position loop: median of five batches of 30 trace replays with no input refresh at the final position. It uses perf_counter_ns and is a different execution regime from the profile-loop mean.",
            "Only elapsed durations are compared across host/device clocks; absolute timestamps are not subtracted across clock domains. The CSV host signpost timestamps are nanoseconds (Tracy message total_ns). Device cycle conversion follows the whole-layer summary.",
            "Gaps are arithmetic differences, not isolated measurements of Python, dispatch, refresh, DRAM, contention or synchronization costs. Those components can overlap, and fixed-position versus successive-position behavior differs. No specific bottleneck or contention cause is inferred from these artifacts alone; negative differences are retained without clamping.",
        ],
    )
    if unprofiled_path is not None:
        unprofiled = json.loads(unprofiled_path.read_text())
        result["separate_unprofiled_observation"] = dict(
            source=source(unprofiled_path),
            median_host_us=statistics.median(unprofiled["traced_decode_host_us"]),
            layer_type=unprofiled["layer_type"],
            input_tokens=unprofiled["length"],
            output_tokens=unprofiled["decode"]["steps"],
            scope="Separate process/instrumentation regime; excluded from all same-run gaps above.",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--runner-json", type=Path, required=True)
    parser.add_argument("--profile-command-journal", type=Path, required=True)
    parser.add_argument("--unprofiled-runner-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.summary, args.runner_json, args.profile_command_journal, args.unprofiled_runner_json)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
