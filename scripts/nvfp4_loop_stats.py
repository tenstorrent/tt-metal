#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize candidate throughput and evaluator overhead without touching hardware."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import re
import time
import xml.etree.ElementTree as ET


def summarize(root):
    results, incomplete = [], []
    for path in sorted(root.glob("*/result.json")):
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            incomplete.append(str(path))
            continue
        result["artifact_directory"] = str(path.parent)
        results.append(result)
    summary = dict(
        evaluations=len(results),
        statuses={},
        validation_levels={},
        stages=0,
        missing_junit_stages=0,
        pytest_counts=dict(passed=0, failed=0, skipped=0),
        gate_datum_comparisons=0,
        detailed_timing_stages=0,
        summed_stage_seconds={},
        pytest_phase_seconds={},
        candidates=[],
        incomplete_artifacts=incomplete,
        jit_audits=[],
    )
    snapshots, intervals, counted_junits = set(), [], set()
    worktrees = sorted({result["candidate"] for result in results if result.get("candidate")})
    summary["distinct_worktrees"] = len(worktrees)
    summary["worktrees"] = worktrees
    for result in results:
        directory = Path(result["artifact_directory"])
        snapshot = json.dumps([result.get("kernel_hashes"), result.get("input_bit_permutation")], sort_keys=True)
        fingerprint = hashlib.sha256(snapshot.encode()).hexdigest()
        snapshots.add(fingerprint)
        status, level = result["status"], result.get("validation_level", "full")
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
        summary["validation_levels"][level] = summary["validation_levels"].get(level, 0) + 1
        start, end = result["started_unix"], result.get("finished_unix", time.time())
        intervals.append((start, end))
        candidate = dict(
            name=directory.name,
            status=status,
            validation_level=level,
            source_snapshot=fingerprint,
            wall_seconds=end - start,
        )
        if "performance" in result:
            perf = result["performance"]
            candidate["cycles_per_tile"] = perf["pipeline_cycles_per_output_tile"]
            profile = Path(perf["profile_csv"])
            if profile.exists():
                with profile.open() as stream:
                    rows = [r for r in csv.DictReader(stream) if r["OP CODE"] == "Nvfp4DeviceOperation"]
                host = [float(r["HOST DURATION [ns]"]) / 1e9 for r in rows if r.get("HOST DURATION [ns]")]
                if host:
                    candidate["first_profiled_host_call_seconds"] = host[0]
                    candidate["steady_median_profiled_host_call_seconds"] = statistics.median(host[2:])
                candidate["profiled_nvfp4_kernel_seconds"] = sum(
                    float(r["DEVICE KERNEL DURATION [ns]"]) / 1e9 for r in rows
                )
        accuracy = directory / "accuracy.json"
        if accuracy.exists():
            try:
                report = json.loads(accuracy.read_text())
                summary["gate_datum_comparisons"] += report.get("valid_values_per_scale_case", 0) * len(
                    report.get("cases", [])
                )
            except json.JSONDecodeError:
                incomplete.append(str(accuracy))
        stage_directories = (
            {p.parent for p in directory.glob("*/pytest.xml")}
            | {p.parent for p in directory.glob("*/timing.json")}
            | {p.parent for p in directory.glob("*/command.json")}
        )
        for stage in sorted(stage_directories):
            summary["stages"] += 1
            junit = stage / "pytest.xml"
            counted_junits.add(junit.resolve())
            cases = []
            if junit.exists():
                try:
                    cases = ET.parse(junit).findall(".//testcase")
                except ET.ParseError:
                    incomplete.append(str(junit))
            else:
                summary["missing_junit_stages"] += 1
            for case in cases:
                outcome = (
                    "skipped"
                    if case.find("skipped") is not None
                    else "failed"
                    if case.find("failure") is not None or case.find("error") is not None
                    else "passed"
                )
                summary["pytest_counts"][outcome] += 1
            log = junit.parent / "run.log"
            if log.exists():
                text = log.read_text()
                metrics = {}
                cleanup_events = dict(
                    (name, float(value)) for name, value in re.findall(r"SAFE_PYTEST_TIMING: (\w+) ([\d.]+)", text)
                )
                if "runner_end" in cleanup_events and "pytest_end" in cleanup_events:
                    key = "runner_cleanup_seconds"
                    summary["summed_stage_seconds"][key] = summary["summed_stage_seconds"].get(key, 0) + (
                        cleanup_events["runner_end"] - cleanup_events["pytest_end"]
                    )
                for label, key in [
                    ("program compile (host waiting)", "program_compile_wait_seconds"),
                    ("JIT build wall time (local builds running)", "jit_build_busy_seconds"),
                ]:
                    match = re.search(re.escape(label) + r"\s+([\d.]+) s", text)
                    if match:
                        metrics[key] = float(match.group(1))
                if metrics:
                    summary["jit_audits"].append(dict(candidate=directory.name, stage=junit.parent.name, **metrics))
            timing = junit.parent / "timing.json"
            if not timing.exists():
                continue
            summary["detailed_timing_stages"] += 1
            data = json.loads(timing.read_text())
            for key in (
                "wall_seconds",
                "lock_wait_seconds",
                "preflight_seconds",
                "test_command_seconds",
                "python_startup_seconds",
                "collection_seconds",
                "post_session_seconds",
            ):
                value = data.get(key)
                if value is not None:
                    summary["summed_stage_seconds"][key] = summary["summed_stage_seconds"].get(key, 0) + value
            for key, value in data.get("pytest_phase_seconds", {}).items():
                summary["pytest_phase_seconds"][key] = summary["pytest_phase_seconds"].get(key, 0) + value
        summary["candidates"].append(candidate)
    auxiliary = dict(passed=0, failed=0, skipped=0)
    for path in root.rglob("*.xml"):
        if path.resolve() in counted_junits:
            continue
        try:
            cases = ET.parse(path).findall(".//testcase")
        except ET.ParseError:
            incomplete.append(str(path))
            continue
        for case in cases:
            outcome = (
                "skipped"
                if case.find("skipped") is not None
                else "failed"
                if case.find("failure") is not None or case.find("error") is not None
                else "passed"
            )
            auxiliary[outcome] += 1
    summary["auxiliary_junit_counts"] = auxiliary
    summary["total_recorded_pytest_counts"] = {key: summary["pytest_counts"][key] + auxiliary[key] for key in auxiliary}
    runners = {}
    for path in root.rglob("*.log"):
        if path.stat().st_size > 10_000_000:
            continue
        events = dict(
            (name, float(value))
            for name, value in re.findall(r"SAFE_PYTEST_TIMING: (\w+) ([\d.]+)", path.read_text(errors="replace"))
        )
        if "runner_start" in events and "runner_end" in events:
            runners[events["runner_start"]] = events
    summary["recorded_safe_runner_processes"] = len(runners)
    summary["all_runner_lock_held_seconds"] = sum(
        e["runner_end"] - e["lock_acquired"] for e in runners.values() if "lock_acquired" in e
    )
    summary["all_runner_cumulative_lock_wait_seconds"] = sum(
        e["lock_acquired"] - e["lock_wait_start"]
        for e in runners.values()
        if "lock_acquired" in e and "lock_wait_start" in e
    )
    summary["unique_source_and_encoding_snapshots"] = len(snapshots)
    if intervals:
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        summary["evaluation_active_union_seconds"] = sum(end - start for start, end in merged)
        summary["evaluation_window_seconds"] = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(summarize(args.root), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
