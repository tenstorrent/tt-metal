# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only native Tracy conversion and four-rank component attribution.

--mode convert preserves the raw capture and uses the stage5 compaction helpers.
--mode summarize consumes ops.csv plus the three tt-perf-report CSV/text files.
Neither mode imports TTNN or opens devices. Use the accompanying capture shell.
"""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.postprocess_multichip_profile import compact, compact_device
from models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_acceptance import (
    capture_health,
    fidelities,
    load_program_metadata,
    program_key,
    validate_executions,
    validate_metadata,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_perf import (
    DURATION,
    GROUP_KEYS,
    SESSION,
    category,
    number,
    read_csv,
    require,
    signature,
    window,
)

PHASES = ("token_out", "model_only", "sampler_only")
RANKS = ("0", "1", "2", "3")
TRACE = "METAL TRACE ID"
ATTRIBUTION_KEYS = (*GROUP_KEYS, "PROGRAM HASH")


def metadata(artifact):
    capture_health(artifact)
    result = json.loads((artifact / "performance.json").read_text())
    require(
        result["passed"] and result["profile"] and not result["headline_eligible"],
        "Require successful diagnostic profile",
    )
    require(result["layers"] == [0, 3] and result["batch_size"] == 1, "Require reduced real layers0/3 B1")
    require(result["steps_per_component"] == 16, "Require 16 measured replays per component")
    require(result["fixture"]["prefix_length"] == 128, "Require canonical 128-token performance fixture")
    require(result["status"] == "PASS", "Require final successful workload status")
    for key in ("TT_METAL_PROFILER_MID_RUN_DUMP", "TT_METAL_PROFILER_CPP_POST_PROCESS"):
        require(result["environment"].get(key) == "1", f"Require {key}=1")
    for phase in PHASES:
        record = result["measurements"][phase]
        require(record["steps"] == 16 and record["runtime_device_only_guard"] == "pass", phase)
        require(record["program_cache_entries"] == result["program_cache_entries_after_prepare"], "Program growth")
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "readbacks", "captures"):
            require(record["loop_counter_delta"][key] == 0, f"{phase}: unexpected {key}")
    require(result["measurements"]["token_out"]["initial_position"] == 128, "Wrong token-out start")
    require(result["measurements"]["model_only"]["initial_position"] == 128, "Wrong model-only start")
    return result


def acceptance(artifact, result, rows):
    capture_health(artifact)
    windows = {phase: window(rows, phase) for phase in PHASES}
    raw = artifact / "raw/.logs"
    with (raw / "cpp_device_perf_report.csv").open(newline="") as stream:
        coverage = validate_executions(windows, csv.DictReader(stream), result["trace_ids"])
    selected = [row for values in windows.values() for row in values]
    programs = load_program_metadata(raw / "tracy_ops_data.csv", {program_key(row) for row in selected})
    inventory = validate_metadata(selected, programs)
    return coverage, programs, inventory


def convert(artifact):
    from tracy.common import PROFILER_BIN_DIR, TRACY_CSVEXPROT_TOOL, resolve_tracy_tool_path
    from tracy.process_ops_logs import process_ops

    metadata(artifact)
    raw = artifact / "raw/.logs"
    trace = raw / "tracy_profile_log_host.tracy"
    require(trace.is_file() and (raw / "cpp_device_perf_report.csv").is_file(), "Missing native trace/device report")
    exporter = resolve_tracy_tool_path(PROFILER_BIN_DIR, TRACY_CSVEXPROT_TOOL)
    require(bool(exporter), "Native Tracy exporter unavailable")
    for name, flags in (
        ("tracy_ops_root_times.csv", ["-u", "-c", "-f", "TT_"]),
        ("tracy_ops_data.csv", ["-m", "-s", ";"]),
    ):
        target = raw / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            require(not partial.exists(), "Preserve and inspect interrupted export")
            with partial.open("w") as stream:
                subprocess.run([str(exporter), *flags, str(trace)], stdout=stream, check=True, timeout=1800)
            partial.replace(target)
    processing = artifact / "postprocess"
    require(not processing.exists(), "Preserve existing postprocess results; summarize without reconversion")
    logs = processing / ".logs"
    logs.mkdir(parents=True)
    for source in raw.iterdir():
        if source.name not in ("tracy_ops_times.csv", "tracy_ops_root_times.csv", "profile_log_device.csv"):
            (logs / source.name).symlink_to(source)
    counts = {
        "host": compact(raw / "tracy_ops_root_times.csv", logs / "tracy_ops_times.csv"),
        "device_metadata": compact_device(raw / "profile_log_device.csv", logs / "profile_log_device.csv"),
        "semantics": "Stage5 native helpers: root ordering unchanged; optional compile/write child host timings omitted; device timing unchanged",
    }
    (artifact / "compaction.json").write_text(json.dumps(counts, indent=2) + "\n")
    process_ops(processing, name_append=None, date=True)
    reports = list((processing / "reports").rglob("ops_perf_results_*.csv"))
    require(len(reports) == 1, f"Expected one native op report, got {reports}")
    shutil.copy2(reports[0], artifact / "ops.csv")
    (artifact / "ops_source.txt").write_text(str(reports[0]) + "\n")
    rows, _ = read_csv(artifact / "ops.csv", ("OP TYPE", "OP CODE", "DEVICE ID", DURATION, SESSION, TRACE))
    acceptance(artifact, metadata(artifact), rows)


def summarize(artifact):
    result = metadata(artifact)
    rows, fields = read_csv(artifact / "ops.csv", ("OP TYPE", "OP CODE", "DEVICE ID", DURATION, SESSION, TRACE))
    traces = {name: str(value) for name, value in result["trace_ids"].items()}
    require(set(traces) == {"model", "sampling"}, "Wrong captured component identities")
    coverage, programs, inventory = acceptance(artifact, result, rows)
    inventory_path = artifact / "native_program_metadata.json"
    summary = {
        "status": "validated_profile",
        "headline_eligible": False,
        "scope": "Real layers0/3 plus full embedding/norm/LM-head, greedy split sampler; batch1/context"
        + str(result["context"]),
        "interpretation": "Kernel sums are operation-work attribution, not wall latency. Four ranks are parallel, never summed into latency. Fenced host samples include instrumentation. Separate fixed-token model and fixed-logit sampler controls are not additive autoregressive decomposition.",
        "regime": {
            key: result[key]
            for key in (
                "layers",
                "context",
                "batch_size",
                "steps_per_component",
                "fixture",
                "environment",
                "source_sha256",
            )
        },
        "phases": {},
        "artifacts": {},
        "execution_coverage": coverage,
        "native_program_metadata": str(inventory_path),
    }
    trace_means = {}
    for phase in PHASES:
        selected = window(rows, phase)
        names = (
            ("model", "sampling") if phase == "token_out" else (("model",) if phase == "model_only" else ("sampling",))
        )
        require({row[TRACE] for row in selected} == {traces[name] for name in names}, f"{phase}: wrong trace IDs")
        components, groups = {}, []
        for name in names:
            samples, sessions_by_rank, ops_per_replay = {}, {}, {}
            for rank in RANKS:
                chosen = [row for row in selected if row["DEVICE ID"] == rank and row[TRACE] == traces[name]]
                sessions = defaultdict(list)
                for row in chosen:
                    require(row[SESSION] not in ("", "-", "None", "N/A"), "Missing replay session")
                    sessions[row[SESSION]].append(row)
                require(len(sessions) == 16, f"{phase}/{name}/rank{rank}: expected16 sessions, got{len(sessions)}")
                replays = list(sessions.values())
                expected = [signature(row, fields) for row in replays[0]]
                require(
                    all([signature(row, fields) for row in replay] == expected for replay in replays),
                    f"{phase}/{name}/rank{rank}: incomplete replay signatures",
                )
                samples[rank] = [
                    sum(number(row[DURATION], "duration", positive=True) for row in replay) / 1e6 for replay in replays
                ]
                sessions_by_rank[rank] = list(sessions)
                ops_per_replay[rank] = len(replays[0])
                grouped = defaultdict(lambda: [0, 0.0])
                for row in chosen:
                    key = (
                        category(row),
                        *(
                            fidelities(row, programs) if key == "MATH FIDELITY" else row.get(key, "")
                            for key in ATTRIBUTION_KEYS
                        ),
                    )
                    grouped[key][0] += 1
                    grouped[key][1] += float(row[DURATION]) / 1000 / 16
                for key, (count, duration) in grouped.items():
                    groups.append(
                        dict(
                            component=name,
                            rank=rank,
                            category=key[0],
                            **dict(zip(ATTRIBUTION_KEYS, key[1:])),
                            count_per_replay=count / 16,
                            kernel_us_per_replay=duration,
                        )
                    )
            means = {rank: statistics.mean(values) for rank, values in samples.items()}
            trace_means[(phase, name)] = means
            components[name] = dict(
                per_rank_kernel_sum_ms=means,
                kernel_sum_samples_ms=samples,
                maximum_rank_mean_kernel_sum_ms=max(means.values()),
                replay_sessions=sessions_by_rank,
                ops_per_replay=ops_per_replay,
            )
        group_path = artifact / f"{phase}_op_groups.csv"
        groups.sort(key=lambda row: (row["component"], row["rank"], -row["kernel_us_per_replay"]))
        with group_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
        report, _ = read_csv(artifact / f"{phase}_perf_report.csv", ("Device", "OP Code", "Device Time [μs]", "Advice"))
        text_path = artifact / f"{phase}_perf_report.txt"
        require(text_path.is_file() and text_path.stat().st_size > 0, "Missing human tt-perf-report")
        summary["phases"][phase] = {
            "components": components,
            "host_fenced_measurement": result["measurements"][phase],
            "top_actual_op_groups": sorted(groups, key=lambda row: -row["kernel_us_per_replay"])[:24],
            "all_groups_csv": str(group_path),
            "dtype_fidelity_note": "Operand metadata validated against native JSON; fidelity is the full native compute-kernel fidelity set ([] only with explicit DM-only kernel inventory). All operands and kernels retained in native_program_metadata.json; ops.csv is unchanged.",
            "tt_perf_report": {
                "csv": str(artifact / f"{phase}_perf_report.csv"),
                "text": str(text_path),
                "merged_rows": len(report),
                "scope": "Merged representative ranks; do not sum as per-rank latency",
                "advice": sorted({row["Advice"] for row in report if row["Advice"].strip()}),
            },
        }
    model, sampler = trace_means[("token_out", "model")], trace_means[("token_out", "sampling")]
    critical = max(RANKS, key=lambda rank: model[rank] + sampler[rank])
    summary["token_out_attribution"] = {
        "rank_with_maximum_total_kernel_sum": critical,
        "model_kernel_sum_ms": model[critical],
        "sampler_kernel_sum_ms": sampler[critical],
        "sampler_fraction_of_same_rank_kernel_work": sampler[critical] / (model[critical] + sampler[critical]),
        "sampler_dominates_kernel_work": sampler[critical] > model[critical],
        "decision": "If sampler dominates, inspect actual LM-head/sampler op rows before full-model completion; reduced-stack ratio is diagnostic and cannot predict all40 ratio.",
    }
    for path in [
        artifact / "performance.json",
        artifact / "ops.csv",
        Path(__file__),
        Path(__file__).with_name("profile_acceptance.py"),
    ]:
        summary["artifacts"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["artifacts"]["native_host_trace"] = str(artifact / "raw/.logs/tracy_profile_log_host.tracy")
    summary["artifacts"]["native_device_report"] = str(artifact / "raw/.logs/cpp_device_perf_report.csv")
    inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    summary["artifacts"][str(inventory_path)] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    (artifact / "component_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["token_out_attribution"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mode", choices=("convert", "summarize"), required=True)
    args = parser.parse_args()
    (convert if args.mode == "convert" else summarize)(args.artifact.resolve())
