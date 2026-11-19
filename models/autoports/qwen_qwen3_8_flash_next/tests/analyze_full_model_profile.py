#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only final CSV/component accounting; never imports TTNN or changes old reports.

All input paths are explicit. Existing outputs are refused. Source disagreement is
reported, not silently treated as a matched experiment. Semantic candidate grouping
requires human review; no stale CSV row offsets are reused after graph changes.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

WEIGHTS = {"layer0": 35, "layer1": 1, "layer3": 12}
LEAVES = ["token_entry", "embedding", "layer0", "layer1", "layer3", "final_mixer", "lm_head", "row_assembly"]
REGIONS = LEAVES + ["split_greedy", "force_argmax", "model"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def signpost(name):
    return {"layer0": "PERF_LAYER_L0", "layer1": "PERF_LAYER_L1", "layer3": "PERF_LAYER_L3"}.get(
        name, "PERF_" + name.upper()
    )


def recorded_sources(report):
    provenance = report.get("performance", {}).get("source_provenance")
    if provenance is not None:
        before, after = provenance["before_setup_sha256"], provenance["after_performance_sha256"]
        if not provenance["unchanged"] or before != after:
            raise ValueError("Resident source snapshots changed during execution")
        return before
    return report.get("source_sha256", report.get("source_hashes", {}))


def source_map(report):
    return {key[key.index("tt/") :]: value for key, value in recorded_sources(report).items() if "tt/" in key}


def shared_source_comparison(report, manifest):
    expected = manifest.get("shared_source_sha256", {})
    observed = recorded_sources(report)
    return {
        "expected_count": len(expected),
        "missing": sorted(expected.keys() - observed.keys()),
        "mismatched": sorted(key for key in expected.keys() & observed.keys() if expected[key] != observed[key]),
        "scope": "Supplemental shared sources, separate from frozen model-local matching; missing hashes are not inferred",
    }


def raw_geometry(row):
    return {
        key: value
        for key, value in row.items()
        if value
        and (
            key
            in (
                "GLOBAL CALL COUNT",
                "DEVICE ID",
                "OP CODE",
                "CORE COUNT",
                "MATH FIDELITY",
                "ATTRIBUTES",
                "COMPUTE KERNEL SOURCE",
                "DATA MOVEMENT KERNEL SOURCE",
            )
            or key.startswith(("INPUT_0_", "INPUT_1_", "OUTPUT_0_"))
        )
    }


def additional_source_conflicts(reports, expected):
    observed = {name: source_map(report) for name, report in reports.items()}
    extra_keys = set().union(*(values.keys() for values in observed.values())) - expected.keys()
    return {
        key: {name: values[key] for name, values in observed.items() if key in values}
        for key in sorted(extra_keys)
        if len({values[key] for values in observed.values() if key in values}) > 1
    }


def analyze(args):
    outputs = [Path(str(args.output_prefix) + suffix) for suffix in (".json", ".md")]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output prefix")
    coverage_path = args.profile_dir / "coverage.json"
    raw_path = args.profile_dir / "ops_perf_results.csv"
    coverage, driver, trace, resident, floor, manifest = map(
        read_json, [coverage_path, args.driver, args.component, args.resident, args.floor, args.source_manifest]
    )
    for key in (
        "passed",
        "all_expected_signposts_preserved",
        "leaves_partition_model",
        "terminal_parent_equals_mixer_plus_head",
        "no_traces",
    ):
        if coverage.get(key) is not True:
            raise ValueError(f"Coverage gate failed: {key}")
    if not all(report.get("passed") is True for report in (driver, trace, resident)):
        raise ValueError("Driver, component and resident artifacts must all pass their own checks")
    if sha256(raw_path) != coverage["csv_sha256"] or sha256(args.driver) != coverage["driver_report_sha256"]:
        raise ValueError("Coverage provenance does not match supplied CSV/driver")
    prefix = driver["profile_prefix"]
    if (driver["layers"], driver["capacity"], driver["batch_size"], prefix["logical_length"]) != (
        [0, 1, 3],
        262144,
        1,
        128,
    ):
        raise ValueError("Expected reduced layers0/1/3, C262144, B1, real128-token prefix")
    if not driver["profile_contract"]["no_live_traces"]:
        raise ValueError("Op attribution requires no live traces")
    if (prefix["measured_start_position"], prefix["measured_end_position"]) != (132, 133):
        raise ValueError("Expected measured position132 to133 after four warmups")
    positions = prefix["positions_after_measured_decode"]
    if {(row["rank"], row["position"]) for row in positions} != {(rank, 133) for rank in range(4)}:
        raise ValueError("Measured position did not advance on all four ranks")
    functional_ms = (
        1000
        * (floor["weight_bytes_per_rank"] + floor["kv_bytes_per_rank"])
        / floor["bandwidth_bytes_per_second_per_rank"]
    )
    if not math.isclose(functional_ms, floor["roofline_ms"]) or not math.isclose(
        functional_ms / 0.10, floor["ten_percent_max_latency_ms"]
    ):
        raise ValueError("Strict useful-weight/KV floor arithmetic differs from artifact")
    if trace["weighted_estimate"]["multiplicities"] != WEIGHTS:
        raise ValueError("Unexpected layer multiplicities")
    if (trace["layers"], trace["capacity"], trace["batch_size"], trace["prompt_length"]) != ([0, 1, 3], 262144, 1, 128):
        raise ValueError("Component scope differs from profile")
    for name, component in trace["components"].items():
        for key in (
            "first_replay_exact",
            "all_outputs_finite",
            "persistent_input",
            "nonblocking_replay",
            "private_input_snapshots_unchanged_after_every_window",
        ):
            if component.get(key) is not True:
                raise ValueError(f"Component {name}: failed {key}")
        if component["interior_host_reads_or_syncs"] != 0 or component["state_restoration_inside_window"]:
            raise ValueError(f"Component {name}: invalid timing boundary")
    rows = {}
    summaries = {}
    for name in REGIONS:
        with (args.profile_dir / f"{name}_perf_report.csv").open() as stream:
            rows[name] = [row for row in csv.DictReader(stream) if row["Global Call Count"]]
        region = coverage["regions"][signpost(name)]
        if set(region["operations_per_device"]) != {"0", "1", "2", "3"}:
            raise ValueError(f"Missing rank in {name}")
        if set(region["operations_per_device"].values()) != {len(rows[name])}:
            raise ValueError(f"Merged-row/rank count mismatch in {name}")
        summaries[name] = {
            "merged_ops": len(rows[name]),
            "merged_kernel_sum_ms": sum(float(row["Device Time [μs]"]) for row in rows[name]) / 1000,
            "merged_eager_op_gap_ms": sum(float(row["Op-to-Op Gap [μs]"] or 0) for row in rows[name]) / 1000,
            "raw_kernel_sum_ms_per_rank": region["kernel_sum_ms_per_device"],
        }
    ids = {int(row["Global Call Count"]) for region in rows.values() for row in region}
    raw = {}
    with raw_path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                call_id = int(row["GLOBAL CALL COUNT"])
            except ValueError:
                continue
            if call_id in ids:
                if call_id in raw:
                    raise ValueError(f"Ambiguous raw call ID {call_id}")
                raw[call_id] = raw_geometry(row)
    if ids != raw.keys():
        raise ValueError("Some final merged rows have no raw geometry")
    selected = LEAVES + ["split_greedy"]
    weighted = sum(WEIGHTS.get(name, 1) * summaries[name]["merged_kernel_sum_ms"] for name in selected)
    per_rank = {
        str(rank): sum(
            WEIGHTS.get(name, 1) * summaries[name]["raw_kernel_sum_ms_per_rank"][str(rank)] for name in selected
        )
        for rank in range(4)
    }
    groups = collections.defaultdict(lambda: {"weighted_ms": 0, "weighted_calls": 0, "example_call_id": None})
    advice = collections.Counter()
    for name in selected:
        for row in rows[name]:
            call_id = int(row["Global Call Count"])
            geometry = raw[call_id]
            key = (
                row["OP Code"].split(" ")[0],
                geometry.get("COMPUTE KERNEL SOURCE", ""),
                geometry.get("DATA MOVEMENT KERNEL SOURCE", ""),
            )
            group = groups[key]
            group["weighted_ms"] += WEIGHTS.get(name, 1) * float(row["Device Time [μs]"]) / 1000
            group["weighted_calls"] += WEIGHTS.get(name, 1)
            group["example_call_id"] = group["example_call_id"] or call_id
            if row.get("Advice"):
                advice[row["Advice"]] += WEIGHTS.get(name, 1)
    group_list = [
        {"op": key[0], "compute_kernel": key[1], "data_movement_kernel": key[2], **value}
        for key, value in groups.items()
    ]
    group_list.sort(key=lambda group: group["weighted_ms"], reverse=True)
    expected = source_map(manifest)
    if not expected:
        raise ValueError("Source manifest has no TT implementation hashes")
    source_checks = {}
    for name, report in (("driver", driver), ("component", trace), ("resident", resident)):
        observed = source_map(report)
        source_checks[name] = {
            "mismatched": [key for key in expected.keys() & observed.keys() if expected[key] != observed[key]],
            "missing": sorted(expected.keys() - observed.keys()),
            "additional": sorted(observed.keys() - expected.keys()),
        }
        source_checks[name]["exact_manifest_match"] = not (
            source_checks[name]["mismatched"] or source_checks[name]["missing"]
        )
        source_checks[name]["shared_sources"] = shared_source_comparison(report, manifest)
    performance = resident["performance"]
    boundary = performance["resident_token_out"]
    if boundary["blocking_replay"] or not boundary["final_readback_outside_timing"]:
        raise ValueError("Resident timing must use nonblocking replay and final read outside window")
    for key in ("readbacks", "synchronizations", "token_refreshes", "position_refreshes", "page_table_refreshes"):
        if boundary["steady_decode_counter_delta"][key] != 0:
            raise ValueError(f"Resident timing has interior host-boundary activity: {key}")
    latency = boundary["ms_per_step"]
    stack = trace["weighted_estimate"]["stack_plus_terminal_ms"]
    split = summaries["split_greedy"]["merged_kernel_sum_ms"]
    extra_conflicts = additional_source_conflicts(
        {"driver": driver, "component": trace, "resident": resident}, expected
    )
    source_match = not extra_conflicts and all(
        value["exact_manifest_match"] and not value["shared_sources"]["mismatched"] for value in source_checks.values()
    )
    report = {
        "scope": "Reduced eager device-operation sums, separate frozen-input component traces, and all48 resident token-out; CPU-only analysis",
        "stage_accepted": False,
        "semantic_advice_review_pending": True,
        "coverage": coverage,
        "profile_prefix": prefix,
        "regions": summaries,
        "weighted_eager": {
            "multiplicities": WEIGHTS,
            "merged_kernel_ms": weighted,
            "kernel_ms_per_rank": per_rank,
            "scope": "Representative device sums, not a physical lower bound or all48 trace latency",
        },
        "component_trace_weighted_estimate": trace["weighted_estimate"],
        "component_limitations": trace["limitations"],
        "source_manifest_comparison": source_checks,
        "additional_model_source_conflicts": extra_conflicts,
        "matched_sources": source_match,
        "matched_sources_scope": "All frozen model-local source hashes; supplemental shared-source coverage is listed separately",
        "full_model_comparison": {
            "all48_resident_token_out_ms": latency,
            "boundary": boundary["scope"],
            "caller_read_token_out_ms": performance["token_out_decode_ms"],
            "ttft_ms": performance["ttft_ms"],
            "relative_to_representative_trace_estimate_percent": 100 * (latency / stack - 1),
            "comparison_is_matched_source": source_match,
            "positive_10_percent_gap_numerically": latency > 1.10 * stack,
            "strict_floor_ms": floor["roofline_ms"],
            "strict_ten_percent_max_latency_ms": floor["ten_percent_max_latency_ms"],
            "strict_roofline_fraction": floor["roofline_ms"] / latency,
            "strict_gate_passed": latency <= floor["ten_percent_max_latency_ms"],
            "remaining_ms_to_remove": latency - floor["ten_percent_max_latency_ms"],
        },
        "sampler": {
            "split_merged_kernel_ms": split,
            "force_argmax_merged_kernel_ms": summaries["force_argmax"]["merged_kernel_sum_ms"],
            "split_fraction_of_all48_percent": 100 * split / latency,
            "generic_TopK_in_split": any("TopKDeviceOperation" in row["OP Code"] for row in rows["split_greedy"]),
            "raw_geometry": [raw[int(row["Global Call Count"])] for row in rows["split_greedy"]],
            "correctness": driver["correctness"],
            "full_vocab_gather_and_native_domination": "Review retained raw shapes and weighted kernel table; not inferred from op names",
        },
        "weighted_operation_kernel_groups": group_list,
        "raw_geometry_by_call_id": raw,
        "advice_weighted_occurrences": dict(advice.most_common()),
        "advice_disposition_constraints": {
            "tracing": "Already enabled in resident path; eager gaps are not extra achievable trace savings",
            "lower_fidelity": "Explicit user constraint and inherited rejection ledger forbid changing selected policy",
            "head_geometry": "Use raw config; report DRAM-sharded core heuristic is not actual worker count",
            "other": "Human reconcile with exact-fusion, head, sampler and compaction rejection artifacts before closure",
        },
    }
    inputs = [
        coverage_path,
        raw_path,
        args.driver,
        args.component,
        args.resident,
        args.floor,
        args.source_manifest,
        Path(__file__),
    ]
    inputs += [args.profile_dir / f"{name}_perf_report.csv" for name in REGIONS]
    report["provenance_sha256"] = {str(path): sha256(path) for path in inputs}
    table = "\n".join(
        f"| {name} | {value['merged_ops']} | {value['merged_kernel_sum_ms']:.9f} | "
        f"{WEIGHTS.get(name, 1) if name in selected else 'enclosing/alternative'} |"
        for name, value in summaries.items()
    )
    md = f"""# Full-model final profile reconciliation

Resident token-out: **{latency:.9f} ms**. Strict useful-weight/KV floor:
**{floor['roofline_ms']:.6f} ms**; ten-percent ceiling:
**{floor['ten_percent_max_latency_ms']:.6f} ms**. Remaining deficit:
**{latency - floor['ten_percent_max_latency_ms']:.9f} ms**.
No intermediate traffic or measured kernel time is added to the denominator.

Representative component stack plus entry/terminal/sampler: **{stack:.9f} ms**;
full-model difference: **{100 * (latency / stack - 1):.6f}%**.
Frozen model-local source-manifest match across all inputs: **{source_match}**. Consult JSON
for missing/additional/mismatched paths and separate shared-source coverage.
Additional recorded model files are listed; they are not silently removed.

| Region | Merged ops | Merged kernel sum, ms | Full-path weight |
|---|---:|---:|---:|
{table}

Weighted35/1/12 eager sum plus entry/embedding/mixer/head/row assembly/split:
**{weighted:.9f} ms**. Raw rank totals: {json.dumps(per_rank)}.
Ordinary merged operations use slowest rank; CCL rows use the report's rank mean.
Enclosing model and terminal regions are never added to their leaves. Eager gaps
are retained in JSON, not treated as clean trace overhead or achievable savings.

Split sampler: **{split:.9f} ms** device sum, **{100 * split / latency:.6f}%**
of resident token-out. Force-argmax is a separately measured alternative.
Review raw sampler gather shapes and native operation contributions in JSON.
The all48 boundary includes split sampling/device feedback; caller-read latency
and TTFT are separate fields. Component traces use reduced residency and frozen
reachable inputs; their estimate is not an exact all48 measured lower bound.

Coverage and input hashes, all selected raw geometries, sorted weighted kernel
groups and report advice are in the companion JSON. Kernel groups may combine
multiple semantic uses and are upper bounds, not predicted savings. Semantic
candidate/advice disposition and final stage review remain pending. This script
does not accept the stage, even when an individual numerical gate passes.
"""
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2) + "\n")
    outputs[1].write_text(md)
    print(
        json.dumps(
            {
                "outputs": list(map(str, outputs)),
                "weighted_eager_ms": weighted,
                "resident_ms": latency,
                "matched_sources": source_match,
                "matched_sources_scope": "All frozen model-local source hashes; supplemental shared-source coverage is listed separately",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "driver", "component", "resident", "floor", "source-manifest", "output-prefix"):
        parser.add_argument("--" + name, type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
