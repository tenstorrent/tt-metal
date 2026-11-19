#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only six-bucket reduced-prefill accounting from completed profiler CSVs.

Example: --profile-dir D/prefill_perf_v1 --driver D/prefill_ops_v1.json
--source-manifest D/final_sources_v2.json --resident D/exact_integrated_resident_v1.json
--output-prefix D/prefill_profile_analysis_v1. Existing outputs are refused.
Whole-model bucket markers cannot identify per-layer costs: this analyzer never
extrapolates reduced totals to all48 layers or interprets eager sums as TTFT.
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

BUCKETS = (1, 2, 4, 8, 16, 32)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_sources(report):
    provenance = report.get("performance", {}).get("source_provenance")
    if provenance is not None:
        before, after = provenance["before_setup_sha256"], provenance["after_performance_sha256"]
        if not provenance["unchanged"] or before != after:
            raise ValueError("Resident source snapshots changed during execution")
        return before
    return report.get("source_sha256", report.get("source_hashes", {}))


def sources(report):
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


def additional_source_conflicts(reports, expected):
    observed = {name: sources(report) for name, report in reports.items()}
    extra_keys = set().union(*(values.keys() for values in observed.values())) - expected.keys()
    return {
        key: {name: values[key] for name, values in observed.items() if key in values}
        for key in sorted(extra_keys)
        if len({values[key] for values in observed.values() if key in values}) > 1
    }


def analyze(args):
    outputs = [Path(str(args.output_prefix) + ext) for ext in (".json", ".md")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve previous evidence: choose a fresh output prefix")
    coverage_path = args.profile_dir / "coverage.json"
    raw_path = args.profile_dir / "ops_perf_results.csv"
    coverage, driver, manifest, resident = [
        json.loads(path.read_text()) for path in (coverage_path, args.driver, args.source_manifest, args.resident)
    ]
    for key in ("passed", "all_expected_signposts_preserved", "no_traces"):
        if coverage.get(key) is not True:
            raise ValueError(f"Coverage gate failed: {key}")
    if not driver["passed"] or not resident["passed"]:
        raise ValueError("Driver/resident evidence has not passed")
    if sha256(raw_path) != coverage["csv_sha256"] or sha256(args.driver) != coverage["driver_report_sha256"]:
        raise ValueError("Coverage hashes do not match raw CSV/driver")
    if (driver["layers"], driver["mesh"], driver["capacity"], driver["batch_size"], driver["buckets"]) != (
        [0, 1, 3],
        [1, 4],
        262144,
        1,
        list(BUCKETS),
    ):
        raise ValueError("Unexpected reduced-prefill scope")
    contract = driver["profile_contract"]
    if not contract["no_live_traces"] or not contract["profile_is_eager_not_trace_latency"]:
        raise ValueError("Expected eager operation attribution without traces")
    if contract["measured_host_uploads"] != 0 or contract["measured_host_reads"] != 0:
        raise ValueError("Host transfer inside measured region")
    measurements = {row["logical_rows"]: row for row in driver["measurements"]}
    if set(measurements) != set(BUCKETS) or len(driver["measurements"]) != len(BUCKETS):
        raise ValueError("Missing or duplicate bucket measurement")
    rows, summaries = {}, {}
    for bucket in BUCKETS:
        measured = measurements[bucket]
        if (measured["start_position"], measured["warmup_chunks"], measured["measured_chunks"]) != (0, 1, 1):
            raise ValueError(f"Unexpected chunk boundary for bucket{bucket}")
        for key in ("no_measured_program_compilation", "all_active_logits_finite", "no_live_traces"):
            if measured[key] is not True:
                raise ValueError(f"Bucket{bucket}: {key} failed")
        positions = measured["qsa_positions"]
        if {(row["rank"], row["position"]) for row in positions} != {(rank, bucket) for rank in range(4)}:
            raise ValueError(f"Bucket{bucket}: position mismatch")
        with (args.profile_dir / f"bucket_{bucket}_perf_report.csv").open() as stream:
            rows[bucket] = [row for row in csv.DictReader(stream) if row["Global Call Count"]]
        region = coverage["regions"][f"PREFILL_BUCKET_{bucket}"]
        if set(region["operations_per_device"]) != {"0", "1", "2", "3"} or set(
            region["operations_per_device"].values()
        ) != {len(rows[bucket])}:
            raise ValueError(f"Bucket{bucket}: merged/rank count mismatch")
        summaries[bucket] = {
            "logical_rows": bucket,
            "measurement": measured,
            "merged_ops": len(rows[bucket]),
            "merged_kernel_sum_ms": sum(float(row["Device Time [μs]"]) for row in rows[bucket]) / 1000,
            "eager_op_gap_ms": sum(float(row["Op-to-Op Gap [μs]"] or 0) for row in rows[bucket]) / 1000,
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
            if call_id not in ids:
                continue
            if call_id in raw:
                raise ValueError(f"Duplicate raw call ID{call_id}")
            raw[call_id] = {
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
    if ids != raw.keys():
        raise ValueError("Some merged rows lack raw geometry")
    for bucket in BUCKETS:
        groups = collections.defaultdict(lambda: {"kernel_ms": 0, "calls": 0, "example_call_id": None})
        advice = collections.Counter()
        for row in rows[bucket]:
            call_id = int(row["Global Call Count"])
            geometry = raw[call_id]
            key = (
                row["OP Code"].split(" ")[0],
                geometry.get("COMPUTE KERNEL SOURCE", ""),
                geometry.get("DATA MOVEMENT KERNEL SOURCE", ""),
            )
            group = groups[key]
            group["kernel_ms"] += float(row["Device Time [μs]"]) / 1000
            group["calls"] += 1
            group["example_call_id"] = group["example_call_id"] or call_id
            if row.get("Advice"):
                advice[row["Advice"]] += 1
        grouped = [
            {"op": key[0], "compute_kernel": key[1], "data_movement_kernel": key[2], **value}
            for key, value in groups.items()
        ]
        summaries[bucket]["kernel_groups"] = sorted(grouped, key=lambda group: group["kernel_ms"], reverse=True)
        summaries[bucket]["advice_occurrences"] = dict(advice.most_common())
    expected = sources(manifest)
    if not expected:
        raise ValueError("Manifest lacks implementation hashes")
    comparisons = {}
    for name, report in (("profile", driver), ("resident", resident)):
        observed = sources(report)
        comparisons[name] = {
            "mismatched": sorted(key for key in expected.keys() & observed.keys() if expected[key] != observed[key]),
            "missing": sorted(expected.keys() - observed.keys()),
            "additional": sorted(observed.keys() - expected.keys()),
        }
        comparisons[name]["exact_manifest_match"] = not (
            comparisons[name]["mismatched"] or comparisons[name]["missing"]
        )
        comparisons[name]["shared_sources"] = shared_source_comparison(report, manifest)
    report = {
        "scope": "Six separate warmed eager prefix chunks in reduced layers0/1/3; no all48 extrapolation",
        "stage_accepted": False,
        "semantic_advice_review_pending": True,
        "coverage": coverage,
        "prompt": driver["prompt"],
        "limitations": driver["limitations"],
        "buckets": summaries,
        "raw_geometry_by_call_id": raw,
        "source_manifest_comparison": comparisons,
        "additional_model_source_conflicts": additional_source_conflicts(
            {"profile": driver, "resident": resident}, expected
        ),
        "source_manifest_comparison_scope": "Frozen model-local source hashes; shared coverage listed separately",
        "full48_ttft_separate": {
            "ms": resident["performance"]["ttft_ms"],
            "scope": resident["performance"]["ttft_scope"],
            "workload": resident["performance"]["workload"],
            "artifact": str(args.resident),
        },
        "full48_prefill_device_estimate_ms": None,
        "missing_class_attribution": "Only whole-body bucket signposts exist; GR kernels repeat across layer boundaries. No unverified35/1/12 weighting of total or signature groups.",
        "advice_disposition_constraints": {
            "tracing": "Eager gaps are not traced TTFT savings; public generator already uses traced prefill",
            "precision": "Preserve selected fidelity/dtypes/KV/CCL, rejection ledger and four-stream residual layout",
            "program_configs": "Require exact geometry and numerical checks for each applicable bucket; padding does not make logical lengths interchangeable",
            "compaction": "Rejected in integrated mixed-prefill; isolated correctness/clean component gain does not authorize re-selection",
            "native": "Group by source/config/shapes, then source-validate semantic boundary before claiming layer-class attribution",
        },
    }
    inputs = [coverage_path, raw_path, args.driver, args.source_manifest, args.resident, Path(__file__)]
    inputs += [args.profile_dir / f"bucket_{bucket}_perf_report.csv" for bucket in BUCKETS]
    report["provenance_sha256"] = {str(path): sha256(path) for path in inputs}
    table = "\n".join(
        f"| {bucket} | {summaries[bucket]['merged_ops']} | {summaries[bucket]['merged_kernel_sum_ms']:.9f} | "
        f"{min(summaries[bucket]['raw_kernel_sum_ms_per_rank'].values()):.9f}–"
        f"{max(summaries[bucket]['raw_kernel_sum_ms_per_rank'].values()):.9f} |"
        for bucket in BUCKETS
    )
    md = f"""# Six-bucket reduced-prefill operation analysis

Each row measures one warmed prefix chunk from position0 through embeddings,
real layers0/1/3, final mixer/head and prefill-output copy. Inputs are uploaded
before the marker; drain, finiteness checks and scheduler updates occur after it.
No sampling, checkpoint save, live trace or measured caller read is included.

| Logical rows | Merged ops | Merged device sum, ms | Raw rank sum min–max, ms |
|---:|---:|---:|---:|
{table}

Ordinary operations use the report's slowest-rank merge, collectives its rank
mean. These are eager device sums with one measured chunk per bucket. The six
rows are alternative shapes, not sequential chunks to add together. No row is
an all48-layer latency, traced TTFT, statistical throughput or HF quality result.
The physical32-row padding does not establish semantic equivalence of buckets.

Separate all48 public-generator TTFT is **{resident['performance']['ttft_ms']:.6f} ms**
for the workload and boundary retained in JSON. It includes reset/setup, complete
prompt prefill, first sample and caller read. Do not subtract a reduced eager
sum from it to infer orchestration or multiply a reduced total by48/3.

Whole-body signposts do not supply individual L0/L1/L3 terms. Consequently the
weighted all48 prefill estimate is **unavailable**, not zero. Raw kernel groups,
per-op geometry and advice are retained for source-guided review; repeated GR
kernels cannot be assigned to a class from an op-name match alone. Any later
class reconstruction needs exact source sequence, boundary anchors and a complete
non-overlapping partition checked independently for all six buckets.

See JSON for source-manifest discrepancies, provenance and grouped kernel costs.
Advice remains pending human disposition: keep tracing distinct from eager gaps;
preserve precision and residual policy; validate geometry changes across logical
lengths; retain integrated compaction rejection. This report does not accept the
stage or claim measured full48 device-prefill accounting.
"""
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2) + "\n")
    outputs[1].write_text(md)
    print(json.dumps({"outputs": list(map(str, outputs)), "buckets": list(BUCKETS), "all48_device_estimate": None}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "driver", "source-manifest", "resident", "output-prefix"):
        parser.add_argument("--" + name, type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
