# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the saved, interleaved experiment-2 runs without touching hardware.

python -m models.demos.llama31_8b_qb2.tests.summarize_experiment2 \
    --artifacts /path/to/artifacts --output /path/to/benchmark-summary.json

Process medians, individual trials and paired differences are retained. Trials
within one process are not treated as independent process-level replications.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def distribution(values):
    median = statistics.median(values)
    return {
        "median": median,
        "minimum": min(values),
        "maximum": max(values),
        "median_absolute_deviation": statistics.median(abs(x - median) for x in values),
    }


def summarize(artifacts):
    plan_path = artifacts / "final-paired-plan.json"
    manifest = json.loads(plan_path.read_text())
    result = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_sha": manifest["source_sha"],
        "configuration": manifest["config"],
        "measurement": manifest["scope"],
        "statistics": "Median of process medians; five warmed trials per process. Paired differences use process medians within the same case/round. No independence claim for trials within a process.",
        "quality_scope": "Fixed prompt, batch one, matched quantized TT teacher stream. Exact teacher logits, all64 request-touched K/V tensors including padding, greedy output and repeated generations. This is not broad quality qualification. Both native and prototype failed the separate provisional HF PCC0.99 diagnostic.",
        "runs": [],
        "cases": {},
    }
    identity = None
    for entry in manifest["plan"]:
        assert entry.get("completed_utc"), f"Incomplete run: {entry['name']}"
        path = artifacts / (entry["name"] + "-result") / "result.json"
        run_path = artifacts / (entry["name"] + ".json")
        d = json.loads(path.read_text())
        record = json.loads(run_path.read_text())
        assert record["returncode"] == 0 and not record.get("timed_out")
        assert not any(record["selected_environment"].get(key) for key in (
            "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_SUM", "TT_METAL_DEVICE_PROFILER_NOC_EVENTS"
        )), "Instrumented run cannot enter headline statistics"
        assert not d["profile_run"] and not d["watcher_enabled"] and d["headline_latency_eligible"]
        assert d["comparison_checks_passed"] and d["agreement_policy"] == "exact"
        assert d["teacher_logits"]["exact"] and all(x["exact"] for x in d["teacher_step_metrics"])
        assert len(d["teacher_cache_metrics"]) == 32
        assert all(x["exact"] for pair in d["teacher_cache_metrics"] for x in pair)
        assert d["teacher_top1_agreement"] == d["greedy_token_agreement"] == 1.0
        current_identity = (d["checkpoint_revision"], d["precision"], d["sampling"], d["batch"])
        if identity is None:
            identity = current_identity
            result["checkpoint_revision"], result["precision"], result["sampling"], result["batch"] = identity
        assert current_identity == identity
        assert d["context"] == entry["context"] and d["tokens"] == entry["tokens"]
        assert d["timing_denominator"] == d["tokens"] - 1
        assert len(d["generation"]) == 5
        trials = []
        for g in d["generation"]:
            assert g["decode_tokens"] == d["timing_denominator"]
            assert not g["teacher_forcing"] and not g["host_sampling"] and not g["readback_each_token"]
            counters = g["steady_counters"]
            assert counters["model_trace_replays"] == counters["sampling_trace_replays"] == g["decode_tokens"]
            assert counters["history_readbacks"] == 1
            assert counters["cache_resets"] == counters["capture_synchronizations"] == 0
            trials.append(g["decode_ms"] / g["decode_tokens"])
        run = {k: entry[k] for k in ("name", "round", "case", "context", "tokens", "implementation", "completed_utc")}
        run.update(
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha=d["sha"],
            source_status=d["source_status"],
            command=record["command"],
            timing_denominator=d["timing_denominator"],
            trials_ms_per_token=trials,
            process_median_ms_per_token=statistics.median(trials),
            model_load_seconds=d["model_load_seconds"],
            experimental_setup_seconds=d["experimental_setup_seconds"],
            warmup_capture_generation_seconds=d["warmup_capture_generation_seconds"],
            warmup_perf=d["warmup_perf"],
            warmed_ttft_ms=[g["ttft_ms"] for g in d["generation"]],
            warmed_request_setup_ms=[g["setup_ms"] for g in d["generation"]],
            exact=True,
            generated_tokens_sha256=hashlib.sha256(json.dumps(d["generated_tokens"]).encode()).hexdigest(),
            teacher_position_range=[d["teacher_positions"][0], d["teacher_positions"][-1]],
        )
        result["runs"].append(run)
    for case in dict.fromkeys(r["case"] for r in result["runs"]):
        runs = [r for r in result["runs"] if r["case"] == case]
        grouped = {}
        for mode in dict.fromkeys(r["implementation"] for r in runs):
            selected = [r for r in runs if r["implementation"] == mode]
            grouped[mode] = dict(
                process_count=len(selected),
                trial_count=sum(len(r["trials_ms_per_token"]) for r in selected),
                process_medians_ms_per_token=[r["process_median_ms_per_token"] for r in selected],
                process_median_distribution_ms_per_token=distribution([r["process_median_ms_per_token"] for r in selected]),
                trial_distribution_ms_per_token=distribution([x for r in selected for x in r["trials_ms_per_token"]]),
                model_load_seconds=distribution([r["model_load_seconds"] for r in selected]),
                experimental_setup_seconds=distribution([r["experimental_setup_seconds"] for r in selected]),
                warmup_capture_generation_seconds=distribution([r["warmup_capture_generation_seconds"] for r in selected]),
            )
        paired = {}
        for mode in grouped:
            if mode == "native":
                continue
            differences = []
            for candidate in (r for r in runs if r["implementation"] == mode):
                native = next(r for r in runs if r["implementation"] == "native" and r["round"] == candidate["round"])
                n, c = native["process_median_ms_per_token"], candidate["process_median_ms_per_token"]
                differences.append(dict(round=candidate["round"], saved_ms_per_token=n - c, latency_reduction_percent=100 * (n - c) / n))
            paired[mode] = dict(by_round=differences, saved_ms_per_token=distribution([p["saved_ms_per_token"] for p in differences]), latency_reduction_percent=distribution([p["latency_reduction_percent"] for p in differences]))
        result["cases"][case] = dict(context=runs[0]["context"], outputs=runs[0]["tokens"], decode_steps=runs[0]["timing_denominator"], implementations=grouped, paired_against_native=paired)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.artifacts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case, values in result["cases"].items():
        print(case, {mode: round(v["process_median_distribution_ms_per_token"]["median"], 6) for mode, v in values["implementations"].items()})


if __name__ == "__main__":
    main()
