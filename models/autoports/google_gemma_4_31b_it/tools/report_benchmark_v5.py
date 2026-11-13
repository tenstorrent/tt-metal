#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble this run's measured grid without importing historical results."""

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
BASE = MODEL / "doc/benchmark"
RUN = BASE / "run"


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    plan = read(BASE / "plan.json")
    if (BASE / "anomaly_check.json").exists() and (BASE / "semantic_concerns.json").exists():
        anomaly = read(BASE / "anomaly_check.json")
        anomaly["semantic_concerns"] = read(BASE / "semantic_concerns.json")
        anomaly["semantic_concerns_count"] = len(anomaly["semantic_concerns"])
        anomaly[
            "anomalies_definition"
        ] = "Mechanical degeneration detector flags only; semantic concerns are listed separately and handed to stage12."
        save(BASE / "anomaly_check.json", anomaly)
    fallback_path = BASE / "setup/native_fallback.json"
    fallback = read(fallback_path) if fallback_path.exists() else {}
    carried = {r["row"]: r for r in fallback.get("preserved_rows", [])}
    measured = []
    releases = []
    missing = []
    deferred_path = BASE / "deferred_rows.jsonl"
    deferred = {}
    if deferred_path.exists():
        for line in deferred_path.read_text().splitlines():
            item = json.loads(line)
            deferred[item["row"]] = item
    for row in plan["rows"]:
        b = row["batch"]
        directory = RUN if row.get("headline") else RUN / "grid" / row["id"]
        if not row.get("headline") and not (directory / "complete.json").exists():
            directory = next(
                (
                    p / "grid" / row["id"]
                    for p in RUN.glob("supplement-*")
                    if (p / "grid" / row["id"] / "complete.json").exists()
                ),
                directory,
            )
        rawpath = directory / f"perf-b{b}.json"
        if not rawpath.exists() or (not row.get("headline") and not (directory / "complete.json").exists()):
            missing.append(
                {
                    **row,
                    "reason": row.get(
                        "missing_reason",
                        (
                            "Not yet measured; run in progress."
                            if args.draft
                            else "Insufficient remaining stage budget for warmup, complete measurement and required closure; no substituted result."
                        ),
                    ),
                }
            )
            if row["id"] in deferred:
                missing[-1]["budget_admission"] = deferred[row["id"]]
                missing[-1]["budget_evidence"] = "deferred_rows.jsonl"
            continue
        raw = read(rawpath)
        if raw.get("failed") or raw.get("completed") != raw.get("num_prompts"):
            missing.append({**row, "reason": "Attempt incomplete; retained raw failure, no rate reported."})
            continue
        assert raw["input_lens"] == [row["input_tokens"]] * raw["completed"], row["id"]
        assert raw["output_lens"] == [row["output_tokens"]] * raw["completed"], row["id"]
        source_identity = read(directory / f"perf-b{b}-server.json")
        assert source_identity["max_num_seqs"] == b and source_identity["max_model_len"] == 262144
        assert source_identity["precision"]["config_id"] == "head_bfp4"
        assert source_identity["configured_layer_count"] == source_identity["layer_count"] == 60
        roof = read(directory / "roofline.json")[str(b)]
        detail = read(directory / roof["prefill"]["evidence"])
        modes = {}
        for step in detail["steps"]:
            k = step["phase"] + ":" + step["sampling_mode"]
            modes[k] = modes.get(k, 0) + 1
        r = {
            **row,
            "route": "eval",
            "endpoint": "/v1/chat/completions",
            "prefix_caching": True,
            "prefill_tok_s": row["input_tokens"] / (raw["mean_ttft_ms"] / 1000),
            "decode_tok_s_user": 1000 / raw["mean_tpot_ms"],
            "ttft_s": raw["mean_ttft_ms"] / 1000,
            "step_ms": raw["mean_tpot_ms"],
            "aggregate_output_tok_s": raw["output_throughput"],
            "planned_requests": row["requests"],
            "requests": raw["num_prompts"],
            "completed": raw["completed"],
            "duration_s": raw["duration"],
            "raw": str(rawpath.relative_to(BASE)),
            "sampling_modes": modes,
            "server_identity": str((directory / f"perf-b{b}-server.json").relative_to(BASE)),
            "phase_accounting": str((directory / "roofline.json").relative_to(BASE)),
            "prefill_flops_pct_host_wall": 100
            * roof["prefill"]["flops"]
            / roof["prefill"]["seconds"]
            / roof["prefill"]["peak_flops_per_second"],
            "decode_dram_pct_host_wall": 100
            * roof["decode"]["dram_bytes"]
            / roof["decode"]["seconds"]
            / roof["decode"]["peak_dram_bytes_per_second"],
        }
        if row["id"] in carried:
            r["archived_source_bundle"] = carried[row["id"]]["original_bundle"]
            r["provenance_manifest"] = "setup/native_fallback.json"
        pred = dict(r["predicted"])
        pred["prefill_aggregate_original"] = {
            key: pred[key] for key in ("prefill_tok_s", "prefill_tok_s_low", "prefill_tok_s_high") if key in pred
        }
        for key in pred["prefill_aggregate_original"]:
            pred[key] /= row["concurrency"]
        pred[
            "prefill_unit_conversion"
        ] = "Original aggregate rate divided by request concurrency. Algebraic unit conversion, no measured-value fitting."
        r["predicted"] = pred
        decode_steps = [s for s in detail["steps"] if s["phase"] == "decode"]
        r["decode_phase_mean_step_ms"] = 1000 * detail["seconds"]["decode"] / len(decode_steps)
        r["modeled_decode_bytes_per_step"] = detail["work"]["decode"] / len(decode_steps)
        r["prefill_fraction_of_window"] = detail["seconds"]["prefill"] / detail["window_seconds"]
        if r["decode_tok_s_user"] < pred["decode_tok_s_user_low"] or r["prefill_tok_s"] < pred.get(
            "prefill_tok_s_low", pred["prefill_tok_s"]
        ):
            active = [s["work"]["active_tokens"] for s in detail["steps"] if s["phase"] == "decode"]
            r["cause"] = {
                "observed": f"Full-phase prefill {detail['seconds']['prefill']:.3f}s, decode {detail['seconds']['decode']:.3f}s; {modes}; average active decode rows {sum(active)/len(active):.2f}.",
                "interpretation": f"Prefill occupies {r['prefill_fraction_of_window']:.1%} of the observed window. Mean consecutive decode-phase interval is {r['decode_phase_mean_step_ms']:.2f}ms versus HTTP per-user TPOT {r['step_ms']:.2f}ms; interleaved prefill and partial-batch scheduling affect delivered token cadence. The prediction used short-context cadence and did not model this interference or the full host path. Longer contexts add the KV work enumerated in this row's accounting. This is host-phase attribution, not a device profile.",
                "prefill_prediction_units": "Original aggregate prefill predictions are preserved and divided by concurrency for the per-request comparison, using the original predicted TTFT formula.",
                "evidence": str((directory / roof["prefill"]["evidence"]).relative_to(BASE)),
            }
            if row["input_tokens"] > 128:
                r["cause"][
                    "prefill_path"
                ] = "tt/batched_prefill.py selects batched prefill only for zero-start prompts <=128 tokens. This row uses the unchanged canonical chunked path; the observed prefill intervals delay other requests' decode tokens at concurrency>1."
        if r.get("cause") and row["concurrency"] == 1:
            r["cause"]["interpretation"] = (
                f"At concurrency1, mean HTTP TTFT is {r['ttft_s'] * 1000:.2f}ms versus predicted {pred['ttft_s'] * 1000:.2f}ms; "
                f"the measured full-prefill interval averages {1000 * detail['seconds']['prefill'] / raw['completed']:.2f}ms/request. "
                "The length-proportional prediction omits fixed request handling, final-head/sampling, dispatch and HTTP costs; these costs are included but not individually isolated in the observed phase. "
                f"Mean decode-phase interval is {r['decode_phase_mean_step_ms']:.2f}ms versus delivered TPOT {r['step_ms']:.2f}ms. "
                f"The executed-shape model estimates {r['modeled_decode_bytes_per_step']:.0f}DRAM bytes/decode step; longer-context KV traffic is enumerated in the linked accounting. "
                "This is full-phase host-wall attribution, not a device profile or a concurrent-prefill interference claim."
            )
        (releases if row.get("release") else measured).append(r)
    for row in measured + releases:
        if not row.get("cause"):
            continue
        control_length = row["input_tokens"] if row["concurrency"] > 1 else 4096
        control = next(
            (
                r
                for r in measured
                if r["batch"] == row["batch"]
                and r["concurrency"] == 1
                and r["input_tokens"] == control_length
                and r["sampling"] == row["sampling"]
            ),
            None,
        )
        if control and control["id"] != row["id"]:
            current_identity = read(BASE / row["server_identity"])
            control_identity = read(BASE / control["server_identity"])
            equal_fields = [
                "model",
                "model_revision",
                "tokenizer_revision",
                "precision",
                "hardware",
                "max_num_seqs",
                "max_model_len",
                "prefix_caching",
                "source_commits",
            ]
            assert all(current_identity[key] == control_identity[key] for key in equal_fields)
            row["cause"]["measured_control"] = {
                "row": control["id"],
                "raw": control["raw"],
                "decode_tok_s_user": control["decode_tok_s_user"],
                "ttft_s": control["ttft_s"],
                "rate_ratio_to_control": row["decode_tok_s_user"] / control["decode_tok_s_user"],
                "modeled_decode_bytes_per_step": control["modeled_decode_bytes_per_step"],
                "server_identity": control["server_identity"],
                "comparison": "Matching configured batch, precision and sampling; server instances may differ (see linked identities). Concurrency1 control at matching length, or 4096-token control for context scaling.",
            }
    identity = read(RUN / "perf-b32-server.json")
    identity["vllm_installation"] = read(BASE / "setup/vllm_installation.json")
    identity["benchmark_harness"] = read(BASE / "setup/main_sources.json")
    save(BASE / "identity.json", identity)
    rates = {
        "schema_version": 1,
        "status": "in_progress" if args.draft else "completed",
        "server_config": identity,
        "server": identity,
        "validated_batches": [1, 8, 32],
        "profile_configurations": {
            str(b): read(RUN / f"perf-b{b}-server.json")
            for b in (1, 8, 32)
            if (RUN / f"perf-b{b}-server.json").exists()
        },
        "workload": plan["workload"],
        "rows": measured,
        "release_rows": releases,
        "release_source": read(BASE / "release_benchmark_matrix.json")["source"],
        "missing_rows": missing,
        "sampling_policies": {
            "gpqa": {"temperature": 1, "top_p": 0.95, "top_k": 20, "seed": 42},
            "swe": {"temperature": 1, "top_p": 0.95, "top_k": 20, "seed": None},
        },
        "profile_identities": [
            str(p.relative_to(BASE)) for p in RUN.rglob("perf-b*-server.json") if "grid" not in p.parts
        ],
        "measurement_note": "Rates are host HTTP token timings on main chat route; GPQA seeded and SWE unseeded rows remain distinct. Stage12 must use its matching policy. Per-user prefill rate=input/mean TTFT includes scheduler waiting; host-wall rooflines are separately named.",
        "traffic_pattern": "Infinite-rate burst with bounded concurrency. Native headline uses max(8,3*batch) requests; other grid rows use max(2,concurrency). Warmup is one concurrency wave. GPQA/SWE headline comparisons therefore differ in population and are not a seed-only experiment.",
        "policy_rate_files": {policy: f"serving_rates-{policy}.json" for policy in ("gpqa", "swe")},
    }
    save(BASE / "serving_rates.json", rates)
    for policy in rates["sampling_policies"]:
        subset = dict(
            rates,
            selected_sampling_policy=policy,
            rows=[r for r in measured if r["sampling"] == policy],
            missing_rows=[r for r in missing if r["sampling"] == policy],
        )
        save(BASE / f"serving_rates-{policy}.json", subset)
    report = [
        "# Serving benchmark" + (" (draft; in progress)" if args.draft else ""),
        "",
        "| Row | Slots | Requests in flight | Input/output | Sampling | Prefill tok/s | Decode tok/s/user | TTFT s | Step ms |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in measured + releases:
        report.append(
            f"| {r['id']} | {r['batch']} | {r['concurrency']} | {r['input_tokens']}/{r['output_tokens']} | {r['sampling']} | {r['prefill_tok_s']:.2f} | {r['decode_tok_s_user']:.2f} | {r['ttft_s']:.3f} | {r['step_ms']:.3f} |"
        )
    report += [
        "",
        "Release matrix: "
        + rates["release_source"]
        + ". Only its five text rows apply to this text-only run. No implementation or serving config was borrowed.",
        "",
        "## Predictions and full-phase accounting",
        "",
        "| Row | Predicted decode range | Measured decode | Predicted prefill range | Measured prefill | Prefill host-wall FLOP % | Decode host-wall DRAM % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in measured + releases:
        p = r["predicted"]
        report.append(
            f"| {r['id']} | {p['decode_tok_s_user_low']:.2f}–{p['decode_tok_s_user_high']:.2f} | {r['decode_tok_s_user']:.2f} | {p.get('prefill_tok_s_low', p['prefill_tok_s']):.2f}–{p.get('prefill_tok_s_high', p['prefill_tok_s']):.2f} | {r['prefill_tok_s']:.2f} | {r['prefill_flops_pct_host_wall']:.3f} | {r['decode_dram_pct_host_wall']:.3f} |"
        )
    report += [
        "",
        "Headline sampling-mode step counts (all grid rows retain their own counts and per-step evidence):",
        "",
        "| Batch | Host-compatible prefill | Device prefill | Device decode |",
        "|---|---:|---:|---:|",
    ]
    for r in measured:
        if r.get("headline"):
            modes = r["sampling_modes"]
            report.append(
                f"| {r['batch']} | {modes.get('prefill:host_compatibility', 0)} | {modes.get('prefill:device', 0)} | {modes.get('decode:device', 0)} |"
            )
    report += [
        "",
        "Predictions were recorded in ../plan.json before the first server launch. Raw per-request ITL, E2E, percentiles, completion counts, throughput and duration remain beside each row. Each row in ../serving_rates.json links its raw timings, accounting and any prediction miss.",
        "The original prefill estimate describes aggregate capacity. For comparison with input/mean per-request TTFT, its rate and bounds are divided by concurrency, consistent with the original serialized-prefill prediction basis. Original aggregate estimates remain in predicted.prefill_aggregate_original; this unit conversion uses no measured values. Decode predictions already have per-user units. Row causes quantify prefill interference and measured decode intervals without claiming device-level attribution.",
        "For stage12 explain_hours.py, pass --rates ../serving_rates-gpqa.json for seed42 traffic or ../serving_rates-swe.json for unseeded SWE traffic. These are exact policy-filtered views of serving_rates.json; the unmodified interpolator does not select a sampling policy itself. Do not interpolate across both policies together.",
        rates["traffic_pattern"],
        "",
        "Phase time is the nonoverlapping wall interval between consecutive adapter-observed actual completions. Plugin host sampling, scheduling, output construction and idle gaps after an adapter return fall in the next interval. The first interval starts at the first measured request; the final client tail belongs to the last phase. This is operator decision 4 attribution, not separately observed plugin timing. FLOPs and DRAM estimates use executed shapes, precision and TP4; both numerators and denominators cover the same measured requests. No profiler or added device synchronization. Device-time utilization is unavailable.",
        "",
        "## Named gaps",
        "",
    ]
    report += [f"- {r['id']}: {r['reason']}" for r in missing] or ["None."]
    report += [
        "",
        "Model: full60-layer generated autoport, pinned model/tokenizer842da3794eaa0b77d5f08bae87a17459d91ff475, head_bfp4, P300x2/TP4, full context262144, prefix caching enabled. Per-profile identities retain actual launch argv, source hashes, module and revision evidence. Each invocation uses a unique cache salt and distinct dataset seed, matching main benchmark cache isolation; production prefix caching stays enabled. Server defaults remain temperature1/top_p0.95/top_k32. Benchmarks send eval top_k20, with seed42 for GPQA and no seed for SWE. All performance requests ignore EOS.",
        "",
        "## Anomaly and logging checks",
        "",
        "See ../anomaly_check.json, ../semantic_concerns.json, ../qualitative_verdict.json and ../../agentic_evals/logging_check.json. Detector flags measure mechanical degeneration, not answer correctness. Independent review identified acoustic-frequency, routing-feasibility and calibration-example inconsistencies in three generic replies; these semantic concerns are recorded for stage12 without attributing them to TT numerics. Stage11 makes no eval score or precision change.",
        "",
        "Four sampled replies naturally stopped after 1413, 1654, 1624 and 3929 generated tokens at a 32768-token cap, using served defaults temperature1/top_p0.95/top_k32. These checks cover 8620 generated tokens in total; they do not demonstrate sustained generation to the eval output caps or a ten-minute soak. Stage12 must assess longer-output behavior. All four complete response bodies match the server logs.",
        "",
        "## Timing and provenance",
        "",
        "Completed grid bundles carried from the budget-handoff attempt remain byte-exact. setup/native_fallback.json maps their discovery copies to archived originals and hashes; embedded old absolute server paths refer to that archived attempt, not the new server. The canonical native summary is a genuine fresh all-batch run, not merged results.",
        "The installed vLLM distribution is 0.31.0+empty and exposes no source Git SHA. identity.json leaves that commit null and includes the installed Python-source fingerprint and distribution metadata hashes; tt-metal, plugin and main harness commits are recorded. No unverified vLLM commit is inferred.",
    ]
    timing = read(BASE / "stage_timing.json")
    timing["elapsed_seconds_through_report"] = time.monotonic() - timing["started_monotonic"]
    timing["within_budget"] = timing["elapsed_seconds_through_report"] <= timing["budget_seconds"]
    save(RUN / "final_timing.json", timing)
    report.append(
        f"Elapsed through report: {timing['elapsed_seconds_through_report']:.1f}s / {timing['budget_seconds']}s. All server launches, warmup and measurements are inside this clock. Setup before the clock repaired host paths, verified main's source and ran16 host accounting tests. No earlier v3/v4 results were reused."
    )
    upstream = RUN / "runner-REPORT.md"
    if not upstream.exists() and (RUN / "REPORT.md").exists() and (RUN / "summary.json").exists():
        shutil.copy2(RUN / "REPORT.md", upstream)
    (RUN / "REPORT.md").write_text("\n".join(report) + "\n")
    packet_path = Path("/home/mvasiljevic/v5-agentic/logs/telemetry/packets/00156d66-a2a8-4738-a330-600ac7418ded.json")
    packet = read(packet_path.with_suffix(".template.json"))
    packet["selected_mesh"] = {"shape": [1, 4], "devices": [0, 1, 2, 3]}
    packet["performance"] = {
        "source": "vllm",
        "prefill_device_us": None,
        "decode_device_us": None,
        "prefill_flops_pct": None,
        "decode_dram_pct": None,
    }
    headline = next((r for r in measured if r.get("headline") and r["batch"] == 1), None)
    if headline:
        packet["performance"].update(
            ttft_ms=1000 * headline["ttft_s"],
            decode_tps=headline["decode_tok_s_user"],
            evidence=str(MODEL.relative_to(MODEL.parents[2]) / "doc/benchmark/run/perf-b1.json"),
        )
    packet["missing_reasons"] = {
        "device_metrics": "Live vLLM profiling prohibited; phase host-wall rooflines are not device-time metrics.",
        "accuracy": "Benchmarks only; evals belong to stage12.",
        "vllm_source_commit": "Installed 0.31.0+empty has no Git metadata; identity.json records source-tree and distribution hashes.",
    }
    if not headline:
        packet["missing_reasons"]["headline"] = "Required4096/128 B1 C1 workload has no complete current measurement."
    packet["notes"] = [
        "Fresh v5 benchmark; full60 layers, head_bfp4, context262144, main chat route.",
        "Headline sampling T1/top_p0.95/top_k20/seed42; unseeded SWE rows retained separately.",
        "Four sampled replies: no mechanical degeneration flags; three semantic concerns handed to stage12. Natural stops1413–3929 tokens do not validate full output caps.",
    ]
    if args.draft:
        packet["notes"].append("Attempt still in progress; this provisional packet will be replaced at closure.")
    save(packet_path, packet)


if __name__ == "__main__":
    main()
