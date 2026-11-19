# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only compact profiler evidence and eligible candidate ranking.

No hardware, no inferred timings, no conflation of clean and profiled runs.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/optimized_decoder"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def byte_model(layer, position):
    """Optimistic logical bytes; excludes padding, rereads and intermediates.

    BFP8 tiles store 1024 mantissas +64 shared exponent bytes, or17/16 bytes
    per useful weight. Linear attention preserves BF16 projection exceptions.
    Router and norm weights stay FP32, KV BF16. Eight UNIQUE selected experts.
    """
    linear = layer != 3
    bfp8 = 17 / 16
    weights = {
        "attention_projection": 2048 * (12352 if linear else 9216) * (2 if linear else bfp8),
        "attention_output": 4096 * 2048 * (2 if linear else bfp8),
        "eight_expert_gate_up_bfp8": 8 * 2048 * 1024 * bfp8,
        "eight_expert_down_bfp8": 8 * 512 * 2048 * bfp8,
        "shared_gate_up_scalar_bfp8": 2048 * 1025 * bfp8,
        "shared_down_bfp8": 512 * 2048 * bfp8,
        "router_fp32": 2048 * 256 * 4,
        "layer_norm_weights_fp32": 2 * 2048 * 4,
    }
    if linear:
        weights.update(convolution_taps_fp32=4 * 8192 * 4, selected_decay_fp32=32 * 4, output_norm_bf16=128 * 2)
        cache = dict(recurrent_read_write_fp32=2 * 32 * 128 * 128 * 4, conv_read_write_bf16=2 * 3 * 8192 * 2)
    else:
        weights["qk_norm_weights_fp32"] = 2 * 256 * 4
        cache = dict(kv_reads_bf16=(position + 1) * 2 * 2 * 256 * 2, kv_write_bf16=2 * 2 * 256 * 2)
    total = sum(weights.values()) + sum(cache.values())
    return dict(
        weight_bytes=weights, cache_bytes=cache, logical_bytes=total, bandwidth_gb_s=512, roofline_ms=total / 512e6
    )


def window(rows, phase):
    start, end = f"PERF_{phase.upper()}", f"PERF_{phase.upper()}_END"
    starts = [i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == start]
    ends = [i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == end]
    assert len(starts) == len(ends) == 1 and starts[0] < ends[0], (phase, starts, ends)
    selected = [r for r in rows[starts[0] + 1 : ends[0]] if r["OP TYPE"] == "tt_dnn_device"]
    assert selected and all(r["DEVICE KERNEL DURATION [ns]"] for r in selected), "Missing device durations"
    assert {r["DEVICE ID"] for r in selected} == {"0"}, "Single-device accounting only"
    return selected


def summarize_profile(doc, layer):
    directory = doc / f"tracy/layer_{layer}"
    result = read(directory / "correctness.json")
    assert result["decoder"] == "OptimizedDecoder" and result["deterministic"]
    assert min(result["prefill_pcc"], result["decode_traced_pcc"]) >= 0.995
    with (directory / "ops.csv").open() as f:
        rows = list(csv.DictReader(f))
    phases = {}
    for phase in ("prefill", "decode"):
        selected = window(rows, phase)
        repetitions = 1 if phase == "prefill" else result["profile_decode_replays"]
        if phase == "decode":
            sessions = defaultdict(list)
            for row in selected:
                sessions[row["METAL TRACE REPLAY SESSION ID"]].append(row)
            assert "" not in sessions and len(sessions) == repetitions, "Missing/duplicate trace sessions"
            signatures = [
                [(r["OP CODE"], r["INPUT_0_DATATYPE"], r["INPUT_1_DATATYPE"]) for r in s] for s in sessions.values()
            ]
            assert all(s == signatures[0] for s in signatures), "Incomplete replay op coverage"
        groups = defaultdict(lambda: [0, 0.0])
        keys = [
            "OP CODE",
            "INPUT_0_DATATYPE",
            "INPUT_1_DATATYPE",
            "MATH FIDELITY",
            "INPUT_1_Y_PAD[LOGICAL]",
            "INPUT_1_X_PAD[LOGICAL]",
            "CORE COUNT",
        ]
        for row in selected:
            key = tuple(row[k] for k in keys)
            groups[key][0] += 1
            groups[key][1] += float(row["DEVICE KERNEL DURATION [ns]"]) / 1000
        compact = [
            dict(zip(keys, key), count_per_pass=n / repetitions, kernel_us_per_pass=us / repetitions)
            for key, (n, us) in groups.items()
        ]
        compact.sort(key=lambda r: r["kernel_us_per_pass"], reverse=True)
        output = directory / f"{phase}_op_groups.csv"
        with output.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=[*keys, "count_per_pass", "kernel_us_per_pass"])
            writer.writeheader()
            writer.writerows(compact)
        report_path = directory / f"{phase}_perf_report.csv"
        with report_path.open() as f:
            report = list(csv.DictReader(f))
        advice = sorted({r.get("Advice", "") for r in report} - {""})
        phases[phase] = dict(
            repetitions=repetitions,
            device_ops_per_pass=len(selected) / repetitions,
            device_kernel_sum_ms=sum(r["kernel_us_per_pass"] for r in compact) / 1000,
            compact_csv=str(output.relative_to(doc)),
            advice=advice,
            report_csv=str(report_path.relative_to(doc)),
        )
    roof = byte_model(layer, result["length"])
    same_run = result.get("profile_decode_host_median_ms")
    clean_path = doc / f"final_{layer}.json"
    clean = read(clean_path) if clean_path.exists() else {}
    return dict(
        layer=layer,
        phases=phases,
        roofline=roof,
        decode_ms_per_token_device=phases["decode"]["device_kernel_sum_ms"],
        decode_ms_per_token_profiled_host=same_run,
        profiled_host_minus_kernel_ms=None if same_run is None else same_run - phases["decode"]["device_kernel_sum_ms"],
        decode_ms_per_token_clean_host=clean.get("decode_traced_warmed_ms"),
        same_run_accounting_complete=same_run is not None,
        clean_timing_evidence=str(clean_path.relative_to(doc)) if clean else None,
    )


def rank_candidates(doc):
    ranked = []
    for search in sorted(doc.rglob("search.json")):
        for row in read(search):
            command = row.get("command", [])
            if "--output" not in command:
                continue
            output = Path(command[command.index("--output") + 1])
            if not output.exists():
                output = search.parent / output.name
            reasons = []
            artifact = read(output) if output.exists() else {}
            if row.get("exit_code") != 0 or not artifact:
                reasons.append("failed_or_missing_result")
            if min(artifact.get("prefill_pcc", 0), artifact.get("decode_traced_pcc", 0)) < 0.995 or not artifact.get(
                "deterministic"
            ):
                reasons.append("correctness_not_proven")
            if "bfp4" in json.dumps(artifact.get("precision_policy", row.get("policy", {}))).lower():
                reasons.append("BFP4_deferred_to_stage8")
            provenance_path = Path(str(output) + ".provenance.json")
            provenance = read(provenance_path) if provenance_path.exists() else {}
            env = provenance.get("env", {})
            if (
                not provenance
                or provenance.get("regime") != "serial"
                or any(
                    env.get(k) not in (None, "", "0")
                    for k in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING")
                )
            ):
                reasons.append("not_proven_clean_serial_timing")
            if (
                artifact.get("length") != 256
                or artifact.get("context") != 512
                or artifact.get("activation_source") not in (None, "seeded random BF16, std0.05")
            ):
                reasons.append("not_matched_S256_context512_seed42_regime")
            ranked.append(
                dict(
                    search=str(search.relative_to(doc)),
                    candidate=row["candidate"],
                    layer=row["layer"],
                    decode_ms=artifact.get("decode_traced_warmed_ms"),
                    prefill_ms=artifact.get("prefill_warmed_ms"),
                    eligible=not reasons,
                    exclusion_reasons=reasons,
                    result=str(output),
                )
            )
    ranked.sort(key=lambda r: (r["layer"], not r["eligible"], r["decode_ms"] or float("inf")))
    return ranked


def functional_sequence(doc, layer):
    path = doc / f"functional_{layer}.json"
    if not path.exists():
        return dict(layer=layer, status="missing")
    result = read(path)
    assert result["decoder"] == "OptimizedDecoder" and result["gate_pass"]
    assert result["prompt_length"] == result["decode_length"] == 128
    assert result["tracking"] is False and result["repeated_bit_equal"]
    samples = result["decode_trace_samples_ms"]
    assert len(samples) == 128 * result["repeats"] and all(t > 0 for t in samples)
    # Exact average theoretical bytes across the128 evolving cache positions.
    bound = sum(byte_model(layer, position)["roofline_ms"] for position in range(128, 256)) / 128
    measured = sum(samples) / len(samples)
    fraction = bound / measured
    return dict(
        layer=layer,
        status="pass" if fraction >= 0.1 else "fail",
        prompt_len=128,
        gen_len=128,
        batch=1,
        repeats=result["repeats"],
        roofline_ms_per_token_estimate=bound,
        decode_ms_per_token_mean=measured,
        roofline_fraction=fraction,
        evidence=str(path.relative_to(doc)),
        timing_boundary=result["timing_boundary"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--ranking-only", action="store_true")
    args = parser.parse_args()
    ranking = rank_candidates(args.doc)
    write(args.doc / "candidate_ranking.json", ranking)
    if args.ranking_only:
        return
    measurements = [summarize_profile(args.doc, layer) for layer in (0, 3)]
    write(
        args.doc / "perf_summary.json",
        dict(
            measurements=measurements,
            functional_128_128=[functional_sequence(args.doc, layer) for layer in (0, 3)],
            limitations=[
                "Decoder-layer evidence only; no full-model throughput claim",
                "Kernel sum excludes device dispatch gaps; profiler host time excludes explicit drains",
                "Roofline is optimistic logical bandwidth bound, not measured traffic or compute bound",
                "Clean host latency is a separate run and is never labeled same-run",
                "FUNCTIONAL128/128 requires explicit timed sequence evidence; correctness stress alone has no timings",
            ],
        ),
    )


if __name__ == "__main__":
    main()
