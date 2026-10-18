"""Reconcile final same-run mesh timings and a conservative payload floor."""

import argparse
import os
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder")


def matrix(k, n, tile_bytes):
    return math.ceil(k / 32) * math.ceil(n / 32) * tile_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="final_")
    parser.add_argument("--qsa-prefix")
    parser.add_argument("--qsa-weight-bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--qsa-cache-bytes", type=int, choices=[512, 1024], default=1024)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    prefix = args.prefix
    runs = []
    for layer in (0, 1, 3):
        prefix = args.qsa_prefix if layer == 3 and args.qsa_prefix else args.prefix
        prof = EVIDENCE / "tracy" / f"{prefix}mesh_l{layer}"
        provenance = json.loads((prof / "provenance.json").read_text())
        timing = json.loads((EVIDENCE / f"{prefix}profile_mesh_l{layer}/timing.json").read_text())
        rows = list(csv.DictReader((prof / "decode_perf_report.csv").open()))
        payload = {
            "two_hyperconnection_read_up_weights": 2 * (matrix(2560, 352, 1088) + matrix(320, 2560, 1088)),
            "replicated_router_and_tp_shared_weights": matrix(2560, 544, 2048)
            + matrix(2560, 320, 2048)
            + matrix(160, 2560, 2048),
            "ideal_balanced_active_experts": 10 / 4 * 3 * matrix(2560, 640, 576),
            "attention_weights": matrix(
                2560, 4224 if layer == 3 else 4160, 576 if layer == 3 and args.qsa_weight_bits == 4 else 1088
            )
            + matrix(6144, 640, 1088),
        }
        if layer != 3:
            payload.update(recurrent_state_read_write=2 * 12 * 128 * 128 * 4, convolution_history=2 * 3 * 2560 * 2)
        if layer == 1:
            payload.update(ple_projection=matrix(2560, 3200, 1088), ple_lookup_tiles_ideal_balance=16 * 5 * 576 / 4)
        if layer == 3:
            # Capture advances once, then two warmups precede the timed window.
            first = timing["position_after_capture"] + 2
            average = first + (timing["decode_replays"] - 1) / 2
            payload.update(
                replicated_pooled_index_keys=65536 * 128 * 2, selected_local_kv=min(average + 1, 2051) * args.qsa_cache_bytes
            )
        categories = {}
        matmuls = {}
        for row in rows:
            code = row["OP Code"]
            category = (
                "collectives"
                if code.startswith(("AllGather", "ReduceScatter", "AllReduce"))
                else (
                    "experts"
                    if "Expert" in code
                    else (
                        "matmul"
                        if "Matmul" in code
                        else (
                            "data_movement"
                            if any(x in code.lower() for x in ("reshape", "tilize", "transpose", "copy", "slice"))
                            else "other_compute"
                        )
                    )
                )
            )
            value = row["Device Time"].replace(",", "").strip()
            if value not in ("", "-", "N/A"):
                categories[category] = categories.get(category, 0) + float(value) / timing["decode_replays"]
            if "Matmul" in code or "Expert" in code:
                keys = (
                    "OP Code",
                    "Math Fidelity",
                    "Input 0 Datatype",
                    "Input 1 Datatype",
                    "DRAM Sharded",
                    "Input 0 Memory",
                    "Inner Dim Block Size",
                    "Output Subblock H",
                    "Output Subblock W",
                )
                entry = {k: row.get(k) for k in keys}
                matmuls[json.dumps(entry, sort_keys=True)] = entry
        windows = provenance["windows"]
        item = dict(
            layer=layer,
            source_sha256=timing["source_sha256"],
            timing=f"{prefix}profile_mesh_l{layer}/timing.json",
            provenance=f"tracy/{prefix}mesh_l{layer}/provenance.json",
            prefill_host_ms=timing["median_prefill_host_ms"],
            prefill_device_ms=windows["prefill"]["per_pass_device_us"] / 1000,
            decode_host_ms=timing["median_traced_decode_ms"],
            decode_device_ms=windows["decode"]["per_pass_device_us"] / 1000,
            decode_device_plus_gap_ms=windows["decode"]["per_pass_device_plus_gap_us"] / 1000,
            decode_categories_us=categories,
            logical_payload_bytes_per_device=payload,
            ideal_balanced_dram_floor_ms=sum(payload.values()) / 512e6,
            worst_expert_imbalance_extra_floor_ms=7.5 * 3 * matrix(2560, 640, 576) / 512e6,
            matmul_rows=list(matmuls.values()),
            native_arguments=timing.get("native_expert_runtime_arguments"),
        )
        assert item["decode_device_ms"] <= item["decode_host_ms"] * 1.2, item
        runs.append(item)
    report = dict(
        bandwidth_source="tt-perf-report1.4.1 Blackhole ArchitectureSpec:512GB/s per ASIC",
        floor_scope="Logical stored-dtype weight/cache payload, ideal balanced EP2.5 of10 active experts/rank. "
        "Excludes dispatch, collective, intermediate traffic, repeated DRAM reads and allocator padding. "
        "Worst EP imbalance has10 selected experts on one rank; separate extra floor is given. "
        "This is a lower bound, not measured bandwidth or measured expert balance.",
        timing_scope="Same signposted profiler run; tt-perf-report1.4.1 selects maximum rank duration for non-collectives and mean rank duration for collective-name rows. Per-row gaps come from selected ranks. Sums of these mixed-rank durations/gaps are not one physical critical-path timeline. "
        "Ordinary uninstrumented latency remains the headline performance result.",
        runs=runs,
    )
    (EVIDENCE / f"{args.output_prefix or args.prefix}performance_accounting.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
