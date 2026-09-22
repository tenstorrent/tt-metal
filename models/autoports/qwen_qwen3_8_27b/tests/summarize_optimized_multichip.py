# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize recorded stage candidates without importing a device library."""

import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    rows, projections, geometry = [], [], []
    for path in sorted(DOC.glob("*.json")):
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or "decode_ms" not in report:
            continue
        environment_path = path.with_suffix(".environment.json")
        environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
        rows.append(
            {
                "artifact": path.name,
                "evidence_kind": (
                    "diagnostic"
                    if path.stem.startswith("gap_counter_")
                    else "watcher"
                    if path.stem.startswith("watcher_")
                    else "profile"
                    if "profile" in path.stem
                    else "timing"
                ),
                "pass_through_thread_pool": environment.get("TT_MESH_PASS_THROUGH_THREAD_POOL"),
                "process_exit_status": (
                    int(path.with_suffix(".exit_status").read_text())
                    if path.with_suffix(".exit_status").exists()
                    else None
                ),
                **{key: report.get(key) for key in ("source_sha256", "layer", "batch", "length", "stack", "baseline")},
                "prefill_ms": report.get("prefill_ms"),
                "traced_prefill_ms": report.get("traced_prefill_ms"),
                "decode_ms": report["decode_ms"],
                "queued_decode_ms": report.get("queued_decode_ms"),
                "min_pcc": min(report.get("pcc", {"missing": 0}).values()),
                "prefill_pcc": report.get("pcc", {}).get("prefill"),
                "decode_pcc": report.get("pcc", {}).get("decode"),
                "policy": json.dumps(report.get("policy", {}), sort_keys=True),
            }
        )
        for name, result in report.get("projection_bench", {}).items():
            projections.append({"artifact": path.name, "projection": name, **result})
            if "fidelity" not in result or "fp32_dest_acc_en" not in result:
                continue
            k, n = result["weight_shape"]
            cores, readers = result["cores"], result["readers"]
            reader_n = ((n + 31) // 32 + 8 * readers - 1) // (8 * readers)
            limit = 4 if result.get("fp32_dest_acc_en", True) else 8
            sub_w = next(v for v in range(limit, 0, -1) if reader_n % v == 0)
            iterations = reader_n // sub_w
            for width in range(sub_w + 1, limit + 1):
                if (reader_n + width - 1) // width < iterations:
                    iterations = (reader_n + width - 1) // width
                    sub_w = width
            geometry.append(
                {
                    "artifact": path.name,
                    "projection": name,
                    "local_k": k,
                    "local_n": n,
                    "readers": readers,
                    "logical_cores": cores,
                    "input_shard_k_tiles": (k + 32 * cores - 1) // (32 * cores),
                    "in0_block_w": result["in0_block_w"],
                    "per_core_M": 1,
                    "reader_N_tiles": reader_n,
                    "compute_N_tiles": sub_w * iterations,
                    "derived_output_subblock": f"1x{sub_w}",
                    "weight_dtype": result["weight_dtype"],
                    "fidelity": result["fidelity"],
                    "fp32_acc": result.get("fp32_dest_acc_en", True),
                    "projection_trace_ms": result["ms"],
                    "whole_decoder_trace_ms": report["decode_ms"],
                    "min_pcc": min(report.get("pcc", {"missing": 0}).values()),
                    "selected_default": path.stem.startswith("after_review_"),
                    "input_memory": result["input_memory"],
                    "output_memory": result["output_memory"],
                }
            )
    for name, data in (
        ("candidate_summary", rows),
        ("projection_summary", projections),
        ("matmul_geometry_search", geometry),
    ):
        (DOC / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
        if data:
            with (DOC / (name + ".csv")).open("w") as out:
                writer = csv.DictWriter(
                    out, fieldnames=list(dict.fromkeys(key for row in data for key in row)), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(data)


if __name__ == "__main__":
    main()
