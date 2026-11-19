# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the audited final default, preserving timing and batch contracts."""

import csv
import json
import math
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tests.stage5_timing_evidence import (
    audit_timing,
    load_timing_map,
    validate_pair,
)

D = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    manifest = json.loads((D / "final_manifest.json").read_text())
    audit = json.loads((D / "final_audit.json").read_text())
    assert audit["status"] == "passed" and audit["source_sha256"] == manifest["source_sha256"]
    timing_map = load_timing_map()
    timing_audit = audit_timing(timing_map)

    def timing_path(path):
        return timing_map[path]

    def timing_data(path):
        return json.loads((D / timing_path(path)).read_text())

    rows = []
    functional = []
    for entry in manifest["artifacts"]:
        selected_path = timing_map.get(entry["path"], entry["path"])
        data = json.loads((D / selected_path).read_text())
        category = entry["category"]
        if category == "prefill_performance":
            old_path = (
                entry["path"]
                .replace(manifest["collection"], "stage4")
                .replace("large_boundaries", "large_prefill_perf")
            )
            before = timing_data(old_path)
            assert (
                data["command"][data["command"].index("--lengths") + 1]
                == before["command"][before["command"].index("--lengths") + 1]
            )
            assert data["steps"] == before["steps"]
            for new in data["records"]:
                if not new.get("label", "").startswith("timing_"):
                    continue
                old = next(r for r in before["records"] if r.get("label") == new["label"])
                rows.append(
                    {
                        "workload": f"layer{data['layer']}_{new['label']}_prefill",
                        "before_ms": old["prefill_ms"],
                        "after_ms": new["prefill_ms"],
                        "before_artifact": timing_path(old_path),
                        "after_artifact": selected_path,
                    }
                )
        if category == "stress":
            assert data["steps"] >= 128 and data["trace"]
            timing = next(r for r in data["records"] if r.get("label") == "timing_128")
            weights = 59047936 if data["layer"] == 0 else 55705600
            state_bytes = 1048576 if data["layer"] == 0 else 2 * 160 * 256 * 17 // 16
            floor_ms = (weights + state_bytes) / 512e9 * 1000
            fraction = floor_ms / timing["decode_ms"]
            assert fraction >= 0.10, (entry["path"], fraction)
            functional.append(
                {
                    "layer": data["layer"],
                    "artifact": selected_path,
                    "batch": 1,
                    "input_tokens": 128,
                    "validated_output_tokens": data["steps"],
                    "benchmark_decode_position": 128,
                    "bandwidth_floor_ms": floor_ms,
                    "traced_warmed_decode_ms": timing["decode_ms"],
                    "fraction_of_conservative_roofline": fraction,
                    "minimum_required_fraction": 0.10,
                    "passed": True,
                }
            )
        if category == "performance" and "layer" in data:
            before = timing_data(f"baseline/perf_{data['layer']}.json")
            old = next(r for r in before["records"] if r.get("label") == "timing_1024")
            new = next(r for r in data["records"] if r.get("label") == "timing_1024")
            for phase in ["prefill", "decode"]:
                rows.append(
                    {
                        "workload": f"layer{data['layer']}_S1024_{phase}",
                        "before_ms": old[phase + "_ms"],
                        "after_ms": new[phase + "_ms"],
                        "before_artifact": timing_path(f"baseline/perf_{data['layer']}.json"),
                        "after_artifact": selected_path,
                    }
                )
        elif category == "batch_matrix" and data["batch"] in (1, 32):
            old_path = f"packed_io_stage4_{data['batch']}.json"
            old = timing_data(old_path)
            assert all(data[k] == old[k] for k in ("batch", "prefix_length", "steps", "io_contract"))
            rows.append(
                {
                    "workload": f"packed_stack_B{data['batch']}_short_decode",
                    "before_ms": old["decode_stack_ms"],
                    "after_ms": data["decode_stack_ms"],
                    "before_artifact": timing_path(old_path),
                    "after_artifact": selected_path,
                }
            )
        elif category == "extended" and data.get("batch") == 32 and data.get("benchmark"):
            old_path = {
                65535: "context64k_stage4_32.json",
                131071: "context128k_stage4_answer1024_32.json",
            }.get(data.get("prefix_length"))
            if old_path is not None:
                old = timing_data(old_path)
                assert all(data[k] == old[k] for k in ("batch", "prefix_length", "steps", "capacity", "io_contract"))
                rows.append(
                    {
                        "workload": f"packed_stack_B32_prefix{data['prefix_length']}_answer{data['steps']}_decode",
                        "before_ms": old["decode_stack_ms"],
                        "after_ms": data["decode_stack_ms"],
                        "before_artifact": timing_path(old_path),
                        "after_artifact": selected_path,
                    }
                )
        elif category == "long_answer":
            old_path = "answer1024_stage4_32.json"
            old = timing_data(old_path)
            assert all(data[k] == old[k] for k in ("batch", "prefix_length", "steps", "io_contract"))
            rows.append(
                {
                    "workload": "packed_stack_B32_4Kprefix_1024answer_decode",
                    "before_ms": old["decode_stack_ms"],
                    "after_ms": data["decode_stack_ms"],
                    "before_artifact": timing_path(old_path),
                    "after_artifact": selected_path,
                }
            )
            positions = [p + data["steps"] - 1 for p in data["starts"]]
            weights = 59047936 + 55705600
            recurrent = data["batch"] * 1048576
            kv = sum(2 * math.ceil((p + 1) / 32) * 32 * 256 * 17 // 16 for p in positions)
            floor_ms = (weights + recurrent + kv) / 512e9 * 1000
            roofline = {
                "target_artifact": selected_path,
                "implementation_sha256": data["source_sha256"],
                "mesh": [1, 4],
                "batch": data["batch"],
                "decode_positions": positions,
                "per_device_bandwidth_bytes_per_second": 512e9,
                "projection_storage_bytes_per_device": weights,
                "recurrent_read_write_bytes_per_device": recurrent,
                "useful_tile_covered_kv_read_bytes_per_device": kv,
                "bandwidth_floor_ms": floor_ms,
                "measured_traced_warmed_ms": data["decode_stack_ms"],
                "fraction_of_conservative_roofline": floor_ms / data["decode_stack_ms"],
                "latency_at_70_percent_ms": floor_ms / 0.70,
                "note": "Weights read once per batch. Serial linear+full layer floors. Omits history, other activations, norms, KV writes, metadata and CCL; does not inflate avoidable traffic. Architectural512GB/s convention from installed tt-perf-report; not measured sustainable bandwidth.",
            }
            (D / "final_roofline.json").write_text(json.dumps(roofline, indent=2) + "\n")
    assert {row["layer"] for row in functional} == {0, 3}
    (D / "final_functional_bar.json").write_text(json.dumps(functional, indent=2) + "\n")
    for row in rows:
        validate_pair(row["before_artifact"], row["after_artifact"])
        row["speedup"] = row["before_ms"] / row["after_ms"]
        row["latency_reduction_percent"] = 100 * (1 - row["after_ms"] / row["before_ms"])
    timing_audit["matched_timing_rows"] = len(rows)
    timing_audit["matched_pairs"] = sorted({(row["before_artifact"], row["after_artifact"]) for row in rows})
    (D / "final_timing_audit.json").write_text(json.dumps(timing_audit, indent=2) + "\n")
    with (D / "final_performance.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (D / "final_performance.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("FINAL_PERFORMANCE_PASS", len(rows))


if __name__ == "__main__":
    main()
