# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only reconciliation of the final fused-decoder evidence."""
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/fused_decoder"


def read(name):
    return json.loads((DOC / name).read_text())


def write(name, value):
    (DOC / name).write_text(json.dumps(value, indent=2) + "\n")


def median(result):
    return statistics.median(result["trace_replay_wall_ms"])


def main():
    source_hash = hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest()
    required = []
    for layer in (0, 3):
        required += [
            f"delivered_{kind}_{layer}.json"
            for kind in ("contract", "batch32", "trace", "mixed", "watcher_contract", "watcher_batch32", "profile")
        ]
        required += [f"delivered_context_{layer}_{length}.json" for length in (262143, 262144)]
        required += [f"delivered_batch2_{layer}_{length}.json" for length in (511, 513)]
        required += [f"delivered_timing_{layer}_{length}.json" for length in (128, 511, 512, 513, 4097)]
        required += [f"delivered_batch32_perf_{layer}.json"]
    for name in required:
        result = read(name)
        provenance = result["provenance"]
        assert provenance["decoder_source_sha256"] == source_hash, name
        assert provenance["decoder_implementation"] == "fused" and provenance["fusion_candidate"] is None, name
    driver = (DOC / "validate_final_driver.log").read_text()
    assert "wait_for FAIL" not in driver and "RUNTIME_AUDIT PASS" in driver
    assert "4 passed" in (DOC / "delivered_pytest.log").read_text()

    rows = []
    for layer in (0, 3):
        for length in (128, 511, 512, 513, 4097):
            baseline = read(f"timing_functional_{layer}_{length}.json")
            final = read(f"delivered_timing_{layer}_{length}.json")
            row = {"layer": layer, "batch": 1, "prefill_length": length}
            for label, result in (("functional", baseline), ("fused", final)):
                row.update(
                    {
                        label + "_prefill_wall_ms": result["warmed_prefill_wall_ms"],
                        label + "_decode_median_ms": median(result),
                        label + "_prefill_pcc": result["prefill_pcc"],
                        label + "_decode_pcc": result["decode_pcc"],
                    }
                )
                assert result["prefill_pcc"] >= 0.995 and result["decode_pcc"] >= 0.995
                assert result["repeated_bit_equal"] and result["prefill_repeated_bit_equal"]
            row["decode_speedup"] = row["functional_decode_median_ms"] / row["fused_decode_median_ms"]
            assert row["decode_speedup"] > 1
            rows.append(row)
    write("performance_comparison.json", rows)
    with (DOC / "performance_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    batch_rows = []
    for layer, baseline_name in ((0, "final_functional_batch32_0"), (3, "functional_batch32_perf_3")):
        baseline = read(baseline_name + ".json")
        final = read(f"delivered_batch32_perf_{layer}.json")
        assert final["prefill_pcc"] >= 0.995 and final["decode_pcc"] >= 0.995
        assert final["repeated_bit_equal"] and final["prefill_repeated_bit_equal"]
        batch_rows.append(
            {
                "layer": layer,
                "batch": 32,
                "prefill_length": 33,
                "baseline_artifact": baseline_name + ".json",
                "final_artifact": f"delivered_batch32_perf_{layer}.json",
                "functional_prefill_wall_ms": baseline["warmed_prefill_wall_ms"],
                "fused_prefill_wall_ms": final["warmed_prefill_wall_ms"],
                "functional_decode_median_ms": median(baseline),
                "fused_decode_median_ms": median(final),
                "decode_speedup": median(baseline) / median(final),
            }
        )
    write("batch32_performance.json", batch_rows)

    roofs, profiles = {}, {}
    for layer, weight_bytes, state_bytes in ((0, 766771200, 6291456), (3, 744488960, 2101248)):
        tag = f"delivered_profile_{layer}"
        metrics = read(f"perf/{tag}/metrics.json")
        result = read(tag + ".json")
        timing = read(f"delivered_timing_{layer}_512.json")
        floor_ms = (weight_bytes + state_bytes) / 512e9 * 1000
        profiles[tag] = {}
        raw = gzip.decompress((DOC / f"perf/{tag}/ops.csv.gz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == metrics["raw_ops_sha256"]
        for mode in ("prefill", "decode"):
            with (DOC / f"perf/{tag}/{mode}_perf_report.csv").open() as stream:
                ops = [row for row in csv.DictReader(stream) if row["ID"].isdigit()]
            conversions = [
                row["OP Code"] for row in ops if any(s in row["OP Code"].lower() for s in ("tilize", "reshard"))
            ]
            assert not conversions, conversions
            matmuls = [row["Math Fidelity"] for row in ops if "MatmulDeviceOperation" in row["OP Code"]]
            assert matmuls and all(value == "HiFi4 BF16 x BF16 => BF16" for value in matmuls), matmuls
            profiles[tag][mode] = {
                "ops": len(ops),
                "tilize_untilize_reshard_ops": conversions,
                "memory_movement_ops": [
                    row["OP Code"]
                    for row in ops
                    if any(value in row["OP Code"].lower() for value in ("sharded", "reshard", "tilize"))
                ],
                "matmul_policy": sorted(set(matmuls)),
            }
        roofs[str(layer)] = {
            "workload": "B1, prefill512, decode context513, one layer on physical chip0",
            "active_bf16_dense_weight_bytes": weight_bytes,
            "mandatory_state_or_kv_bytes": state_bytes,
            "bandwidth_bytes_per_second": 512e9,
            "roofline_floor_ms": floor_ms,
            "unprofiled_decode_median_ms": median(timing),
            "roofline_fraction_unprofiled": floor_ms / median(timing),
            "same_profile_host_decode_median_ms": median(result),
            "same_profile_device_kernel_ms": metrics["decode"]["device_kernel_sum_us"] / 1000,
            "same_profile_device_kernel_plus_gap_ms": metrics["decode"]["kernel_plus_gap_us"] / 1000,
            "same_profile_roofline_fraction": floor_ms / median(result),
            "profile_artifact": tag + ".json",
            "timing_artifact": f"delivered_timing_{layer}_512.json",
        }
        assert roofs[str(layer)]["roofline_fraction_unprofiled"] >= 0.70
    write("roofline.json", roofs)
    write("profiler_audit.json", profiles)
    write(
        "acceptance.json",
        {
            "status": "device_and_host_gates_passed_review_pending",
            "decoder_source_sha256": source_hash,
            "pcc_threshold": 0.995,
            "context": 262144,
            "required_final_artifacts": required,
            "driver_log": "validate_final_driver.log",
            "pytest_log": "delivered_pytest.log",
            "roofline": "roofline.json",
            "performance": "performance_comparison.json",
            "profile_audit": "profiler_audit.json",
            "watcher_audit": "delivered_watcher_audit.json",
            "review": None,
            "local_commit": None,
        },
    )
    print("FUSED_EVIDENCE_PASS", len(required), "final-source results")


if __name__ == "__main__":
    main()
