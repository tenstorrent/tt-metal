# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only audit of selected, frozen B early-guard evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HEADER = "experiments/sdpa-l2/compute-sprint-v2/compensated/identity_early/compute_streaming.hpp"
HELPER = "experiments/sdpa-l2/compute-sprint-v2/compensated/identity/identity.hpp"
TRANSFERS = ("B-early-transfer-01", "B-early-odd-heldout-01", "B-early-short-heldout-01")
TIMINGS = ("B-early-paired-final", "B-early-distinct32k-normal-final",
           "B-early-distinct32k-grow-01", "B-early-distinct8k-transition-01",
           "B-early-distinct256k-normal-01")


def check_sources(record):
    for source in (HEADER, HELPER):
        assert record["source_sha256"][source] == hashlib.sha256((ROOT/source).read_bytes()).hexdigest(), source


def main():
    count = 0
    for label in TRANSFERS:
        record = json.loads((HERE/(label+".json")).read_text())
        check_sources(record)
        grouped = {}
        for case in record["results"]:
            assert case["bitwise_equal"] and case["unequal_elements"] == 0
            assert case["max_abs_change"] == 0 and case["trace_replays"] == 2
            group = grouped.setdefault(case["distribution"], [])
            group.append(case)
            count += 1
        for group in grouped.values():
            assert {case["mode"] for case in group} == {"canonical", "disabled", "identity_early"}
            assert len({case["output_sha256"] for case in group}) == 1
            descriptor_sets = []
            for case in group:
                descriptor_sets.append([
                    (descriptor["compile_time_args"],
                     [(key, value) for key, value in (descriptor["defines"] or [])
                      if key != "SDPA_REVIEW_IDENTITY_EARLY"])
                    for descriptor in case["descriptors"]])
            assert descriptor_sets[0] == descriptor_sets[1] == descriptor_sets[2]
    for label in TIMINGS:
        record = json.loads((HERE/(label+".json")).read_text())
        check_sources(record)
        baseline = record["results"]["canonical"]
        for name, case in record["results"].items():
            assert case["bitwise_equal_canonical"] and case["unequal_elements"] == 0
            assert case["trace_equal"] is True
            assert case["output_sha256"] == baseline["output_sha256"]
            for field in ("numeric_defines", "cb_counts", "kv_slots", "math_fidelity",
                          "fp32_dest_acc_en", "math_approx_mode", "dst_full_sync_en"):
                assert case[field] == baseline[field], (label, name, field)
            assert case["kv_slots"] == 2 and case["fp32_dest_acc_en"] is False
            assert len(case["times_ms"]) == 10
            assert abs(case["tflops_per_core"] - record["useful_flops"]/(case["median_ms"]*1e9)) < 1e-12
            count += 1
    print(f"PASS {count} records: 48 fullchip raw-bit/two-replay records and 15 final timing/raw-trace records.")
    print("Frozen implementation/helper hashes and numerical/CB contracts checked; not a compiler/firmware closure.")


if __name__ == "__main__":
    main()
