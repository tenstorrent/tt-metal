# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit final default artifacts without importing a device library."""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def main():
    source_hash = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    records = []

    def check(name, *, prefill_only=False, watcher=False, stress=False, prefill_trace=False):
        path = DOC / (name + ".json")
        result = json.loads(path.read_text())
        assert path.with_suffix(".exit_status").read_text().strip() == "0", name
        assert result["source_sha256"] == source_hash, name
        assert result["mesh_shape"] == [1, 4] and result["baseline"] is False, name
        assert result["policy"] == {} and not result.get("cache_pcc_diagnostic_only"), name
        env = json.loads((DOC / (name + ".environment.json")).read_text())
        assert env["TT_MESH_PASS_THROUGH_THREAD_POOL"] == "1", name
        assert min(result["pcc"].values()) >= 0.995, name
        if not prefill_only:
            for key in ("trace_bitwise_equal", "changed_input_replay_bitwise_equal", "post_decode_state_bitwise_equal"):
                assert result[key], (name, key)
            assert result["runtime_fallback_audit"].startswith("passed:"), name
            assert min(v for values in result["per_user_pcc"].values() for v in values) >= 0.995, name
        if stress:
            assert result["queued_stress_iterations"] >= 100 and result["queued_stress_bitwise_equal"], name
        if prefill_trace:
            assert result["prefill_trace_bitwise_equal"] and result["changed_prefill_trace_bitwise_equal"], name
        if watcher:
            assert env["TT_METAL_WATCHER"] == "10" and not env["TT_METAL_WATCHER_DISABLE_ETH"], name
            assert not env["TT_METAL_DEVICE_PROFILER"], name
        records.append(
            {
                "artifact": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "min_pcc": min(result["pcc"].values()),
            }
        )

    cases = sorted(
        path.stem
        for path in DOC.glob("final_*.json")
        if re.fullmatch(r"final_l[03]_b\d+_s\d+_c[01]|final_stack_b\d+", path.stem)
    )
    assert len(cases) == 31, cases
    for name in cases:
        check(name)
    for name in ("after_review_l0", "after_review_l3", "after_review_stack"):
        check(name)
    for row in json.loads((DOC / "final_tail_timing_matrix.json").read_text()):
        check(row["name"])
    for name in ("final_stress_stack_b1", "final_stress_stack_b32"):
        check(name, stress=True)
    for name in ("watcher_linear_tail", "watcher_linear_batch32", "watcher_full_batch32", "watcher_stack_batch3"):
        check(name, watcher=True)
    check("watcher_review_prefill_trace_stack", watcher=True, prefill_trace=True)
    for row in json.loads((DOC / "review_trace_prefill_matrix.json").read_text()):
        check(row["name"], prefill_trace=True)
    for layer in (0, 3):
        for length in (262143, 262144):
            check(f"capacity_l{layer}_s{length}", prefill_only=length == 262144)
    check("capacity_stack_s262143")
    capacity = json.loads((DOC / "memory_capacity_plan.json").read_text())
    assert capacity["status"] == "validated" and capacity["source_sha256"] == source_hash
    assert capacity["context"] == 262144 and len(capacity["capacity_evidence"]) == 5
    assert "31 passed" in (DOC / "final_correctness_pytest.log").read_text()
    assert "18 passed" in (DOC / "native_final_mesh_tests.log").read_text()
    assert (DOC / "native_final_mesh_tests.exit_status").read_text().strip() == "0"
    assert "18 passed" in (DOC / "native_formatted_mesh_tests.log").read_text()
    assert (DOC / "native_formatted_mesh_tests.exit_status").read_text().strip() == "0"
    assert "18 passed" in (DOC / "native_lint_final.log").read_text()
    assert (DOC / "native_lint_final.exit_status").read_text().strip() == "0"
    for name in (
        "review_profile_l0",
        "review_profile_l3",
        "review_profile_stack",
        "review_profile_l0_s2049",
        "review_profile_l3_s2049",
        "review_profile_l3_s4094",
    ):
        check(name)
        for device in range(4):
            for phase in ("prefill", "decode"):
                for suffix in ("csv", "txt"):
                    path = DOC / "tracy" / name / f"device{device}_{phase}_perf_report.{suffix}"
                    assert path.exists() and path.stat().st_size > 0, path
    accounting = json.loads((DOC / "performance_accounting.json").read_text())
    assert len(accounting) == 6 and all(row["source_sha256"] == source_hash for row in accounting)
    boundary = json.loads((DOC / "inter_layer_profile_audit.json").read_text())
    assert len(boundary) == 4 and all(row["source_sha256"] == source_hash for row in boundary)
    casts = []
    for layer in (0, 3):
        for device in range(4):
            counts = []
            for prefix in ("final_profile", "review_profile"):
                path = DOC / "tracy" / f"{prefix}_l{layer}" / f"device{device}_prefill_perf_report.csv"
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                counts.append(
                    sum(
                        row["OP Code"] == "TypecastDeviceOperation"
                        and row["Input 0 Datatype"] == row["Output Datatype"] == "BFLOAT16"
                        for row in rows
                    )
                )
            assert counts == [4, 0], (layer, device, counts)
            casts.append(
                {
                    "layer": layer,
                    "device": device,
                    "before_identity_casts": counts[0],
                    "after_identity_casts": counts[1],
                }
            )
    (DOC / "review_cast_audit.json").write_text(json.dumps(casts, indent=2) + "\n")
    (DOC / "final_validation_index.json").write_text(
        json.dumps({"status": "passed", "model_source_sha256": source_hash, "default_artifacts": records}, indent=2)
        + "\n"
    )
    print(f"PASS: {len(records)} final default artifacts, TP4/PCC/trace/stress/watcher/context/profile gates")


if __name__ == "__main__":
    main()
