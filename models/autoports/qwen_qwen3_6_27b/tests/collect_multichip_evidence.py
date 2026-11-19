# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed host aggregation of the final multichip device evidence."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="delivery")
    parser.add_argument("--profile-prefix")
    args = parser.parse_args()
    prefix = args.prefix
    root = Path(__file__).resolve().parents[1]
    doc = root / "doc/multichip_decoder"
    source = hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    report = {"source_sha256": source, "gates": {}}

    def read(name, current=True):
        path = doc / (name + ".json")
        d = json.loads(path.read_text())
        assert d.get("pass", True) and d.get("status", "pass") not in ("fail", "failed", "running"), name
        if current:
            assert d["source_sha256"] == source, name
        return d

    def record(group, name, data):
        report["gates"].setdefault(group, []).append({"artifact": name + ".json", **data})

    for suite, shapes in {
        "boundary": [(2, 256), (2, 257), (32, 256), (32, 257)],
        "correctness": [(2, 1), (1, 31), (3, 513), (31, 33), (1, 1025)],
        "perf": [(1, 512), (32, 33), (32, 513)],
        "watcher": [(32, 33), (3, 513), (31, 33), (17, 33), (31, 31)],
    }.items():
        for layer in (0, 3):
            for batch, length in shapes:
                name = f"{prefix}_{suite}_sharded_l{layer}_b{batch}_s{length}"
                d = read(name)
                assert d["finite"] and d["trace"]["bit_equal_eager"]
                if suite != "perf":
                    assert d["trace"]["positive_control_rejected"]
                if suite in ("watcher", "boundary"):
                    assert d["provenance"]["watcher"] == "10"
                    assert (doc / (name + ".watcher.log")).exists()
                    if (batch, length) == (31, 31):
                        fixture = json.loads((doc / f"text_l{layer}_b31_s31.json").read_text())
                        assert d["reference_metadata"]["activation_file_sha256"] == fixture["sha256"]
                scores = [d["prefill"], d["decode"], *d["contribution"].values()]
                scores += [v for values in d["per_user_pcc"].values() for v in values]
                scores += [v for values in d["state"].values() for v in values]
                assert all(math.isfinite(v) and v >= 0.995 for v in scores), name
                data = {"prefill_pcc": d["prefill"], "decode_pcc": d["decode"], "trace_exact": True}
                if suite == "boundary":
                    rows, expected = {
                        (2, 256): (512, "separate_matmul"),
                        (2, 257): (514, "native_mmrs"),
                        (32, 256): (8192, "native_mmrs"),
                        (32, 257): (8224, "separate_matmul"),
                    }[(batch, length)]
                    observed = d["prefill_row_dispatch"]
                    assert len(observed) == 2
                    attention = "linear_attn.out_proj.weight" if layer == 0 else "self_attn.o_proj.weight"
                    assert {r["weight"] for r in observed} == {attention, "mlp.down_proj.weight"}
                    for row in observed:
                        assert row["flattened_rows"] == rows and row["path"] == expected
                        pieces = row["native_piece_shapes"]
                        if expected == "native_mmrs":
                            assert all(0 < shape[1] <= 128 for shape in pieces)
                            assert sum(shape[1] for shape in pieces) == ((rows + 31) // 32) * 32
                            assert row["output_shape"][-1] == 1280
                        else:
                            assert pieces == [] and row["output_shape"][-1] == 5120
                    assert (
                        d["dispatch_observer_sha256"]
                        == hashlib.sha256((root / "tests/run_multichip_boundaries.py").read_bytes()).hexdigest()
                    )
                    data["row_dispatch"] = observed
                record(suite, name, data)
    for layer in (0, 3):
        for batch, length in [(1, 128), (32, 33)]:
            name = f"{prefix}_stress_l{layer}_b{batch}_s{length}_tp4"
            d = read(name)
            assert d["stress"]["bit_equal"] and d["stress"]["runs"] == 2
            assert d["stress"]["steps"] == (128 if batch == 1 else 512)
            record("stress", name, d["stress"])
        for length in (262143, 262144):
            name = f"{prefix}_context_l{layer}_b1_s{length}_tp4"
            d = read(name)
            assert d["full_decoder_all_tokens"] and d["reserved_bytes_per_device"] == 20 * 1024**3
            if length == 262143:
                assert d["traced_decode_bit_equal"]
            record(
                "context",
                name,
                {key: d[key] for key in ("pcc", "contribution_pcc", "largest_sampled_allocation_bytes_per_device")},
            )
    for batch, length in [(2, 65), (32, 33)]:
        name = f"{prefix}_stack_l0_b{batch}_s{length}_tp4"
        d = read(name)
        assert d["layers"] == [0, 1, 2, 3, 4]
        assert d["no_boundary_conversion"] and d["trace_bit_equal"]
        assert min(d["pcc"].values()) >= 0.995
        record("stack", name, {"pcc": d["pcc"], "boundary_shape": d["boundary_shape"]})
    for batch in (1, 32):
        name = f"{prefix}_resources_b{batch}"
        d = read(name, current=False)
        assert d["status"] == "pass" and d["late_program_cache_growth"] == 0
        assert d["source_sha256"][str(root.relative_to(root.parents[2]) / "tt/multichip_decoder.py")] == source
        assert d["environment"]["TT_METAL_WATCHER"] == "10"
        assert d["prefill_batch"] == batch
        assert d["prefill_length"] == (65 if batch == 1 else 33)
        assert all(replay["pass"] for replay in d["replays"])
        assert d["main_trace_allocated_bytes_per_device"] <= 32 * 1024**2
        record(
            "resources",
            name,
            {"trace_bytes": d["main_trace_allocated_bytes_per_device"], "late_program_cache_growth": 0},
        )
    host = read("host_validation")
    assert host["pass"] and host["partition_cases"] == 630 and host["physical_page_remap_cases"] == 32
    report["host_contracts"] = "630 partitions,32 page remaps,14 norm ranks"
    audit = read("runtime_audit", current=False)
    assert audit["source_hashes"]["multichip_decoder"] == source and not audit["violations"]
    for layer in (0, 3):
        for batch, suffix in ((1, ""), (32, ""), (32, "_s513")):
            name = f"{args.profile_prefix or prefix + '_profile'}_l{layer}_b{batch}{suffix}"
            read(name)
            perf = doc / "perf" / name
            for filename in (
                "decode_perf_report.csv",
                "prefill_perf_report.csv",
                "decode_perf_report.txt",
                "prefill_perf_report.txt",
                "metrics.json",
                "accounting.json",
                "ops.csv.gz",
            ):
                assert (perf / filename).is_file(), perf / filename
            metrics = json.loads((perf / "metrics.json").read_text())
            raw = gzip.decompress((perf / "ops.csv.gz").read_bytes())
            assert hashlib.sha256(raw).hexdigest() == metrics["raw_ops_sha256"]
            rows = list(csv.DictReader(io.StringIO(raw.decode())))
            ranks = {row["DEVICE ID"] for row in rows if row["DEVICE ID"]}
            assert ranks == {"0", "1", "2", "3"}
            mm = [row for row in rows if row["OP CODE"] == "MatmulDeviceOperation"]
            dtypes = {row["INPUT_1_DATATYPE"] for row in mm}
            fidelities = {row["MATH FIDELITY"] for row in mm}
            assert dtypes == {"BFLOAT8_B"}
            assert fidelities == ({"LoFi", "HiFi2"} if layer == 0 else {"LoFi"})
            command = json.loads((doc / (name + ".command.json")).read_text())
            assert command["environment"].get("TT_METAL_PROFILER_MID_RUN_DUMP", "0") == "0"
            log = (doc / (name + ".log")).read_text()
            assert "TT_FATAL" not in log and "End marker found" not in log
            assert "markers were dropped" not in log
            for phase in ("prefill", "decode"):
                assert metrics[phase]["device_ops"] > 0 and metrics[phase]["device_kernel_sum_us"] > 0
            record(
                "profile",
                name,
                {
                    "report_directory": str(perf.relative_to(doc)),
                    "rank_ids": sorted(ranks),
                    "matmul_weight_dtypes": sorted(dtypes),
                    "matmul_fidelities": sorted(fidelities),
                    "raw_ops_sha256": metrics["raw_ops_sha256"],
                },
            )
    for suffix in ("qkv_watcher", "compaction_watcher"):
        name = f"{prefix}_{suffix}"
        d = read(name, current=False)
        assert d["status"] == "pass" and d["source_sha256"]["tt/multichip_decoder.py"] == source
        assert d["environment"]["TT_METAL_WATCHER"] == "10"
        assert (doc / (name + ".watcher.log")).is_file()
        record("local_kernels", name, {"status": "pass", "watcher": "10"})
    pytest_log = (doc / f"{prefix}_pytest.log").read_text()
    assert "4 passed" in pytest_log and "FAILED" not in pytest_log
    report["pytest"] = "4 passed"
    precision = json.loads((doc / "pcc31_control_summary.json").read_text())
    assert precision["source_sha256"] == source
    assert precision["separate_mmrs_decode_scores_identical"] and precision["separate_mmrs_state_scores_identical"]
    assert min(precision["hifi4_min_per_user_pcc"].values()) >= 0.995
    assert all(min(d["min_per_user_pcc"].values()) >= 0.995 for d in precision["recorded_controls"].values())
    report["controlled_synthetic_precision_sensitivity"] = {
        "artifact": "pcc31_control_summary.json",
        "failing_synthetic_decode_pcc": precision["synthetic_failure"]["pcc"],
        "acceptance": "Unchanged0.995 bar on recorded target activations; matched HiFi4 and separate-MMRS controls preserved.",
    }
    report["pass"] = True
    (doc / "gate_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({name: len(entries) for name, entries in report["gates"].items()})


if __name__ == "__main__":
    main()
