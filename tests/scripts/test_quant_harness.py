# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks for the multi-format quantized-decoder harness."""

import csv
import json

from scripts.quant_harness import (
    FORMATS,
    EvaluationError,
    check_accuracy_report,
    compare,
    math_overlay,
    profile_metrics,
)
from scripts.quant_loop_stats import summarize

MXFP8 = FORMATS["mxfp8"]


def test_formats_are_self_consistent():
    for name, spec in FORMATS.items():
        assert spec.name == name
        assert all(test in spec.trusted_files for test in spec.accuracy_tests + spec.screen_tests + spec.perf_tests)
        assert spec.kernels.parts[-1] == "kernels"


def test_mxfp8_math_overlay_repeats_marked_sweep(tmp_path):
    candidate = tmp_path / "candidate"
    compute = candidate / MXFP8.kernels / "compute"
    compute.mkdir(parents=True)
    source = """void kernel_main() {
    for (uint32_t tile = 0; tile < count; ++tile) {
        setup();
        // MXFP8_MATH_SWEEP_BEGIN
        decode(0);
        decode(1);
        // MXFP8_MATH_SWEEP_END
        finish();
    }
}
"""
    (compute / "dequantize.cpp").write_text(source)
    (compute / "mxfp8_sfpu.hpp").write_text("// shared candidate header\n")
    overlay = math_overlay(MXFP8, candidate, tmp_path / "results", 128)
    instrumented = (overlay / MXFP8.kernels / "compute/dequantize.cpp").read_text()
    assert instrumented.index("setup()") < instrumented.index("repeat < 128") < instrumented.index("decode(0)")
    assert instrumented.index("decode(1)") < instrumented.index("MXFP8_MATH_SWEEP_END") < instrumented.index("finish")
    assert instrumented.count("decode(") == 2, "The sweep must be repeated by a loop, not duplicated"
    assert (compute / "dequantize.cpp").read_text() == source
    assert (overlay / MXFP8.kernels / "compute/mxfp8_sfpu.hpp").is_symlink()


def test_mxfp8_math_overlay_requires_markers(tmp_path, expect_error):
    candidate = tmp_path / "candidate"
    compute = candidate / MXFP8.kernels / "compute"
    compute.mkdir(parents=True)
    (compute / "dequantize.cpp").write_text("void kernel_main() {}\n")
    with expect_error(EvaluationError, "MXFP8_MATH_SWEEP_BEGIN"):
        math_overlay(MXFP8, candidate, tmp_path / "results", 128)


def test_mxfp8_accuracy_report_must_match_every_kernel_file(expect_error):
    hashes = {"kernels/compute/dequantize.cpp": "a", "kernels/dataflow/reader.cpp": "b"}
    report = {"accuracy_gate": {"passed": True}, "kernel_hashes": dict(hashes)}
    check_accuracy_report(MXFP8, report, hashes)
    report["kernel_hashes"]["kernels/dataflow/reader.cpp"] = "stale"
    with expect_error(EvaluationError, "different kernel sources"):
        check_accuracy_report(MXFP8, report, hashes)
    with expect_error(EvaluationError, "Numerical gate"):
        check_accuracy_report(MXFP8, {"accuracy_gate": {"passed": False}, "kernel_hashes": hashes}, hashes)


def test_compare_rejects_different_formats(tmp_path, expect_error):
    result = {
        "status": "passed",
        "format": "mxfp8",
        "evaluator_hashes": {"gate": "one"},
        "performance": {"workload": {}, "clock_mhz": 1350, "pipeline_cycles_per_output_tile": 100, "pipeline_mad": 0},
    }
    legacy = {key: value for key, value in result.items() if key != "format"}  # Pre-refactor results are NVFP4.
    (tmp_path / "a.json").write_text(json.dumps(result))
    (tmp_path / "b.json").write_text(json.dumps(legacy))
    with expect_error(EvaluationError, "different formats"):
        compare(tmp_path / "a.json", tmp_path / "b.json")


def test_mxfp8_profile_filters_its_op_code(tmp_path):
    report = tmp_path / "profiler/reports/run"
    report.mkdir(parents=True)
    (report / "profile_log_device.csv").write_text("ARCH: blackhole, CHIP_FREQ[MHz]: 1000\n")
    (tmp_path / "workload.json").write_text(
        json.dumps(dict(calls=2, warmup_calls=1, cores=1, maximum_output_tiles_per_core=2))
    )
    fields = ["OP CODE", "CORE COUNT", "DEVICE KERNEL DURATION [ns]", "DEVICE TRISC1 KERNEL DURATION [ns]"]
    with (report / "ops_perf_results_run.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for code, duration in [
            ("Mxfp8DeviceOperation", 9999),
            ("Nvfp4DeviceOperation", 1),
            ("Mxfp8DeviceOperation", 400),
        ]:
            writer.writerow(dict(zip(fields, [code, 1, duration, duration])))
    metrics = profile_metrics(MXFP8, tmp_path)
    assert metrics["pipeline_samples"] == [200.0]
    assert metrics["op_code"] == "Mxfp8DeviceOperation"


def test_loop_stats_reads_each_formats_profile(tmp_path):
    profile = tmp_path / "profile.csv"
    fields = ["OP CODE", "HOST DURATION [ns]", "DEVICE KERNEL DURATION [ns]"]
    with profile.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for code in ["Mxfp8DeviceOperation"] * 3 + ["Nvfp4DeviceOperation"]:
            writer.writerow(dict(zip(fields, [code, 1e9, 1e9])))
    output = tmp_path / "runs/run-0"
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps(
            dict(
                status="passed",
                format="mxfp8",
                op_code="Mxfp8DeviceOperation",
                started_unix=1,
                finished_unix=2,
                kernel_hashes={},
                performance=dict(
                    op_code="Mxfp8DeviceOperation", pipeline_cycles_per_output_tile=10, profile_csv=str(profile)
                ),
            )
        )
    )
    (output / "accuracy.json").write_text(json.dumps(dict(valid_values_per_case=10, cases=[{}, {}])))
    result = summarize(tmp_path / "runs")
    assert result["candidates"][0]["format"] == "mxfp8"
    assert result["candidates"][0]["profiled_kernel_seconds"] == 3
    assert result["gate_datum_comparisons"] == 20
