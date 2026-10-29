# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks for rejecting incomplete/failed optimization evidence."""

import csv
import json

from scripts.nvfp4_harness import EvaluationError, compare, profile_metrics, validate_junit


def test_nvfp4_harness_rejects_masked_pytest_failure(tmp_path, expect_error):
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="numerics"><failure message="bad ULP"/></testcase></testsuite></testsuites>'
    )
    with expect_error(EvaluationError, "Failed"):
        validate_junit(junit)


def test_nvfp4_harness_requires_expected_skips(tmp_path, expect_error):
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="ok"/><testcase name="excluded"><skipped/></testcase></testsuite></testsuites>'
    )
    with expect_error(EvaluationError, "unexpectedly skipped"):
        validate_junit(junit)
    validate_junit(junit, expected_skips=1)
    junit.write_text(
        '<testsuites><testsuite><testcase name="all skipped"><skipped/></testcase></testsuite></testsuites>'
    )
    with expect_error(EvaluationError, "empty"):
        validate_junit(junit, expected_skips=1)


def test_nvfp4_harness_normalizes_output_tiles_and_clock(tmp_path, expect_error):
    report = tmp_path / "profiler/reports/run"
    report.mkdir(parents=True)
    (report / "profile_log_device.csv").write_text("ARCH: blackhole, CHIP_FREQ[MHz]: 1350\n")
    workload = {"calls": 3, "warmup_calls": 1, "cores": 2, "maximum_output_tiles_per_core": 4}
    (tmp_path / "workload.json").write_text(json.dumps(workload))
    with (report / "ops_perf_results_run.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["OP CODE", "CORE COUNT", "DEVICE KERNEL DURATION [ns]", "DEVICE TRISC1 KERNEL DURATION [ns]"],
        )
        writer.writeheader()
        for duration in [9999, 400, 400]:
            writer.writerow(
                {
                    "OP CODE": "Nvfp4DeviceOperation",
                    "CORE COUNT": 2,
                    "DEVICE KERNEL DURATION [ns]": duration,
                    "DEVICE TRISC1 KERNEL DURATION [ns]": duration,
                }
            )
    metrics = profile_metrics(tmp_path, repetitions=128)
    assert metrics["pipeline_cycles_per_output_tile"] == 135, "Incorrect output-tile or clock normalization"
    assert metrics["math_cycles_per_output_tile"] == 135 / 128, "Math repetitions were not amortized"
    workload["calls"] = 4
    (tmp_path / "workload.json").write_text(json.dumps(workload))
    with expect_error(EvaluationError, "Expected 4"):
        profile_metrics(tmp_path)


def test_nvfp4_harness_requires_matching_evaluator(tmp_path, expect_error):
    baseline = {
        "status": "passed",
        "evaluator_hashes": {"gate": "one"},
        "performance": {"workload": {}, "clock_mhz": 1350, "pipeline_cycles_per_output_tile": 100, "pipeline_mad": 0},
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["evaluator_hashes"]["gate"] = "two"
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    with expect_error(EvaluationError, "different trusted evaluators"):
        compare(baseline_path, candidate_path)
    candidate["evaluator_hashes"] = baseline["evaluator_hashes"]
    candidate["performance"]["pipeline_cycles_per_output_tile"] = 90
    candidate_path.write_text(json.dumps(candidate))
    result = compare(baseline_path, candidate_path)
    assert result["screening_improvement"], "A 10% improvement with zero observed MAD should pass screening"
    assert result["improvement_percent"] == 10, "Incorrect speedup normalization"
