# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce rejected GDN scheduling and placement results from sealed receipts."""

import gzip
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize():
    artifacts = {}
    for item in json.loads((ROOT / "manifest.json").read_text())["artifacts"]:
        path = ROOT / item["stored"]
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["original"]
        assert item["original"] not in artifacts
        artifacts[item["original"]] = data

    def load(name):
        return json.loads(artifacts[name])

    def junit(name, tests):
        suites = ET.fromstring(artifacts[name]).findall(".//testsuite")
        for key, expected in (("tests", tests), ("errors", 0), ("failures", 0), ("skipped", 0)):
            assert sum(int(s.get(key, 0)) for s in suites) == expected, (name, key)

    experiments, rows = [], []
    for key, receipt_name, unit_count in (
        ("gdn-resident-schedule", "schedule", 12),
        ("gdn-bank-placement", "placement", 17),
    ):
        result, control, source = f"{key}-v1", f"{key}-control-v1", f"{key}-source-v1"
        receipt = load(f"{result}/{receipt_name}.json")
        queue = load(f"{result}/queue.json")
        launch = load(f"{control}/launch-unit.json")
        terminal = load(f"{control}/collection-unit.json")
        assert launch["ActiveState"] == "active" and int(launch["MainPID"]) > 0
        assert "MainPID=0" in terminal["stdout"] and "ActiveState=inactive" in terminal["stdout"]
        assert receipt["device_ids"] == [0, 4, 12, 8]
        assert queue["state"] == receipt["state"] == "completed"
        assert queue["cpu_tests_passed"] == unit_count
        assert queue["cleanup_completed"] and receipt["cleanup_completed"] and receipt["passed"]
        assert receipt["trace_allocation_tracking"]
        assert not receipt["full_model_qualified"] and not receipt["promoted_to_serving"]
        log = artifacts[f"{control}/run.log"].decode()
        assert "SAFE_PYTEST_RESULT: PASS" in log
        assert "SAFE_PYTEST: Resetting device" not in log
        assert "still alive before trace replay" not in log
        junit(f"{control}/unit.xml", unit_count)
        junit(f"{result}/hardware.xml", 1)
        source_hashes = load(f"{control}/source-manifest.json")
        for name, data in artifacts.items():
            if name.startswith(source + "/"):
                relative = name.removeprefix(source + "/")
                assert hashlib.sha256(data).hexdigest() == source_hashes[relative]
                local = ROOT.parents[1] / relative.removeprefix("models/demos/qwen38_27b_qb2/")
                assert local.read_bytes() == data, relative
        hashes = receipt.get("compute_sha256", receipt.get("kernel_sha256"))
        for name, digest in hashes.items():
            filename = f"compute-{name}.cpp" if receipt_name == "schedule" else name
            assert hashlib.sha256(artifacts[f"{result}/{filename}"]).hexdigest() == digest
        assert [c["batch"] for c in receipt["cases"]] == [16, 32]
        for case in receipt["cases"]:
            assert case["passed"] and case["readonly_inputs"] and case["stable_addresses"]
            assert case["steps"] == 64 and case["cycles"] == 1
            assert len(case["addresses"]) == len(set(case["addresses"]))
            assert [(c["phase"], c["steps"]) for c in case["checks"]] == [("cancellation", 1), ("trace", 64)]
            for check in case["checks"]:
                assert len(check["ranks"]) == 16
                for rank in check["ranks"]:
                    assert rank["state_exact"] and rank["output_exact"]
                    assert rank["state_dense"]["passed"] and rank["output_dense"]["passed"]
            timing = case["timing"]
            arms = timing["arms"]
            assert len(arms) == 5 and arms[0]["variant"] == "before" and arms[-1]["variant"] == "after"
            for arm in arms:
                assert arm["steps"] == 2592 and len(arm["call_us"]) == 5
                assert all(math.isfinite(t) and t > 0 for t in arm["call_us"])
                assert statistics.median(arm["call_us"]) == arm["median_us"]
                assert arm["state"] == arms[0]["state"] and arm["output"] == arms[0]["output"]
            base = (arms[0]["median_us"] + arms[-1]["median_us"]) / 2
            assert base == timing["control_us"]
            assert timing["timing_stable"] and timing["control_drift_fraction"] < 0.03
            for arm, comparison in zip(arms[1:-1], timing["comparisons"]):
                saved = base - arm["median_us"]
                assert saved == comparison["saved_us"]
                ms = saved * 48 / 1000
                assert ms == comparison["projected_48_layer_saving_ms"]
                rows.append(
                    dict(
                        experiment=receipt_name,
                        batch=case["batch"],
                        variant=arm["variant"],
                        control_us=base,
                        candidate_us=arm["median_us"],
                        latency_change_pct=(arm["median_us"] / base - 1) * 100,
                        control_drift_pct=timing["control_drift_fraction"] * 100,
                        projected_48_layer_saving_ms=ms,
                        projected_b16_tsu_gain=1000 / (47.882258 - ms) - 1000 / 47.882258
                        if case["batch"] == 16
                        else None,
                    )
                )
        experiments.append(
            dict(
                experiment=receipt_name,
                cpu_tests=unit_count,
                invocation_id=launch["InvocationID"],
                controller_seconds=queue["finished_at"] - queue["started_at"],
                terminal=terminal,
                scope=receipt["scope"],
            )
        )
    return dict(
        verified_raw_files=len(artifacts),
        experiments=experiments,
        rows=rows,
        model_measured_tsu=20.884562,
        selected_variants=[],
        full_model_qualified=False,
        promoted_to_serving=False,
        long_horizon_4096_tested=False,
    )


if __name__ == "__main__":
    summary = summarize()
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
