# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify raw receipts and reproduce the resident-reader experiment summary."""

import gzip
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def summarize():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    artifacts = {}
    for item in manifest["artifacts"]:
        data = read(ROOT / item["stored"])
        assert len(data) == item["bytes"], item["original"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["original"]
        assert item["original"] not in artifacts
        artifacts[item["original"]] = data

    def load(name):
        return json.loads(artifacts[name])

    def junit(name, expected):
        suites = ET.fromstring(artifacts[name]).findall(".//testsuite")
        assert sum(int(s.get("tests", 0)) for s in suites) == expected
        assert all(not int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))

    rows, runs = [], []
    for version, modes, arms, unit_count in (("v1", 3, 3, 10), ("v2", 6, 5, 22)):
        control = f"gdn-resident-padding-control-{version}"
        result = f"gdn-resident-padding-{version}"
        queue = load(f"{result}/queue.json")
        receipt = load(f"{result}/padding.json")
        launch = load(f"{control}/launch-unit.json")
        observed = load(f"{control}/collection-unit.json")
        assert queue["state"] == "completed" and queue["cleanup_completed"]
        assert receipt["state"] == "completed" and receipt["passed"] and receipt["cleanup_completed"]
        assert receipt["device_ids"] == [0, 4, 12, 8]
        assert observed["MainPID"] == "0" and observed["ActiveState"] in ("inactive", "failed")
        assert receipt["full_model_qualified"] is False and receipt["promoted_to_serving"] is False
        junit(f"{control}/unit.xml", unit_count)
        junit(f"{result}/hardware.xml", 1)
        source_manifest = load(f"{control}/source-manifest.json")
        for relative, digest in source_manifest.items():
            original = f"gdn-resident-padding-source-{version}/{relative}"
            if original in artifacts:
                assert hashlib.sha256(artifacts[original]).hexdigest() == digest
                if version == "v2":
                    local = ROOT.parents[1] / relative.removeprefix("models/demos/qwen38_27b_qb2/")
                    assert hashlib.sha256(local.read_bytes()).hexdigest() == digest, relative
        for mode, digest in receipt["reader_sha256"].items():
            assert hashlib.sha256(artifacts[f"{result}/reader-{mode}.cpp"]).hexdigest() == digest
        assert len(receipt["reader_sha256"]) == modes
        assert [c["batch"] for c in receipt["cases"]] == [16, 32]
        for case in receipt["cases"]:
            assert case["passed"] and case["readonly_inputs"] and case["stable_addresses"]
            assert case["cycles"] == 1 and case["steps"] == 64
            assert [(c["phase"], c["steps"]) for c in case["checks"]] == [("cancellation", 1), ("trace", 64)]
            for check in case["checks"]:
                assert len(check["ranks"]) == 4 * modes
                for rank in check["ranks"]:
                    assert rank["state_exact"] and rank["output_exact"]
                    assert rank["state_dense"]["passed"] and rank["output_dense"]["passed"]
            timing = case["timing"]["arms"]
            assert len(timing) == arms
            for arm in timing:
                assert arm["steps"] == 2592 and len(arm["call_us"]) == 5
                assert all(t > 0 for t in arm["call_us"])
                assert arm["median_us"] == statistics.median(arm["call_us"])
                assert arm["state"] == timing[0]["state"] and arm["output"] == timing[0]["output"]
            before, after = timing[0]["median_us"], timing[-1]["median_us"]
            baseline = (before + after) / 2
            drift = abs(after - before) / baseline
            assert drift < 0.03
            for candidate in timing[1:-1]:
                saved_ms = (baseline - candidate["median_us"]) * 48 / 1000
                rows.append(
                    dict(
                        version=version,
                        batch=case["batch"],
                        variant=candidate["variant"],
                        before_us=before,
                        after_us=after,
                        control_us=baseline,
                        candidate_us=candidate["median_us"],
                        latency_change_pct=(candidate["median_us"] / baseline - 1) * 100,
                        control_drift_pct=drift * 100,
                        projected_48_layer_saved_ms=saved_ms,
                        projected_b16_tsu=(1000 / (47.882258 - saved_ms)) if case["batch"] == 16 else None,
                        selected=False,
                    )
                )
        log = artifacts[f"{control}/run.log"].decode()
        assert "SAFE_PYTEST_RESULT: PASS" in log
        assert "SAFE_PYTEST: Resetting device" not in log
        runs.append(
            dict(
                version=version,
                invocation_id=launch["InvocationID"],
                cpu_tests=unit_count,
                controller_seconds=queue["finished_at"] - queue["started_at"],
                trace_allocation_warnings=[
                    line for line in log.splitlines() if "allocat" in line.lower() and "warn" in line.lower()
                ],
                terminal_observation=observed,
            )
        )
    return dict(
        verified_raw_files=len(artifacts),
        native_build_and_exact_component_checks_passed=True,
        changing_input_steps=64,
        timing_steps_per_arm=2592,
        changing_input_4096_qualified=False,
        model_measured_tsu=20.884562,
        full_model_qualified=False,
        promoted_to_serving=False,
        runs=runs,
        rows=rows,
    )


if __name__ == "__main__":
    result = summarize()
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
