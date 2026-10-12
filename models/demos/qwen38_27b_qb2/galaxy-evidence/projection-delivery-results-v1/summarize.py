# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify sealed projection-delivery attempts and reproduce their conclusions."""

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
        assert len(data) == item["bytes"], item["original"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["original"]
        assert item["original"] not in artifacts
        artifacts[item["original"]] = data

    def load(name):
        return json.loads(artifacts[name])

    def junit(name, *, tests, failures=0):
        suites = ET.fromstring(artifacts[name]).findall(".//testsuite")
        for key, expected in (("tests", tests), ("failures", failures), ("errors", 0), ("skipped", 0)):
            assert sum(int(s.get(key, 0)) for s in suites) == expected, (name, key)

    attempts, rows, projections = [], [], []
    for version, expected_cases in (("v1", 0), ("v2", 16), ("v3", 0), ("v4", 24)):
        control = f"projection-delivery-control-{version}"
        result = f"projection-delivery-{version}"
        queue = load(f"{result}/queue.json")
        receipt = load(f"{result}/delivery.json")
        launch = load(f"{control}/launch-unit.json")
        observed = load(f"{control}/collection-unit.json")
        assert launch["ActiveState"] == "active" and int(launch["MainPID"]) > 0
        assert observed["MainPID"] == "0" and observed["ActiveState"] in ("inactive", "failed")
        assert receipt["device_ids"] == [0, 4, 12, 8]
        assert receipt["cleanup_completed"] and receipt["trace_allocation_tracking"]
        assert not receipt["full_model_qualified"] and not receipt["promoted_to_serving"]
        assert len(receipt["cases"]) == expected_cases
        junit(f"{control}/unit.xml", tests=13)
        manifest = load(f"{control}/source-manifest.json")
        for relative, digest in manifest.items():
            original = f"projection-delivery-source-{version}/{relative}"
            if original in artifacts:
                assert hashlib.sha256(artifacts[original]).hexdigest() == digest, original
                if version == "v4":
                    local = ROOT.parents[1] / relative.removeprefix("models/demos/qwen38_27b_qb2/")
                    assert hashlib.sha256(local.read_bytes()).hexdigest() == digest, relative
        log = artifacts[f"{control}/run.log"].decode()
        assert "SAFE_PYTEST: Resetting device" not in log
        attempt = dict(
            version=version,
            state=receipt["state"],
            invocation_id=launch["InvocationID"],
            cpu_tests=13,
            controller_seconds=queue["finished_at"] - queue["started_at"],
            cases=expected_cases,
            cleanup_completed=receipt["cleanup_completed"],
            terminal_observation=observed,
        )
        attempts.append(attempt)
        if not expected_cases:
            assert queue["state"] == receipt["state"] == "failed" and not receipt["passed"]
            junit(f"{result}/hardware.xml", tests=1, failures=1)
            expected_error = "still alive before trace replay" if version == "v1" else "Shape, slice"
            assert expected_error in receipt["detail"]
            attempt["failure"] = receipt["detail"]
            continue
        assert queue["state"] == receipt["state"] == "completed" and receipt["passed"]
        assert queue["cleanup_completed"] and queue["exact_component_match"]
        assert "SAFE_PYTEST_RESULT: PASS" in log
        assert "still alive before trace replay" not in log
        junit(f"{result}/hardware.xml", tests=1)
        cases = receipt["cases"]
        for case in cases:
            assert case["accuracy_passed"] and case["changed_input_trace_passed"] and case["exact_baseline"]
            assert case["trace_calls"] == 16 and len(case["samples_us"]) == 5
            assert all(math.isfinite(t) and t > 0 for t in case["samples_us"])
            assert len(case["output_sha256"]) == 4
            assert [c["input"] for c in case["accuracy"]] == [0, 1, 0]
            for check in case["accuracy"]:
                assert check["passed"] and len(check["baseline"]) == 4
                assert all(c["equal"] for c in check["baseline"])
            if version == "v4":
                expected = case["delivery"] if case["input_layout"] == "compact_l1" else "reshard"
                assert case["effective_delivery"] == expected
        comparisons = []
        for batch in (16, 32):
            layouts = (("output", "compact_l1"), ("down", "compact_l1"))
            if version == "v4":
                layouts += (("output", "public_dram"),)
            for role, layout in layouts:
                group = [c for c in cases if (c["batch"], c["role"], c["input_layout"]) == (batch, role, layout)]
                assert [c["label"] for c in group] == ["before", "native_input", "matmul_output", "after"]
                assert all(c["output_sha256"] == group[0]["output_sha256"] for c in group)
                before, after = group[0]["samples_us"], group[-1]["samples_us"]
                base = statistics.median(before + after)
                drift = abs(statistics.median(after) / statistics.median(before) - 1)
                assert drift <= 0.03
                for candidate in group[1:-1]:
                    current = statistics.median(candidate["samples_us"])
                    multiplicity = 48 if role == "output" else 64
                    comparison = dict(
                        version=version,
                        batch=batch,
                        role=role,
                        input_layout=layout,
                        delivery=candidate["delivery"],
                        control_us=base,
                        candidate_us=current,
                        latency_change_pct=(current / base - 1) * 100,
                        control_drift_pct=drift * 100,
                        projected_model_saving_ms=(base - current) * multiplicity / 1000,
                    )
                    comparisons.append(comparison)
                    rows.append(comparison)
        assert len(comparisons) == len(receipt["comparisons"])
        for measured, recorded in zip(comparisons, receipt["comparisons"]):
            assert recorded["comparison_qualified"] and recorded["controls_match"] and recorded["accuracy_passed"]
            for key in ("control_us", "candidate_us", "projected_model_saving_ms"):
                assert measured[key] == recorded[key]
        for delivery in ("native_input", "matmul_output"):
            saved = sum(
                r["projected_model_saving_ms"]
                for r in comparisons
                if r["batch"] == 16 and r["input_layout"] == "compact_l1" and r["delivery"] == delivery
            )
            projections.append(
                dict(
                    version=version,
                    delivery=delivery,
                    b16_projected_saved_ms=saved,
                    b16_projected_tsu=1000 / (47.882258 - saved),
                    b16_projected_tsu_gain=1000 / (47.882258 - saved) - 1000 / 47.882258,
                    full_model_measured=False,
                )
            )
    return dict(
        verified_raw_files=len(artifacts),
        attempts=attempts,
        rows=rows,
        projections=projections,
        model_measured_tsu=20.884562,
        default_delivery="reshard",
        full_model_qualified=False,
        promoted_to_serving=False,
        accuracy_scope="Actual layer-0 BFP8 weights; synthetic rank/user-distinct inputs and native TP4 reductions",
    )


if __name__ == "__main__":
    summary = summarize()
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("verified_raw_files", "projections")}, indent=2))
