# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.tests.bounded_profile import CASES, SCOPE, check_artifact_budget, collect
from models.demos.qwen38_27b_qb2.tests.layer_profile_report import DURATIONS, analyze


def fixture(root):
    length, batch = CASES[0]
    receipt = dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        scope=SCOPE,
        recurrence="native",
        prefill_calls=0,
        decode_calls=3,
        output_hashes=["same"] * 3,
        device_ids=[0, 1, 2, 3],
        cells=[dict(input_tokens=length, batch=batch)],
    )
    prefix = f"P0_S{length}_B{batch}"
    rows = [{"OP TYPE": "signpost", "OP CODE": prefix + "_MODEL_BEGIN"}]
    for layer in (0, 3):
        stage = prefix + f"_L{layer}_decode_forward"
        rows.append({"OP TYPE": "signpost", "OP CODE": stage + "_BEGIN"})
        for device in receipt["device_ids"]:
            rows.append(
                {
                    "OP TYPE": "tt_dnn_device",
                    "OP CODE": "Op",
                    "DEVICE ID": str(device),
                    "GLOBAL CALL COUNT": str(layer),
                    **{name: "100" for name in DURATIONS.values()},
                }
            )
        rows.append({"OP TYPE": "signpost", "OP CODE": stage + "_END"})
    rows.append({"OP TYPE": "signpost", "OP CODE": prefix + "_MODEL_END"})

    def write():
        (root / "profile.json").write_text(json.dumps(receipt))
        (root / "hardware.xml").write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        trace = root / "tracy"
        trace.mkdir(exist_ok=True)
        (trace / "cpp_device_perf_report.csv").write_text("compact-timings")
        with (trace / "ops_perf_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["OP TYPE", "OP CODE", "DEVICE ID", "GLOBAL CALL COUNT", *DURATIONS.values()]
            )
            writer.writeheader()
            writer.writerows(rows)

    write()
    return receipt, rows, write


def test_bounded_capture_is_distinct_from_original_six_case_profile(tmp_path, expect_error):
    receipt, rows, _ = fixture(tmp_path)
    with expect_error(ValueError, "missing required"):
        analyze(rows, receipt)
    report = collect(tmp_path, *CASES[0], "native")
    assert not report["p0_gate_passed"]
    assert report["measurements_complete"] and report["synthetic_caches"]
    assert len(report["device_totals"]) == 4
    assert all(x["firmware_ns"] == 200 for x in report["device_totals"])
    assert str(CASES[0][0]) in (tmp_path / "analysis/report.md").read_text()


@pytest.mark.parametrize(
    "failure",
    [
        "prefill",
        "changed_output",
        "too_many_calls",
        "cleanup",
        "wrong_variant",
        "missing_rank",
        "missing_layer",
        "missing_compute",
        "missing_end",
    ],
)
def test_incomplete_or_changed_capture_cannot_publish_timings(tmp_path, failure, expect_error):
    receipt, rows, write = fixture(tmp_path)
    if failure == "prefill":
        receipt["prefill_calls"] = 1
    elif failure == "changed_output":
        receipt["output_hashes"][-1] = "changed"
    elif failure == "too_many_calls":
        receipt["decode_calls"] = 4
    elif failure == "cleanup":
        receipt["cleanup_completed"] = False
    elif failure == "wrong_variant":
        receipt["recurrence"] = "single_step"
    elif failure == "missing_rank":
        rows[:] = [r for r in rows if r.get("DEVICE ID") != "3"]
    elif failure == "missing_layer":
        for r in rows:
            r["OP CODE"] = r["OP CODE"].replace("L3_decode_forward", "L0_other")
    elif failure == "missing_compute":
        del next(r for r in rows if r.get("DEVICE ID") == "0")[DURATIONS["compute_ns"]]
    elif failure == "missing_end":
        rows.pop()
    write()
    with expect_error(ValueError, "Missing|missing|Incomplete"):
        collect(tmp_path, *CASES[0], "native")
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize("failure", ["file", "total", "space"])
def test_export_budget_rejects_growth_before_host_exhaustion(tmp_path, monkeypatch, failure, expect_error):
    (tmp_path / "a.csv").write_bytes(b"a" * 8)
    (tmp_path / "b.csv").write_bytes(b"b" * 8)
    monkeypatch.setattr(
        "models.demos.qwen38_27b_qb2.tests.bounded_profile.shutil.disk_usage", lambda p: SimpleNamespace(free=100)
    )
    options = dict(maximum_total=20, maximum_file=10, minimum_free=10)
    assert check_artifact_budget(tmp_path, **options) == 16
    options[{"file": "maximum_file", "total": "maximum_total", "space": "minimum_free"}[failure]] = {
        "file": 7,
        "total": 15,
        "space": 101,
    }[failure]
    with expect_error(RuntimeError, "budget|free space"):
        check_artifact_budget(tmp_path, **options)
