# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protect profile attribution from warmup, rank summation and lost timings."""

import json

import pytest

from models.demos.qwen38_27b_qb2.tests.layer_profile_report import PROFILE_CASES, analyze, write_report


def fixture():
    receipt = dict(
        passed=True,
        state="completed",
        device_ids=[0, 1, 2, 3],
        cells=[dict(input_tokens=length, batch=batch) for length, batch in PROFILE_CASES],
    )
    rows = []
    for length, batch in PROFILE_CASES:
        prefix = f"P0_S{length}_B{batch}"

        def marker(label, boundary):
            rows.append({"OP TYPE": "signpost", "OP CODE": f"{prefix}_{label}_{boundary}"})

        def operation(index, fw):
            for device in receipt["device_ids"]:
                rows.append(
                    {
                        "OP CODE": "MatmulDeviceOperation",
                        "OP TYPE": "tt_dnn_device",
                        "DEVICE ID": str(device),
                        "GLOBAL CALL COUNT": str(index),
                        "DEVICE FW DURATION [ns]": str(fw),
                        "DEVICE KERNEL DURATION [ns]": str(fw - 10),
                        "DEVICE BRISC KERNEL DURATION [ns]": "20",
                        "DEVICE NCRISC KERNEL DURATION [ns]": "10",
                        "DEVICE TRISC1 KERNEL DURATION [ns]": "15",
                    }
                )

        operation(0, 900000)  # unmarked compilation / warmup, excluded
        marker("MODEL", "BEGIN")
        marker("L0_decode_forward", "BEGIN")
        marker("L0__delta", "BEGIN")
        operation(1, 100)
        marker("L0__delta", "END")
        operation(2, 200)
        marker("L0_decode_forward", "END")
        marker("MODEL", "END")
    return rows, receipt


def test_parallel_ranks_and_nested_stages_are_not_double_counted(tmp_path):
    report = analyze(*fixture())
    assert report["measurements_complete"]
    assert not report["p0_gate_passed"]
    assert len(report["device_totals"]) == 4 * len(PROFILE_CASES)
    assert all(row["firmware_ns"] == 300 and row["device_op_rows"] == 2 for row in report["device_totals"])
    for total in report["device_totals"]:
        stages = [
            r
            for r in report["exclusive_stages"]
            if all(r[key] == total[key] for key in ("input_tokens", "batch", "device"))
        ]
        assert sum(row["firmware_ns"] for row in stages) == total["firmware_ns"]
    output = tmp_path / "report"
    write_report(report, output)
    assert json.loads((output / "profile-summary.json").read_text())["p0_gate_passed"] is False
    assert "Input 262016" in (output / "report.md").read_text()


@pytest.mark.parametrize("failure", ["missing_end", "wrong_end", "missing_rank", "duplicate", "trace", "nan"])
def test_invalid_measurements_cannot_be_reported(failure, expect_error):
    rows, receipt = fixture()
    op = next(row for row in rows if row.get("GLOBAL CALL COUNT") == "1")
    if failure == "missing_end":
        rows.pop()
    elif failure == "wrong_end":
        rows[-1]["OP CODE"] = rows[-1]["OP CODE"].replace("MODEL", "L0__delta")
    elif failure == "missing_rank":
        rows = [row for row in rows if row.get("DEVICE ID") != "3"]
    elif failure == "duplicate":
        index = rows.index(op)
        rows.insert(index, dict(op))
    elif failure == "trace":
        op["METAL TRACE ID"] = "0"
    else:
        op["DEVICE FW DURATION [ns]"] = "nan"
    with expect_error(ValueError, "signpost|measurements|Duplicate|Trace|Invalid"):
        analyze(rows, receipt)


def test_missing_device_timing_is_reported_as_incomplete():
    rows, receipt = fixture()
    op = next(row for row in rows if row.get("GLOBAL CALL COUNT") == "1")
    del op["DEVICE FW DURATION [ns]"]
    report = analyze(rows, receipt)
    assert not report["measurements_complete"]
    assert sum(row.get("missing_firmware_ns_rows", 0) for row in report["device_totals"]) == 1


def test_long_context_priority_retains_capacity_and_short_context_control():
    assert (8192, 1) in PROFILE_CASES and (8192, 16) in PROFILE_CASES
    assert {(131072, 1), (131072, 8), (262016, 1), (262016, 4)} <= set(PROFILE_CASES)
    assert all((length + 128) * batch <= 1050592 for length, batch in PROFILE_CASES)
