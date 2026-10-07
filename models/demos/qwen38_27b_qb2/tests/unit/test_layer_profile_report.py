# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protect profile attribution from warmup, rank summation and lost timings."""

import json
from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.tests.layer_profile_report import (
    PROFILE_CASES,
    analyze,
    drain_after_call,
    require_storage_headroom,
    write_report,
)


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


def test_prefill_drain_bounds_records_for_each_chunk():
    pending, outputs = [], []

    def prefill(value, *, slot):
        assert not pending, "Previous chunk was not drained"
        pending.append((value, slot))
        return value

    def drain():
        outputs.extend(pending)
        pending.clear()

    wrapped = drain_after_call(prefill, drain)
    tokens = object()
    for chunk in range(64):
        assert wrapped(tokens, slot=chunk) is tokens
    assert outputs == [(tokens, chunk) for chunk in range(64)]


def test_failed_prefill_does_not_mask_failure_with_profiler_drain(expect_error):
    def fail():
        raise RuntimeError("prefill failed")

    def unexpected_drain():
        raise AssertionError("Drain must not run after failed prefill")

    with expect_error(RuntimeError, "prefill failed"):
        drain_after_call(fail, unexpected_drain)()


@pytest.mark.parametrize("full_path", ["artifacts", "jit"])
def test_storage_guard_checks_both_filesystems(monkeypatch, full_path, expect_error):
    monkeypatch.setattr(
        "models.demos.qwen38_27b_qb2.tests.layer_profile_report.shutil.disk_usage",
        lambda path: SimpleNamespace(free=15 * 1024**3 if path == full_path else 32 * 1024**3),
    )
    with expect_error(RuntimeError, f"Profiler storage guard: {full_path}"):
        require_storage_headroom(["artifacts", "jit"])


@pytest.mark.parametrize("exhausted_at", ["before", "after"])
def test_storage_guard_stops_prefill_at_chunk_boundaries(exhausted_at, expect_error):
    calls = []

    def guard():
        if exhausted_at == "before" or "drain" in calls:
            raise RuntimeError("Profiler storage guard")

    wrapped = drain_after_call(lambda: calls.append("prefill"), lambda: calls.append("drain"), storage_guard=guard)
    with expect_error(RuntimeError, "Profiler storage guard"):
        wrapped()
    assert calls == ([] if exhausted_at == "before" else ["prefill", "drain"])
