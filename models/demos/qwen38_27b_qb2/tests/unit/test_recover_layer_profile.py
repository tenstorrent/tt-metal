# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check recovery's join boundaries and prevent hidden missing measurements."""

import pytest

from models.demos.qwen38_27b_qb2.tests.layer_profile_report import analyze
from models.demos.qwen38_27b_qb2.tests.recover_layer_profile import join_windows
from models.demos.qwen38_27b_qb2.tests.unit.test_layer_profile_report import fixture


def inputs():
    rows, receipt = fixture()
    ops, posts, timings = {}, {}, []
    for timestamp, row in enumerate(rows, start=1):
        if row.get("OP TYPE") == "signpost":
            posts[str(timestamp)] = dict(data="TT_SIGNPOST: " + row["OP CODE"], tracy_time=str(timestamp))
        else:
            ops[timestamp] = dict(
                op_code=row["OP CODE"],
                op_type=row["OP TYPE"],
                device_id=int(row["DEVICE ID"]),
                global_call_count=timestamp,
                tracy_time=str(timestamp),
                metal_trace_id=None,
            )
            timings.append(dict(row, **{"GLOBAL CALL COUNT": str(timestamp)}))
    return ops, posts, timings, receipt


def test_recovery_preserves_windows_ranks_and_device_timings():
    ops, posts, timings, receipt = inputs()
    report = analyze(join_windows(ops, posts, iter(timings)), receipt)
    assert report["measurements_complete"]
    assert all(row["firmware_ns"] == 300 and row["device_op_rows"] == 2 for row in report["device_totals"])


def test_missing_compact_row_remains_visible():
    ops, posts, timings, receipt = inputs()
    del timings[4]  # First measured row, after four unmarked warmup ranks.
    report = analyze(join_windows(ops, posts, iter(timings)), receipt)
    assert not report["measurements_complete"]
    assert sum(row.get("missing_firmware_ns_rows", 0) for row in report["device_totals"]) == 1


@pytest.mark.parametrize("failure", ["duplicate", "device", "trace", "capture", "unclosed"])
def test_corrupt_join_is_rejected(failure, expect_error):
    ops, posts, timings, _ = inputs()
    op_id = int(timings[4]["GLOBAL CALL COUNT"])
    if failure == "duplicate":
        timings.append(dict(timings[4]))
    elif failure == "device":
        timings[4]["DEVICE ID"] = "99"
    elif failure == "trace":
        timings[4]["METAL TRACE ID"] = "0"
    elif failure == "capture":
        ops[op_id]["metal_trace_id"] = 0
    else:
        del posts[next(reversed(posts))]
    with expect_error(ValueError, "Duplicate|mismatch|Trace|Incomplete"):
        join_windows(ops, posts, iter(timings))
