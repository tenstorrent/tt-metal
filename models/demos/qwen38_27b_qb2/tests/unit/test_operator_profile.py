# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic inventory must retain unknown ops and account for every elapsed cycle."""

import copy
import math

from models.demos.qwen38_27b_qb2.demo import watch_operator_profile as watcher
from models.demos.qwen38_27b_qb2.tests.operator_profile import build_report
from models.demos.qwen38_27b_qb2.tests.unit.test_full_trace_profile import fixture as trace_fixture


def fixture():
    rows, profile = trace_fixture()
    profile.update(
        input_tokens=32768,
        batch=16,
        operand_hashes={"state": "same"},
        source_sha256={"model": "same"},
        precision=dict(
            decode_recurrence="single_step_compact_gdn",
            recurrent_dtype="float32",
            kv_cache_dtype="bfloat8_b",
            weight_groups={"all": "bfloat8_b"},
        ),
    )
    for row in rows:
        if row.get("GLOBAL CALL COUNT") == "9000":
            row.update(
                {
                    "OP CODE": "MatmulDeviceOperation",
                    "INPUT_1_DATATYPE": "BFLOAT8_B",
                    "INPUT_1_MEMORY": "DRAM",
                    "INPUT_1_W_PAD[LOGICAL]": "1[1]",
                    "INPUT_1_Z_PAD[LOGICAL]": "1[1]",
                    "INPUT_1_Y_PAD[LOGICAL]": "32[32]",
                    "INPUT_1_X_PAD[LOGICAL]": "64[48]",
                    "ATTRIBUTES": "test",
                }
            )
    baseline = copy.deepcopy(profile)
    for row in baseline["replays"]:
        row["host_step_s"] /= 1.1
    return rows, profile, baseline


def test_dynamic_operator_counts_and_disjoint_spans():
    rows, profile, baseline = fixture()
    report = build_report(rows, profile, baseline)
    assert report["distinct_operation_types"] == 3
    assert report["device_op_rows_per_rank"] == [66] * 12
    assert report["encoded_weight_bytes_per_chip"] == 2176
    assert math.isclose(report["profiler_overhead_fraction"], 0.1)
    assert report["projections"][0]["projection"] == "Other weight projection"
    assert report["projections"][0]["stored_kn"] == [32, 64]
    assert not report["physical_dram_counters"] and not report["counts_are_program_counts"]
    for rank in report["disjoint_rank_timelines"]:
        assert math.isclose(sum(rank["families_ms"].values()), rank["firmware_span_ms"])
        assert rank["families_ms"]["other"] > 0
        assert rank["families_ms"]["uncovered_gap"] > 0


def test_partial_or_changed_capture_cannot_yield_bandwidth(expect_error):
    rows, profile, baseline = fixture()
    for damaged in (rows + [copy.deepcopy(rows[-1])], rows[:-1]):
        with expect_error(ValueError, ".*"):
            build_report(damaged, profile, baseline)
    changed = copy.deepcopy(baseline)
    changed["operand_hashes"]["state"] = "different"
    with expect_error(ValueError, "mismatch"):
        build_report(rows, profile, changed)


def test_unknown_weight_precision_or_memory_is_rejected(expect_error):
    rows, profile, baseline = fixture()
    for column, value in (
        ("INPUT_1_DATATYPE", "BFLOAT4_B"),
        ("INPUT_1_MEMORY", "L1"),
        ("INPUT_1_X_PAD[LOGICAL]", "48"),
    ):
        modified = copy.deepcopy(rows)
        next(row for row in modified if row.get("OP CODE") == "MatmulDeviceOperation")[column] = value
        with expect_error(ValueError, ".*"):
            build_report(modified, profile, baseline)


def test_different_projection_count_on_one_rank_is_rejected(expect_error):
    rows, profile, baseline = fixture()
    next(row for row in rows if row.get("OP CODE") == "MatmulDeviceOperation")["OP CODE"] = "OtherOperation"
    with expect_error(ValueError, "Projection calls differ"):
        build_report(rows, profile, baseline)


def test_overlapping_families_are_explicit_not_double_counted():
    rows, profile, baseline = fixture()
    # Extend each first matmul into the following operation, preserving the
    # same clock ratio and overall replay span.
    for row in rows:
        if row.get("OP CODE") == "MatmulDeviceOperation":
            row["DEVICE FW END CYCLE"] = str(int(row["DEVICE FW START CYCLE"]) + 1500)
            row["DEVICE FW DURATION [ns]"] = str(1000)
    report = build_report(rows, profile, baseline)
    for rank in report["disjoint_rank_timelines"]:
        assert rank["families_ms"]["overlap"] > 0
        assert math.isclose(sum(rank["families_ms"].values()), rank["firmware_span_ms"])


def test_live_or_failed_producer_is_not_ready(monkeypatch, expect_error):
    props = dict(MainPID="15", ActiveState="active", LoadState="loaded", Result="success", InvocationID="original")
    monkeypatch.setattr(watcher, "properties", lambda unit: props)
    receipt = dict(state="completed", cleanup_completed=True)
    assert not watcher.finished("unit", "original", receipt)
    props.update(MainPID="0", ActiveState="failed", Result="exit-code")
    assert not watcher.finished("unit", "original", receipt)
    props.update(InvocationID="changed")
    with expect_error(ValueError, "invocation changed"):
        watcher.finished("unit", "original", receipt)
    props.update(MainPID="0", ActiveState="inactive", LoadState="not-found", InvocationID="", Result="success")
    assert watcher.finished("unit", "original", receipt)
    assert not watcher.finished("unit", "original", dict(state="completed", cleanup_completed=False))
