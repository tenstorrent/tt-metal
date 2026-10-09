# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prevent partial, reordered or overlapping traces from yielding a false pass."""

import copy

import pytest

from models.demos.qwen38_27b_qb2.tests.full_trace_profile import SCOPE, analyze, intervals_by_stage


def fixture():
    windows = {"MODEL": (1, 2000), "SAMPLER": (2001, 3000)}
    windows.update({f"L{i}": (100 + 20 * i, 110 + 20 * i) for i in range(64)})
    markers = [
        dict(**{"OP TYPE": "signpost", "OP CODE": f"FULLTRACE_{label}_{boundary}", "HOST START TS": str(t)})
        for label, times in windows.items()
        for boundary, t in zip(("BEGIN", "END"), times)
    ]
    rows = []
    for device in (0, 4, 8, 12):
        for replay in range(3):
            # Independent chip clocks cannot be compared across devices.
            base = device * 10**12 + replay * 100000
            for stage in range(66):
                sampler = stage == 65
                start = base + stage * 1000
                end = start + (900 if stage < 64 else 500 if stage == 64 else 300)
                capture = 102 + 20 * stage if stage < 64 else 1900 if stage == 64 else 2100
                rows.append(
                    {
                        "OP TYPE": "tt_dnn_device",
                        "OP CODE": "sample" if sampler else "layer_op",
                        "METAL TRACE ID": str(7 if sampler else 3),
                        "METAL TRACE REPLAY SESSION ID": str(replay + (31 if sampler else 11)),
                        "DEVICE ID": str(device),
                        "GLOBAL CALL COUNT": str(9000 + stage),
                        "HOST START TS": str(capture),
                        "DEVICE FW START CYCLE": str(start),
                        "DEVICE FW END CYCLE": str(end),
                        "DEVICE FW DURATION [ns]": str((end - start) / 1.5),
                        "DEVICE KERNEL DURATION [ns]": str((end - start) / 1.5 - 1),
                    }
                )
    receipt = dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        scope=SCOPE,
        layer_indices=list(range(64)),
        device_ids=[0, 4, 8, 12],
        prefill_calls=0,
        output_hashes=["same_logits"] * 5,
        token_hashes=["same_tokens"] * 5,
        model_trace_id=3,
        sample_trace_id=7,
        replays=[dict(host_step_s=65300 / 1.5 * 1.02 / 1e9) for _ in range(3)],
    )
    # Export ordering deliberately differs from capture time, as replay CSVs do.
    return markers + list(reversed(rows)), receipt


def test_all_ranks_replays_and_capture_timestamp_attribution():
    rows, receipt = fixture()
    report = analyze(rows, receipt)
    assert report["full_trace_reconciliation_passed"] and not report["p0_gate_passed"]
    assert len(report["ranks"]) == 12
    assert all(row["device_op_rows"] == 66 for row in report["ranks"])
    for row in report["ranks"]:
        assert abs(sum(row["disjoint_stage_ns"].values()) - row["span_ns"]) < 1e-6
        assert abs(row["span_ns"] - 65300 / 1.5) < 1e-6


def test_overlap_and_gaps_are_not_double_counted():
    assert intervals_by_stage([(0, 20, "A"), (10, 30, "B"), (40, 50, "A")]) == {
        "A": 20,
        "B": 10,
        "overlap": 10,
        "uncovered_gap": 10,
    }


@pytest.mark.parametrize(
    "damage,message",
    [
        ("duplicate", "Duplicate replay operation"),
        ("missing_layer", "missing one or more decoder layers"),
        ("missing_replay", "exactly three model and sampler replays"),
        ("missing_time", "Missing DEVICE FW END CYCLE"),
        ("mixed_clock", "Inconsistent profiler clock conversion"),
        ("wrong_capture", "outside its capture window"),
        ("unknown_rank", "Unqualified device"),
        ("missing_marker", "Incomplete capture window"),
    ],
)
def test_reject_incomplete_or_corrupt_trace(damage, message, expect_error):
    rows, receipt = fixture()
    if damage == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif damage == "missing_layer":
        rows = [r for r in rows if not (r.get("DEVICE ID") == "0" and r.get("GLOBAL CALL COUNT") == "9000")]
    elif damage == "missing_replay":
        rows = [r for r in rows if not (r.get("DEVICE ID") == "0" and r.get("METAL TRACE REPLAY SESSION ID") == "11")]
    elif damage == "missing_time":
        rows[-1]["DEVICE FW END CYCLE"] = "-"
    elif damage == "mixed_clock":
        rows[-1]["DEVICE FW DURATION [ns]"] = "1"
    elif damage == "wrong_capture":
        rows[-1]["HOST START TS"] = "4000"
    elif damage == "unknown_rank":
        rows[-1]["DEVICE ID"] = "99"
    else:
        rows = [r for r in rows if r.get("OP CODE") != "FULLTRACE_L3_END"]
    with expect_error(ValueError, message):
        analyze(rows, receipt)


def test_large_host_gap_is_reported_without_qualifying():
    rows, receipt = fixture()
    receipt["replays"][1]["host_step_s"] *= 2
    report = analyze(rows, receipt)
    assert report["measurements_complete"]
    assert not report["full_trace_reconciliation_passed"]
    assert report["comparisons"][1]["relative_unaccounted_time"] > 0.5


def test_reject_changed_logits_even_if_timing_is_complete(expect_error):
    rows, receipt = fixture()
    receipt["output_hashes"][-1] = "changed"
    with expect_error(ValueError, "logits must match exactly"):
        analyze(rows, receipt)
