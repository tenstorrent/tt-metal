# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from models.demos.qwen38_27b_qb2.tests.large_operator_profile import (
    LABELS,
    boundary_calls,
    capture_plan,
    collect_counters,
    compare_outputs,
    selected_labels,
)


def receipt():
    return dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        device_ids=[0, 4, 12, 8],
        batch=16,
        context=32768,
        source_sha256={"source": "a" * 64},
        precision={"weights": "bfloat8_b"},
        cases=[
            dict(
                label=label,
                passed=True,
                samples=[dict(repeat=i, hashes=["b" * 64] * 4, accuracy=[dict(passed=True)] * 4) for i in range(3)],
            )
            for label in LABELS
        ],
    )


def operations():
    rows, call = [], 0
    for label in LABELS:
        for repeat in range(3):
            tag = f"LARGE_OP_{label}_{repeat}"
            rows.append({"OP TYPE": "signpost", "OP CODE": tag + "_BEGIN"})
            call += 1
            for device in (0, 4, 12, 8):
                rows.append(
                    {
                        "OP TYPE": "tt_dnn_device",
                        "OP CODE": "SdpaDecodeDeviceOperation" if label == "SDPA_COMPLETE" else "MatmulDeviceOperation",
                        "DEVICE ID": str(device),
                        "GLOBAL CALL COUNT": str(call),
                        "DEVICE KERNEL DURATION [ns]": "42000",
                        "CORE COUNT": "3",
                    }
                )
            rows.append({"OP TYPE": "signpost", "OP CODE": tag + "_END"})
    return rows


def test_complete_boundary_coverage():
    calls, inventory = boundary_calls(operations(), receipt())
    assert len(calls) == len(inventory) == 60


@pytest.mark.parametrize("label", LABELS)
def test_split_capture_keeps_three_samples_and_all_ranks(label):
    full = receipt()
    partial = deepcopy(full)
    partial["boundary_selection"] = label
    partial["cases"] = [c for c in partial["cases"] if c["label"] == label]
    rows = operations()
    offset = list(LABELS).index(label) * 18
    calls, _ = boundary_calls(rows[offset : offset + 18], partial)
    assert len(calls) == 12
    compare_outputs(full, partial)


def test_split_plan_covers_every_group_and_boundary():
    groups = [["instrn"], ["l1_0"], ["l1_1"]]
    plan = capture_plan(groups, True)
    assert len(plan) == 17
    assert plan[0] == ("before", [], "all") and plan[-1] == ("after", [], "all")
    assert {(tuple(group), boundary) for _, group, boundary in plan[1:-1]} == {
        (tuple(group), boundary) for group in groups for boundary in LABELS
    }


def test_split_capture_cannot_hide_missing_cases(expect_error):
    partial = receipt()
    partial["cases"].pop()
    with expect_error(ValueError, "Missing or repeated"):
        compare_outputs(receipt(), partial)
    with expect_error(ValueError, "Unsupported operator"):
        selected_labels("UNKNOWN")


@pytest.mark.parametrize("damage", ["missing_rank", "duplicate", "missing_end", "wrong_op"])
def test_reject_incomplete_boundaries(damage, expect_error):
    rows = operations()
    if damage == "missing_rank":
        del rows[2]
    elif damage == "duplicate":
        rows.insert(2, dict(rows[1]))
    elif damage == "missing_end":
        rows.pop()
    else:
        rows[1]["OP CODE"] = "OtherDeviceOperation"
    message = {
        "missing_rank": "Operator boundary lacks",
        "duplicate": "Duplicate target",
        "missing_end": "Incomplete operator",
        "wrong_op": "Operator boundary lacks",
    }[damage]
    with expect_error(ValueError, message):
        boundary_calls(rows, receipt())


def raw_rows():
    rows = []
    for core in (1, 2):
        common = {"PCIe slot": "0", "run host ID": "1", "core_x": str(core), "core_y": "1"}
        for kind, cycle in (("ZONE_START", 100), ("ZONE_END", 200)):
            rows.append(
                dict(
                    common,
                    **{
                        "RISC processor type": "TRISC_1",
                        "zone name": "TRISC-KERNEL",
                        "type": kind,
                        "timer_id": "100",
                        "time[cycles since reset]": str(cycle),
                    },
                )
            )
        rows.append(
            dict(
                common,
                **{
                    "RISC processor type": "BRISC",
                    "zone name": "",
                    "timer_id": "9090",
                    "meta data": '{"counter type":"FPU_COUNTER";"value":20;"ref cnt":100}',
                },
            )
        )
    return rows


def collect(rows):
    return collect_counters(rows, {(0, 1): dict(label="OUTPUT_LOCAL", cores=3)}, {"FPU_COUNTER"}, {})


def test_reader_only_core_not_counted_as_compute():
    result = collect(raw_rows())
    assert result["compute_core_calls"] == 2
    assert result["summaries"][0]["ratio_median"] == 0.2


@pytest.mark.parametrize("damage", ["counter", "start", "end", "duplicate", "zero_interval"])
def test_reject_incomplete_compute_counter_coverage(damage, expect_error):
    rows = raw_rows()
    if damage == "counter":
        rows.pop()
    elif damage == "start":
        del rows[0]
    elif damage == "end":
        del rows[1]
    elif damage == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows[-1]["meta data"] = rows[-1]["meta data"].replace('"ref cnt":100', '"ref cnt":0')
    message = {
        "counter": "Incomplete compute-core",
        "start": "Unpaired compute",
        "end": "Unpaired compute",
        "duplicate": "Duplicate or unexpected",
        "zero_interval": "Invalid counter interval",
    }[damage]
    with expect_error(ValueError, message):
        collect(rows)


@pytest.mark.parametrize("damage", [None, "source", "output", "precision", "accuracy", "cleanup"])
def test_cross_pass_output_and_source_identity(damage, expect_error):
    before, after = receipt(), deepcopy(receipt())
    if damage is None:
        compare_outputs(before, after)
        return
    if damage == "source":
        after["source_sha256"] = {"source": "f" * 64}
    elif damage == "output":
        for s in after["cases"][0]["samples"]:
            s["hashes"][0] = "f" * 64
    elif damage == "precision":
        after["precision"]["weights"] = "bfloat4_b"
    elif damage == "accuracy":
        after["cases"][0]["samples"][0]["accuracy"][0]["passed"] = False
    else:
        after["cleanup_completed"] = False
    message = {
        "source": "Counter pass changed",
        "output": "Counter pass changed",
        "precision": "Counter pass changed",
        "accuracy": "Operator accuracy",
        "cleanup": "Incomplete physical",
    }[damage]
    with expect_error(ValueError, message):
        compare_outputs(before, after)
