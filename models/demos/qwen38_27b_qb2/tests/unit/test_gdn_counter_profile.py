# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_counter_profile import (
    collect_counters,
    compare_outputs,
    counter_names,
    pipeline_calls,
)
from models.demos.qwen38_27b_qb2.tests.unit.test_gdn_phase_profile import pipeline_report


def report():
    result = pipeline_report()
    result.update(source_sha256={"kernel": "a" * 64}, instrumented_sha256={"kernel": "b" * 64})
    return result


def test_native_counter_array_inventory(expect_error):
    header = "constexpr auto fpu_counters = {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}};"
    assert counter_names(header, ["fpu"]) == {"FPU_COUNTER", "SFPU_COUNTER"}
    with expect_error(ValueError, "array missing"):
        counter_names(header, ["pack"])


def op_rows():
    rows, call = [], 0
    for batch in (16, 32):
        for padding in ("zero", "skip"):
            for zones in (0, 1):
                for step in range(3):
                    label = f"GDN_PIPELINE_B{batch}_{padding.upper()}_ZONES{zones}_STEP{step}"
                    rows.append({"OP TYPE": "signpost", "OP CODE": label + "_BEGIN"})
                    for _ in range(2):
                        call += 1
                        for device in (3, 4, 5, 6):
                            rows.append(
                                {
                                    "OP TYPE": "tt_dnn_device",
                                    "OP CODE": "GenericOpDeviceOperation",
                                    "DEVICE ID": str(device),
                                    "GLOBAL CALL COUNT": str(call),
                                    "DEVICE KERNEL DURATION [ns]": "42000",
                                    "CORE COUNT": "2",
                                }
                            )
                    rows.append({"OP TYPE": "signpost", "OP CODE": label + "_END"})
    return rows


def test_exact_pipeline_operation_inventory():
    calls = pipeline_calls(op_rows(), report())
    assert len(calls) == 192
    assert calls[3, 1]["kind"] == "recurrence" and calls[3, 2]["kind"] == "epilogue"


@pytest.mark.parametrize("damage", ["rank", "duplicate", "missing_end", "repeated_case"])
def test_reject_damaged_signpost_inventory(damage, expect_error):
    rows = op_rows()
    if damage == "rank":
        del rows[2]
    elif damage == "duplicate":
        rows.insert(2, dict(rows[1]))
    elif damage == "missing_end":
        rows.pop()
    else:
        rows.extend(deepcopy(rows[:10]))
    with expect_error(ValueError, "pipeline"):
        pipeline_calls(rows, report())


def raw_rows():
    return [
        {
            "timer_id": "9090",
            "PCIe slot": "3",
            "run host ID": "1",
            "core_x": str(core),
            "core_y": "1",
            "meta data": "{'counter type': 0; 'value': 20; 'ref cnt': 100}",
        }
        for core in (1, 2)
    ]


def collect(rows):
    calls = {(3, 1): dict(cores=2, config=(16, "skip", 0, 0), kind="recurrence")}
    return collect_counters(rows, calls, {"FPU_COUNTER"}, {0: "FPU_COUNTER"})


def test_native_counter_payload_and_core_coverage():
    result = collect(raw_rows())
    assert result["active_core_calls"] == 2
    assert result["summaries"][0]["ratio_median"] == 0.2


@pytest.mark.parametrize("damage", ["missing_core", "missing_type", "duplicate", "zero_ref", "malformed", "unexpected"])
def test_reject_partial_or_invalid_counters(damage, expect_error):
    rows = raw_rows()
    if damage == "missing_core":
        rows.pop()
    elif damage == "duplicate":
        rows.append(dict(rows[0]))
    elif damage == "zero_ref":
        rows[0]["meta data"] = rows[0]["meta data"].replace("100", "0")
    elif damage == "missing_type":
        rows[0]["timer_id"] = "1"
    elif damage == "malformed":
        rows[0]["meta data"] = "{}"
    else:
        rows[0]["meta data"] = rows[0]["meta data"].replace("'counter type': 0", "'counter type': 'UNKNOWN'")
    with expect_error(ValueError, "ounter"):
        collect(rows)


@pytest.mark.parametrize("damage", [None, "source", "ranks", "outputs"])
def test_exact_cross_pass_output_check(damage, expect_error):
    baseline, observed = report(), report()
    if damage is None:
        compare_outputs(baseline, observed)
        return
    if damage == "source":
        observed["source_sha256"]["kernel"] = "f" * 64
    elif damage == "ranks":
        observed["device_ids"].reverse()
    else:
        for case in observed["cases"]:
            case["hashes"]["output"][0] = "f" * 64
    with expect_error(ValueError, "Counter"):
        compare_outputs(baseline, observed)
