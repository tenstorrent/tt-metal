# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.gemma4_d_p.demo.prefill_perf import _replay_spans


def replay_records():
    records = {}
    for chip in range(2):
        rows = []
        for replay in range(6):
            start = replay * 1_000_000 + chip * 100_000_000
            duration = 900_000 if replay % 3 == 0 else (replay + chip) * 1000
            rows.extend([(11, start, start + 100, 1.0, 0), (12, start + 200, start + duration, 1.0, 1)])
        records[chip] = list(reversed(rows))
    return records


def test_device_spans_exclude_warmups_host_gaps_and_other_chip_clocks():
    sources = {("q.cpp",): 0, ("k.cpp",): 1}
    captures = _replay_spans(replay_records(), list(range(6)), 2, 2, 2, sources)
    assert [(c["slot"], c["sample"]) for c in captures] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [c["device_us"] for c in captures] == [2, 3, 5, 6]
    assert [c["profiled_replay_host_us"] for c in captures] == [1, 2, 4, 5]
    assert captures[0]["profile"] == {
        "chip_id": 1,
        "kernel_sources": {0: ("q.cpp",), 1: ("k.cpp",)},
        "programs": [(101_000_000, 101_000_100, 0), (101_000_200, 101_002_000, 1)],
    }


@pytest.mark.parametrize(
    "damage", ["missing_chip", "missing_record", "truncated_replay", "wrong_program", "invalid_time"]
)
def test_device_spans_reject_incomplete_records(damage, expect_error):
    records = replay_records()
    if damage == "missing_chip":
        del records[1]
    elif damage == "missing_record":
        records[1].pop()
    elif damage == "truncated_replay":
        for rows in records.values():
            del rows[:2]
    elif damage == "wrong_program":
        record = records[1][0]
        records[1][0] = (999, *record[1:])
    else:
        runtime_id, start, _, frequency, source = records[1][0]
        records[1][0] = (runtime_id, start, start, frequency, source)
    with expect_error(AssertionError, ".*"):
        _replay_spans(records, list(range(6)), 2, 2, 2)
