# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.gemma4_d_p.demo import prefill_perf
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
    assert captures[0]["chips"][1]["program_us"] == pytest.approx(1.9)
    assert captures[0]["chips"][1]["gap_us"] == pytest.approx(0.1)
    assert captures[0]["profile"] == {
        "chip_id": 1,
        "kernel_sources": {0: ("q.cpp",), 1: ("k.cpp",)},
        "programs": [(101_000_000, 101_000_100, 0), (101_000_200, 101_002_000, 1)],
    }


def test_device_program_time_does_not_double_count_overlapping_programs():
    records = {
        0: [(11, 0, 100, 1.0, None), (12, 50, 200, 1.0, None), (11, 1000, 1100, 1.0, None), (12, 1050, 1200, 1.0, None)]
    }
    capture = _replay_spans(records, [1, 1], 1, 1, 1)[0]
    assert capture["chips"][0]["span_us"] == pytest.approx(0.2)
    assert capture["chips"][0]["program_us"] == pytest.approx(0.2)
    assert capture["chips"][0]["gap_us"] == 0


def test_eager_replays_validate_kernels_with_changing_runtime_ids(expect_error):
    records = replay_records()
    for chip, rows in records.items():
        records[chip] = [(i + 100, *row[1:]) for i, row in enumerate(rows)]
    captures = _replay_spans(records, list(range(6)), 2, 2, 2, validate_sources=True)
    assert [c["device_us"] for c in captures] == [2, 3, 5, 6]
    records[1][-4] = (*records[1][-4][:-1], 999)
    with expect_error(AssertionError, "Incomplete program sequence"):
        _replay_spans(records, list(range(6)), 2, 2, 2, validate_sources=True)


def test_hybrid_replays_use_trace_ids_when_kernel_metadata_is_overwritten():
    records = replay_records()
    for chip, rows in records.items():
        records[chip] = [
            (runtime_id, start, end, frequency, i) if runtime_id == 11 else (i + 100, start, end, frequency, source)
            for i, (runtime_id, start, end, frequency, source) in enumerate(rows)
        ]
    captures = _replay_spans(records, list(range(6)), 2, 2, 2, validate_sources=True)
    assert [c["device_us"] for c in captures] == [2, 3, 5, 6]


def test_device_samples_retry_filtered_runtime_id_zero(monkeypatch):
    incomplete = replay_records()
    for rows in incomplete.values():
        rows.pop()
        rows[0] = (65535, *rows[0][1:])
        rows[1] = (1, *rows[1][1:])
    attempts = iter([(incomplete, list(range(6)), None), (replay_records(), list(range(6)), None)])
    monkeypatch.setattr(prefill_perf, "_collect_device_samples", lambda *args, **kwargs: next(attempts))
    captures = prefill_perf._device_samples(
        SimpleNamespace(get_num_devices=lambda: 2), run=lambda: None, repeats=2, num_slots=2
    )
    assert [c["device_us"] for c in captures] == [2, 3, 5, 6]
    assert all(c["collection_attempts"] == 2 for c in captures)


def test_device_samples_do_not_retry_other_missing_records(monkeypatch, expect_error):
    incomplete = replay_records()
    for rows in incomplete.values():
        rows.pop()
    attempts = iter([(incomplete, list(range(6)), None)])
    monkeypatch.setattr(prefill_perf, "_collect_device_samples", lambda *args, **kwargs: next(attempts))
    with expect_error(AssertionError, "Cannot split"):
        prefill_perf._device_samples(
            SimpleNamespace(get_num_devices=lambda: 2), run=lambda: None, repeats=2, num_slots=2
        )


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
