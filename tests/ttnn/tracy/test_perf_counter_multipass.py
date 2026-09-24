#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from tracy.perf_counter_multipass import (
    PERF_COUNTER_L1_GROUPS,
    PERF_COUNTER_MAX_GROUPS_PER_PASS,
    arch_l1_groups,
    merge_perf_counter_device_logs,
    perf_counter_groups_to_bitfield,
    schedule_perf_counter_passes,
)

CSV_HEADER = (
    "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, "
    "trace id, trace id counter, zone name, type, source line, source file, meta data\n"
)


def row(*fields):
    """A device log row: the eight leading fields, then the trace columns and the rest."""
    return ",".join(str(f) for f in fields) + ",,,,,,,\n"


def assert_pass_invariants(passes):
    for p in passes:
        assert len(p) <= PERF_COUNTER_MAX_GROUPS_PER_PASS
        assert sum(g in PERF_COUNTER_L1_GROUPS for g in p) <= 1


def test_full_blackhole_set_schedules_one_pass_per_l1_bank():
    groups = ["fpu", "pack", "unpack", "instrn", "l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"]
    passes = schedule_perf_counter_passes(groups)
    assert len(passes) == 6
    assert_pass_invariants(passes)
    assert sorted(g for p in passes for g in p) == sorted(groups)


def test_wormhole_all_set_schedules_two_passes():
    groups = ["fpu", "pack", "unpack", "instrn", "l1_0", "l1_1"]
    passes = schedule_perf_counter_passes(groups)
    assert len(passes) == 2
    assert_pass_invariants(passes)
    assert sorted(g for p in passes for g in p) == sorted(groups)


def test_requests_that_fit_stay_single_pass():
    assert schedule_perf_counter_passes(["fpu", "pack", "unpack"]) == [["fpu", "pack", "unpack"]]
    assert len(schedule_perf_counter_passes(["fpu", "instrn", "l1_0"])) == 1


def test_two_l1_banks_force_two_passes():
    assert len(schedule_perf_counter_passes(["l1_0", "l1_1"])) == 2


def test_group_cap_forces_extra_pass():
    assert len(schedule_perf_counter_passes(["fpu", "pack", "unpack", "instrn"])) == 2


def test_dedup_case_insensitive_and_empty():
    assert schedule_perf_counter_passes(["FPU", "fpu", "Pack"]) == [["fpu", "pack"]]
    assert schedule_perf_counter_passes([]) == []


def test_bitfield_matches_perf_counters_hpp_bits():
    assert perf_counter_groups_to_bitfield(["fpu", "pack", "unpack", "l1_0", "instrn"]) == 47
    assert perf_counter_groups_to_bitfield(["l1_4"]) == 1 << 8
    assert perf_counter_groups_to_bitfield(["l1_5"]) == 1 << 9


def test_merge_keeps_pass0_whole_and_appends_only_counter_rows(tmp_path):
    pass0 = tmp_path / "pass_0.csv"
    pass1 = tmp_path / "pass_1.csv"
    merged = tmp_path / "merged.csv"
    pass0.write_text(CSV_HEADER + row(0, 1, 1, "BRISC", 4096, 100, 0, 7) + row(0, 1, 1, "BRISC", " 9090 ", 110, 42, 7))
    pass1.write_text(
        CSV_HEADER
        + row(0, 1, 1, "BRISC", 4096, 105, 0, 7)
        + row(0, 1, 1, "BRISC", 9090, 115, 43, 7)
        + row(0, 1, 1, "TRISC_0", 9090, 116, 44, 7)
    )

    merge_perf_counter_device_logs([pass0, pass1], merged)

    lines = merged.read_text().splitlines()
    assert lines[0] == CSV_HEADER.rstrip("\n")
    assert lines[1:3] == [
        row(0, 1, 1, "BRISC", 4096, 100, 0, 7).rstrip(),
        row(0, 1, 1, "BRISC", " 9090 ", 110, 42, 7).rstrip(),
    ]
    # appended rows keep their data but take pass 0's timestamp for the same core and run host id, whatever the RISC
    assert lines[3:] == [
        row(0, 1, 1, "BRISC", 9090, 110, 43, 7).rstrip(),
        row(0, 1, 1, "TRISC_0", 9090, 110, 44, 7).rstrip(),
    ]
    assert sum("4096" in line for line in lines) == 1
    assert sum(line.startswith("PCIe") for line in lines) == 1


def test_merge_anchors_each_trace_replay_on_its_own_pass0_row(tmp_path):
    pass0 = tmp_path / "pass_0.csv"
    pass1 = tmp_path / "pass_1.csv"
    merged = tmp_path / "merged.csv"

    def traced(ts, data, replay):
        return ",".join(str(f) for f in (0, 1, 1, "BRISC", 9090, ts, data, 7, 3, replay)) + ",,,,,\n"

    # two replays of one traced op share the run host id and differ in the trace id counter
    pass0.write_text(CSV_HEADER + traced(110, 42, 0) + traced(210, 42, 1))
    pass1.write_text(CSV_HEADER + traced(115, 43, 0) + traced(215, 43, 1) + traced(315, 43, 2))

    merge_perf_counter_device_logs([pass0, pass1], merged)

    lines = merged.read_text().splitlines()
    # the third replay has no pass 0 row and is dropped
    assert lines[3:] == [traced(110, 43, 0).rstrip(), traced(210, 43, 1).rstrip()]


def test_merge_drops_counter_rows_with_no_pass0_anchor(tmp_path):
    pass0 = tmp_path / "pass_0.csv"
    pass1 = tmp_path / "pass_1.csv"
    merged = tmp_path / "merged.csv"
    pass0.write_text(CSV_HEADER + row(0, 1, 1, "BRISC", 9090, 110, 42, 7))
    pass1.write_text(CSV_HEADER + row(0, 1, 1, "BRISC", 9090, 115, 43, 7) + row(0, 2, 2, "BRISC", 9090, 116, 44, 9))

    merge_perf_counter_device_logs([pass0, pass1], merged)

    lines = merged.read_text().splitlines()
    assert lines[1:] == [
        row(0, 1, 1, "BRISC", 9090, 110, 42, 7).rstrip(),
        row(0, 1, 1, "BRISC", 9090, 110, 43, 7).rstrip(),
    ]


def test_arch_l1_groups():
    assert arch_l1_groups(True) == ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"]
    assert arch_l1_groups(False, is_quasar=True) == []
    assert arch_l1_groups(False) == ["l1_0", "l1_1"]
