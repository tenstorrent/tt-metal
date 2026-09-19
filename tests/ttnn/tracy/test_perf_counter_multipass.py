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

CSV_HEADER = "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID\n"


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


def test_all_non_l1_groups_and_one_l1_bank_fit_one_pass():
    # fpu, pack, unpack, instrn plus one L1 bank is the largest single pass mask (47 with l1_0); measured to fit the
    # BRISC firmware on Blackhole (8684 of 8704 bytes) and Wormhole (7600 of 7712).
    passes = schedule_perf_counter_passes(["fpu", "pack", "unpack", "instrn", "l1_0"])
    assert passes == [["l1_0", "fpu", "pack", "unpack", "instrn"]]
    assert perf_counter_groups_to_bitfield(passes[0]) == 47


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
    # the cap is a parameter; with the default every non L1 group fits one pass
    assert len(schedule_perf_counter_passes(["fpu", "pack", "unpack", "instrn"])) == 1
    assert len(schedule_perf_counter_passes(["fpu", "pack", "unpack", "instrn"], max_groups_per_pass=3)) == 2


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
    pass0.write_text(CSV_HEADER + "0,1,1,BRISC,4096,100,0,7\n0,1,1,BRISC, 9090 ,110,42,7\n")
    pass1.write_text(CSV_HEADER + "0,1,1,BRISC,4096,105,0,7\n0,1,1,BRISC,9090,115,43,7\n0,1,1,TRISC_0,9090,116,44,7\n")

    merge_perf_counter_device_logs([pass0, pass1], merged)

    lines = merged.read_text().splitlines()
    assert lines[0] == CSV_HEADER.rstrip("\n")
    assert lines[1:3] == ["0,1,1,BRISC,4096,100,0,7", "0,1,1,BRISC, 9090 ,110,42,7"]
    assert lines[3:] == ["0,1,1,BRISC,9090,115,43,7", "0,1,1,TRISC_0,9090,116,44,7"]
    assert sum("4096" in line for line in lines) == 1
    assert sum(line.startswith("PCIe") for line in lines) == 1


def test_arch_l1_groups():
    assert arch_l1_groups(True) == ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"]
    assert arch_l1_groups(False, is_quasar=True) == []
    assert arch_l1_groups(False) == ["l1_0", "l1_1"]


def _zone(run_id, ts, phase, core=(1, 1)):
    timer = {"id": 1, "zone_name": "BRISC-FW", "type": phase, "run_host_id": run_id}
    return (timer, ts, 0, "BRISC", core)


def _counter(run_id, ts, core=(1, 1)):
    timer = {"id": 9090, "zone_name": "", "type": "TS_DATA", "run_host_id": run_id}
    return (timer, ts, 7, "BRISC", core)


def test_get_ops_keeps_merged_pass_counters_inside_their_op():
    # a merged multipass log carries counter records from later runs, whose timestamps fall outside the
    # FW window of the first run; they must not open a spurious op or swallow the next one
    from tracy.process_device_log import get_ops

    timeseries = [
        _zone(1, 100, "ZONE_START"),
        _counter(1, 150),
        _zone(1, 200, "ZONE_END"),
        _zone(2, 300, "ZONE_START"),
        _counter(2, 350),
        _zone(2, 400, "ZONE_END"),
        _counter(1, 5000),  # from pass 2, same op, much later run
        _counter(2, 5100),
    ]
    timeseries.sort(key=lambda ts: ts[1])
    ops = get_ops(timeseries)
    assert [op["timeseries"][0][0]["run_host_id"] for op in ops] == [1, 2]
    counters = [sorted(ts[1] for ts in op["timeseries"] if ts[0]["type"] == "TS_DATA") for op in ops]
    assert counters == [[150, 5000], [350, 5100]]
    assert all(op["timeseries"][0][0]["type"] == "ZONE_START" for op in ops)
