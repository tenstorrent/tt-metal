# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 1a of the Nsight-counters plan: derive the buffer size, never guess it.

The per-zone counter-marker count is a fixed function of the enabled counter
groups and the architecture (perf_counters.hpp / hw_counters.h). These tests
pin that formula so the on-device DRAM marker buffer can be sized to fit a
known workload, and cross-check it against a real device capture when one is
present.
"""

import csv
import glob
import json
import os

import pytest

from tracy.perf_counter_sizing import (
    L1_OPTIONAL_MARKER_BUDGET,
    markers_per_zone,
    recommend_program_support_count,
    single_pass_l1_headroom,
)


def test_fpu_group_is_three_markers_on_blackhole():
    # FPU group = FPU_COUNTER, SFPU_COUNTER, MATH_COUNTER.
    assert markers_per_zone("blackhole", ["fpu"]) == 3


def test_sfpu_alias_matches_fpu():
    assert markers_per_zone("blackhole", ["sfpu"]) == markers_per_zone("blackhole", ["fpu"])


def test_groups_sum_on_blackhole():
    # 3 + 5 + 22 + 59 + 16 = 105 markers/zone on BRISC.
    assert markers_per_zone("blackhole", ["fpu", "pack", "unpack", "instrn", "l1_0"]) == 105


def test_all_expands_to_single_pass_set():
    # "all" = fpu | pack | unpack | l1_0 | instrn (L1 bank 1 needs a 2nd pass).
    assert markers_per_zone("blackhole", ["all"]) == 3 + 5 + 22 + 16 + 59


def test_blackhole_only_groups_have_counters():
    assert markers_per_zone("blackhole", ["l1_2"]) == 16
    # L1 banks 2-4 do not exist on Wormhole.
    assert markers_per_zone("wormhole_b0", ["l1_2"]) == 0


def test_wormhole_pack_differs_from_blackhole():
    assert markers_per_zone("wormhole_b0", ["pack"]) == 14
    assert markers_per_zone("blackhole", ["pack"]) == 5


def test_single_pass_headroom_flags_l1_optional_overflow():
    # A modest group set fits the 250-optional-marker L1 budget.
    assert single_pass_l1_headroom("blackhole", ["fpu", "instrn"]) > 0
    # The budget is the documented L1 optional marker count.
    assert L1_OPTIONAL_MARKER_BUDGET == 250


def test_recommend_program_support_count_covers_workload():
    # 4400 ops/device with headroom must not round below the op count.
    rec = recommend_program_support_count(4400)
    assert rec >= 4400


def test_formula_matches_real_capture():
    """Cross-check the formula against a real fpu capture if one exists."""
    logs = sorted(
        glob.glob("generated/profiler/*/.logs/profile_log_device.csv"),
        key=os.path.getmtime,
        reverse=True,
    )
    capture = None
    for path in logs:
        with open(path) as f:
            head = f.readline()
        if "blackhole" in head.lower():
            capture = path
            break
    if capture is None:
        pytest.skip("no blackhole device capture available to cross-check")

    # Count distinct counter types emitted per (core, zone) for the FPU group.
    per_zone_types = {}
    with open(capture) as f:
        reader = csv.reader(f)
        next(reader, None)  # arch banner
        next(reader, None)  # column header
        for row in reader:
            if len(row) < 15 or row[4].strip() != "9090":
                continue
            meta = row[14]
            if "counter type" not in meta:
                continue
            ctype = meta.split('"counter type":')[1].split(";")[0].strip().strip('"')
            key = (row[1], row[2], row[7])  # core_x, core_y, run_host_id
            per_zone_types.setdefault(key, set()).add(ctype)

    if not per_zone_types:
        pytest.skip("capture has no perf-counter markers")

    fpu_types = {"FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"}
    fpu_only_zones = [t for t in per_zone_types.values() if t <= fpu_types]
    assert fpu_only_zones, "no FPU-only zones found in capture"
    # Every FPU-only zone must carry exactly the formula's marker count.
    assert all(len(t) == markers_per_zone("blackhole", ["fpu"]) for t in fpu_only_zones)
