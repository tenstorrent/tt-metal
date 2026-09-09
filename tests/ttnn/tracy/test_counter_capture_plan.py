# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 5 of the Nsight-counters plan: one-command, repeatable counter capture.

The capture plan must be deterministic and self-documenting: a pinned buffer
size derived from the workload, the exact env applied, and a warning when the
requested groups overflow a single L1 pass. These tests pin that plan logic
(no device needed).
"""

from tracy.capture_counters import build_capture_plan


def test_plan_pins_buffer_to_cover_workload():
    plan = build_capture_plan(groups=["fpu"], programs_per_device=4400, arch="blackhole")
    count = int(plan["env"]["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"])
    assert count >= 4400


def test_plan_sets_counter_groups():
    plan = build_capture_plan(groups=["fpu", "instrn"], programs_per_device=100, arch="blackhole")
    assert plan["groups"] == ["fpu", "instrn"]


def test_all_groups_fit_single_l1_pass_on_blackhole():
    # Honest finding: even all nine BH groups (169 markers) fit the L1 optional
    # budget. The per-zone L1 vector is never the bottleneck on BH; the DRAM
    # program-support count is, which the pinned buffer below covers.
    plan = build_capture_plan(
        groups=["fpu", "pack", "unpack", "instrn", "l1_0", "l1_1", "l1_2", "l1_3", "l1_4"],
        programs_per_device=100,
        arch="blackhole",
    )
    assert plan["l1_headroom"] >= 0
    assert plan["warnings"] == []


def test_plan_threads_compute_core_sample_env():
    plan = build_capture_plan(groups=["fpu"], programs_per_device=100, arch="blackhole", compute_core_sample=3)
    assert plan["env"]["TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE"] == "3"


def test_plan_no_sample_env_when_unset():
    plan = build_capture_plan(groups=["fpu"], programs_per_device=100, arch="blackhole")
    assert "TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE" not in plan["env"]


def test_plan_archive_dir_is_timestamped():
    plan = build_capture_plan(
        groups=["fpu"], programs_per_device=100, arch="blackhole", archive_root="/tmp/traces", timestamp="2026_06_15"
    )
    assert plan["archive_dir"] == "/tmp/traces/2026_06_15"
