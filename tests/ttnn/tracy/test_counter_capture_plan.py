# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capture plans match the actual CLI passes and reserve serialized packet bytes."""

import json

import pytest

from tracy.capture_counters import _archive_artifacts, build_capture_plan
from tracy.perf_counter_multipass import resolve_perf_counter_groups, schedule_perf_counter_passes
from tracy.perf_counter_sizing import PROGRAM_SUPPORT_BYTES


def test_plan_pins_buffer_for_counter_packets_not_only_op_count():
    plan = build_capture_plan(groups=["fpu", "instrn"], programs_per_device=20, arch="blackhole")
    count = int(plan["env"]["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"])
    assert count == 832
    assert plan["programs_between_drains"] == 20
    assert plan["env"]["TT_METAL_DEVICE_ARCH"] == "blackhole"
    assert plan["passes"][0]["counter_records_per_zone"] == 62
    assert plan["passes"][0]["counter_marker_slots_per_zone"] == 186
    assert plan["l1_headroom"] == 48


@pytest.mark.parametrize("arch,pass_count", [("blackhole", 6), ("wormhole_b0", 2)])
def test_all_uses_cli_schedule_and_covers_every_pass(arch, pass_count):
    plan = build_capture_plan(groups=["all"], programs_per_device=100, arch=arch)
    expected = schedule_perf_counter_passes(resolve_perf_counter_groups(["all"], arch))
    assert [p["groups"] for p in plan["passes"]] == expected
    assert len(plan["passes"]) == pass_count
    assert plan["program_support_count"] == max(p["program_support_count"] for p in plan["passes"])
    for p in plan["passes"]:
        assert len(p["groups"]) <= 3
        assert sum(g.startswith("l1_") for g in p["groups"]) <= 1
        needed = 100 * p["estimated_bytes_per_program_per_risc"] * plan["safety"]
        assert plan["program_support_count"] * PROGRAM_SUPPORT_BYTES >= needed


def test_plan_threads_compute_core_sample_env():
    plan = build_capture_plan(groups=["fpu"], programs_per_device=100, arch="blackhole", compute_core_sample=3)
    assert plan["env"]["TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE"] == "3"
    plan = build_capture_plan(groups=["fpu"], programs_per_device=100, arch="blackhole")
    assert "TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE" not in plan["env"]


def test_empty_or_unknown_groups_fail_closed():
    for groups in ([], ["fpu", "typo"], ["all", "typo"]):
        with pytest.raises(ValueError):  # allow-pytest.raises: no device conftest
            build_capture_plan(groups=groups, programs_per_device=10, arch="blackhole")


def test_archived_plan_preserves_pass_and_sizing_assumptions(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    csv = report / "ops.csv"
    csv.write_text("counter data\n")
    (report / "capture.tracy").write_bytes(b"trace")
    plan = build_capture_plan(["all"], 20, "blackhole", archive_root=str(tmp_path / "archive"), timestamp="run1")
    _archive_artifacts(str(csv), plan["archive_dir"], plan)
    archived = tmp_path / "archive/run1"
    assert json.loads((archived / "capture_plan.json").read_text()) == plan
    assert (archived / "ops.csv").read_text() == "counter data\n"
    assert (archived / "capture.tracy").read_bytes() == b"trace"


def test_profiler_wrapper_delegates_once_to_cli_scheduler(monkeypatch):
    from tracy import process_model_log

    calls = []
    monkeypatch.setattr(process_model_log.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    process_model_log.run_device_profiler(
        "pytest case.py", "capture", capture_perf_counters_groups=["fpu", "l1_0", "l1_1", "l1_5"], op_support_count=832
    )
    assert len(calls) == 1, "legacy two-pass wrapper must not repeat the CLI's complete pass plan"
    command = calls[0][0][0][0]
    assert "--profiler-capture-perf-counters=fpu,l1_0,l1_1,l1_5 --perf-counter-multipass" in command
    assert "--op-support-count 832" in command
