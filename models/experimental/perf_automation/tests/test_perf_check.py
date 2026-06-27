# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""measure_candidate — the on-device perf-feedback tool the kernel author calls so it iterates
toward a FASTER kernel, not merely a correct one. Tests the faster/slower/no-gain framing +
that the in-process MCP server builds (reuses REMEASURE's measure_runs)."""

from agent import perf_check


def test_format_faster():
    t = perf_check.format_measure_result({"status": "ok", "device_ms": 8.0}, baseline_ms=10.0)
    assert t.startswith("FASTER") and "20.0%" in t


def test_format_slower_tells_agent_to_change_approach():
    t = perf_check.format_measure_result({"status": "ok", "device_ms": 12.0}, baseline_ms=10.0)
    assert t.startswith("SLOWER") and "20.0%" in t and "approach" in t


def test_format_no_gain_within_noise():
    # 0.1% delta -> within the 2% noise band -> NO GAIN (not a keep)
    t = perf_check.format_measure_result({"status": "ok", "device_ms": 9.99}, baseline_ms=10.0)
    assert t.startswith("NO GAIN") and "approach" in t


def test_format_crash_surfaces_error():
    t = perf_check.format_measure_result({"status": "crash", "error": "tracy profiler overflow"}, baseline_ms=10.0)
    assert t.startswith("MEASURE FAILED") and "tracy" in t


def test_format_no_baseline():
    t = perf_check.format_measure_result({"status": "ok", "device_ms": 9.0}, baseline_ms=None)
    assert "9.0" in t and "no baseline" in t


def test_server_builds_and_tool_name():
    srv = perf_check.make_perf_check_server(lambda: {"status": "ok", "device_ms": 8.0}, baseline_ms=10.0)
    assert srv is not None  # SDK has in-process MCP (create_sdk_mcp_server)
    assert perf_check.PERF_CHECK_TOOL == "mcp__perfcheck__measure_candidate"
    assert perf_check.PERF_CHECK_SERVER == "perfcheck"


def test_prompt_note_directs_measurement_and_iteration():
    note = perf_check._PROMPT_NOTE
    assert "measure_candidate" in note and "faster" in note.lower() and "approach" in note.lower()
