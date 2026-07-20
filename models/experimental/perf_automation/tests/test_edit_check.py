# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""check_candidate_edit — the on-device validation tool the editor calls before submitting.
Tests the result formatting + that the in-process MCP server builds (reuses GATE_PCC's pcc check)."""

from agent import edit_check


def test_format_pass():
    t = edit_check.format_check_result({"status": "ok", "pcc": 0.9991})
    assert t.startswith("PASS") and "0.9991" in t


def test_format_pcc_low():
    t = edit_check.format_check_result({"status": "pcc_low", "pcc": 0.81})
    assert "RUNS" in t and "0.81" in t and "regressed" in t


def test_format_crash_surfaces_error():
    t = edit_check.format_check_result({"status": "crash", "error": "RuntimeError: Invalid core_grid type"})
    assert t.startswith("FAIL") and "Invalid core_grid" in t


def test_format_tolerates_none():
    assert edit_check.format_check_result(None).startswith("FAIL")


def test_server_builds_and_tool_name():
    srv = edit_check.make_edit_check_server(lambda: {"status": "ok", "pcc": 0.999})
    assert srv is not None  # SDK has in-process MCP (create_sdk_mcp_server)
    assert edit_check.EDIT_CHECK_TOOL == "mcp__editcheck__check_candidate_edit"
    assert edit_check.EDIT_CHECK_SERVER == "editcheck"


def test_prompt_note_directs_validation():
    assert "check_candidate_edit" in edit_check._PROMPT_NOTE and "PASS" in edit_check._PROMPT_NOTE
