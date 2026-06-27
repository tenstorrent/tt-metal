"""check_candidate_edit — let the edit/structural agent VALIDATE its edit on-device BEFORE it
finishes, instead of submitting blind and learning only via an expensive post-hoc profile.

The verified gap: the authoring agents have rich SIGHT (roofline gap, op->source, shapes) but no
way to TEST a candidate edit during authoring — so they submit illegal configs (e.g. an invalid
core_grid) that crash the forward, and only find out after a full device run. This exposes the
correctness check as an in-process MCP tool the agent can call: write -> check -> fix -> check ->
finish only when it passes.

REUSES pcc_runner.run_pcc (the SAME check GATE_PCC uses) — does NOT duplicate the test-running
logic — so 'passes here' means the harness's gate will agree. Runs sequentially (no concurrent
profiler during APPLY), so it cannot deadlock the device.
"""

from __future__ import annotations

import asyncio
from typing import Callable

EDIT_CHECK_TOOL = "mcp__editcheck__check_candidate_edit"
EDIT_CHECK_SERVER = "editcheck"

_PROMPT_NOTE = (
    "\n\nVALIDATE BEFORE FINISHING: you have a tool `check_candidate_edit` — call it to run the "
    "model's correctness test on the edit you've made so far (a forward pass + PCC, no profiling). "
    "If it returns FAIL (e.g. an illegal core_grid / a crash), FIX the error and call it again; "
    "only output your final JSON once it PASSES. This is cheap insurance — do NOT submit an "
    "unvalidated edit."
)


def format_check_result(res: dict | None) -> str:
    """Turn a pcc_runner-style {status, pcc?, error?} into the message the agent sees."""
    status = (res or {}).get("status")
    if status == "ok":
        return f"PASS — the edit runs and is correct (pcc={(res or {}).get('pcc')})."
    if status == "pcc_low":
        return f"RUNS but PCC too low ({(res or {}).get('pcc')}): correctness regressed — adjust the edit."
    return (
        "FAIL — the edit crashes / does not run. Fix this, then check again:\n"
        + str((res or {}).get("error", ""))[-1200:]
    )


def make_edit_check_server(validate: Callable[[], dict]):
    """Build an in-process MCP server exposing `check_candidate_edit`. `validate()` runs the
    model's correctness check and returns {status: ok|pcc_low|crash, pcc?, error?} (bind it to
    pcc_runner.run_pcc + the live ctx). Returns the McpSdkServerConfig (or None if the SDK lacks
    the in-process MCP API, so callers degrade gracefully)."""
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except Exception:  # noqa: BLE001 — older SDK without in-process MCP; caller runs without the tool
        return None

    @tool(
        "check_candidate_edit",
        "Validate the edit you've made SO FAR: runs the model's correctness test on device "
        "(forward pass + PCC, NO profiling). Returns PASS, or FAIL with the error. Call this "
        "before finishing; if it FAILS, fix the error and call again — only finish on PASS.",
        {},
    )
    async def check_candidate_edit(args):  # noqa: ANN001
        res = await asyncio.to_thread(validate)
        return {"content": [{"type": "text", "text": format_check_result(res)}]}

    return create_sdk_mcp_server(EDIT_CHECK_SERVER, tools=[check_candidate_edit])
