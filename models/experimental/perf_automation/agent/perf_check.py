"""measure_candidate — let the KERNEL-authoring agent MEASURE its kernel's device time vs the
baseline BEFORE it finishes, so it iterates toward a kernel that is actually FASTER, not merely
correct.

The verified gap (seamless kernel run, 2026-06-21): the kernel lever now authors a real ttl
kernel that passes PCC 0.9999 — but it stops the moment `check_candidate_edit` (correctness)
passes, because it has no SPEED signal. Result: baseline 9.403ms -> kernel 9.397ms (noise). A
correct-but-not-faster kernel is reverted at REMEASURE, wasting the whole iteration. This exposes
the perf measurement as an in-process MCP tool the agent can call: author -> check correct ->
measure speed -> if not faster, change the APPROACH (grid, fidelity, decomposition) -> measure
again -> finish only once it BEATS the baseline.

REUSES measure.measure_runs (the SAME profiler path REMEASURE uses) — does NOT duplicate the
measurement logic — so 'faster here' means the harness's REMEASURE will agree. Runs the profiler
sequentially (one device run at a time), so it cannot deadlock the device. The profiler is
expensive, so this tool is wired ONLY for the kernel lever (knob edits are already measured by
REMEASURE) and the agent is told to use it sparingly — correctness first, then a few speed probes.
"""

from __future__ import annotations

import asyncio
from typing import Callable

PERF_CHECK_TOOL = "mcp__perfcheck__measure_candidate"
PERF_CHECK_SERVER = "perfcheck"

_PROMPT_NOTE = (
    "\n\nMEASURE BEFORE FINISHING: you have a tool `measure_candidate` — it profiles your kernel "
    "on-device and reports its time vs the baseline. A kernel that is correct but NOT faster than "
    "the baseline is WORTHLESS (it gets reverted). Workflow: (1) get correctness PASS via "
    "`check_candidate_edit` first, then (2) call `measure_candidate`. If it is NOT clearly faster "
    "(>2% beyond noise), do NOT finish — change the APPROACH: occupy more of the grid, raise "
    "math-fidelity only where needed, fuse the surrounding eltwise into the kernel, or pick a "
    "better tiling/decomposition — then measure again. Profiling is expensive, so spend most of "
    "your probes on correctness and only measure a candidate you believe is faster. Output your "
    "final JSON only once the kernel is BOTH correct AND faster than the baseline."
)


def format_measure_result(res: dict | None, baseline_ms: float | None) -> str:
    """Turn a {status, device_ms?|error?} into the message the agent sees, framed vs the baseline."""
    status = (res or {}).get("status")
    if status != "ok":
        return (
            "MEASURE FAILED — the kernel crashed or could not be profiled. Confirm it passes "
            "`check_candidate_edit` first, then fix the failure:\n" + str((res or {}).get("error", ""))[-1000:]
        )
    cur = (res or {}).get("device_ms")
    if cur is None:
        return "MEASURE FAILED — no device_ms returned from the profiler."
    if not baseline_ms:
        return f"Measured device_ms={cur:.4f} (no baseline to compare against)."
    delta = baseline_ms - cur
    pct = (delta / baseline_ms) * 100.0 if baseline_ms else 0.0
    if pct > 2.0:
        return (
            f"FASTER — kernel {cur:.4f}ms vs baseline {baseline_ms:.4f}ms ({pct:.1f}% faster). "
            f"This is a real win; you may finish (after confirming correctness)."
        )
    if pct < -2.0:
        return (
            f"SLOWER — kernel {cur:.4f}ms vs baseline {baseline_ms:.4f}ms ({-pct:.1f}% SLOWER). "
            f"This kernel hurts. Change the approach (more grid / better tiling / fuse eltwise) and measure again."
        )
    return (
        f"NO GAIN — kernel {cur:.4f}ms vs baseline {baseline_ms:.4f}ms ({pct:+.1f}%, within noise). "
        f"Correct but not worth keeping. Change the approach to make it materially faster, then measure again."
    )


def make_perf_check_server(measure: Callable[[], dict], baseline_ms: float | None):
    """Build an in-process MCP server exposing `measure_candidate`. `measure()` runs the SAME
    profiler REMEASURE uses and returns {status: ok|crash, device_ms?, error?} (bind it to
    measure.measure_runs + the live ctx). `baseline_ms` frames the result as faster/slower.
    Returns the McpSdkServerConfig (or None if the SDK lacks the in-process MCP API, so callers
    degrade gracefully)."""
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except Exception:  # noqa: BLE001 — older SDK without in-process MCP; caller runs without the tool
        return None

    @tool(
        "measure_candidate",
        "Profile your kernel on-device and report its time vs the baseline (FASTER / SLOWER / NO "
        "GAIN). Expensive — call it only AFTER `check_candidate_edit` passes, on a candidate you "
        "believe is faster. Finish only once the kernel is faster than the baseline.",
        {},
    )
    async def measure_candidate(args):  # noqa: ANN001
        res = await asyncio.to_thread(measure)
        return {"content": [{"type": "text", "text": format_measure_result(res, baseline_ms)}]}

    return create_sdk_mcp_server(PERF_CHECK_SERVER, tools=[measure_candidate])
