"""Agentic single-component perf-test builder.

The one-shot generator + regex error-scraping (`_extract_error`) cannot parse the open-ended error
formats a component perf test hits (python tracebacks, ttnn TT_FATAL, C++ `unordered_map::at`,
unpack/shape errors). This builder instead lets an SDK agent write the test, RUN it through an
in-process tool that routes to `_run_perf_node` (device self-heal + output bounding live there, and
the agent never touches the device), READ the raw output itself, and iterate until it passes or the
module is genuinely eager-only. The agent's own conversation carries what it built and what it
already tried, so it does not repeat dead ends.

Returns False (never raises) when the SDK lacks the in-process MCP API or the env is not wired, so
the caller degrades to the one-shot generator.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

PERF_RUN_SERVER = "perfrun"
PERF_RUN_TOOL = "mcp__perfrun__run_perf_test"

_COMPONENT_RUN_TIMEOUT_S = int(os.environ.get("PERF_MCP_COMPONENT_RUN_TIMEOUT_S", "240"))
_TIMEOUT_CODES = {124, 137, 143, -9, -15}

_WEDGE_RETRY_GUIDANCE = (
    "The trace capture HUNG the device (timed out; the harness reset it). The code BETWEEN "
    "ttnn.begin_trace_capture and ttnn.end_trace_capture touched the HOST — a trace records only pure "
    "device-op dispatch. RESTRUCTURE so the captured region is host-free: build the module and ALL its "
    "inputs / masks / constants ONCE, resident on device, BEFORE begin_trace_capture; nothing between "
    "begin/end may call ttnn.from_torch / ttnn.to_torch / .item() / .cpu() / torch construction / python "
    "shape or control-flow. If one op inside is the host bit, move it OUT of the captured region or "
    "replace it with a device-only equivalent. Then call run_perf_test again to keep attempting the "
    "TRACE. There is NO eager fallback — eager is only ever used when the operator sets TT_PERF_TRACE=0."
)


def _eager_flag() -> bool:
    return os.environ.get("TT_PERF_TRACE") == "0"


def _judge_output(rc, out: str) -> str:
    from .perf_test_gen import _parse_trace_path

    text = out or ""
    if _eager_flag():
        return "PASS_EAGER" if (rc == 0 and "FORWARD_WALL_MS=" in text) else "FAIL"
    if rc in _TIMEOUT_CODES or "WEDGE" in text:
        return "WEDGE"
    traced = ("TRACE_PER_TOKEN_MS=" in text) and bool(_parse_trace_path(text))
    if rc == 0 and traced:
        return "PASS_TRACE"
    return "FAIL"


def _bound_output(out: str, limit: int = 16000) -> str:
    if not out:
        return "(no output captured)"
    for marker in ("=== FAILURES ===", "=== ERRORS ==="):
        idx = out.rfind(marker)
        if idx != -1:
            return out[idx:][:limit]
    return out[-limit:]


def _run_and_format(node_abs: str, state: dict | None = None) -> str:
    from .perf_test_gen import _run_perf_node

    if state is None:
        state = {"wedges": 0, "passed": False}
    env = {"TT_PERF_NUM_CQ": "1", "TT_PERF_TRACE": "0" if _eager_flag() else "1"}
    rc, out = _run_perf_node(node_abs, env, timeout_s=_COMPONENT_RUN_TIMEOUT_S)
    verdict = _judge_output(rc, out)
    if verdict == "WEDGE":
        state["wedges"] += 1
        return f"VERDICT=WEDGE\nrc={rc}\n(trace hang {state['wedges']}) {_WEDGE_RETRY_GUIDANCE}"
    if verdict in ("PASS_TRACE", "PASS_EAGER"):
        state["passed"] = True
    return f"VERDICT={verdict}\nrc={rc}\n----- raw test output -----\n{_bound_output(out)}"


def _make_perf_run_server(node_abs: str, state: dict):
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except Exception:  # noqa: BLE001
        return None

    @tool(
        "run_perf_test",
        "Run the perf test you have written on the device and return its RAW output. It routes "
        "through the harness runner, which handles ALL device execution and recovery (reset, "
        "cooldown) for you — you must NEVER run pytest/tt-smi/kill or open a device yourself. Read "
        "the returned output: the first line is VERDICT=PASS_TRACE / WEDGE / FAIL. On FAIL the raw "
        "traceback + the input the test built follow — fix and call again. On WEDGE the trace hung — "
        "restructure the captured region to be host-free and keep attempting the trace. There is NO "
        "eager fallback; PASS_TRACE is the only success (eager mode exists only when the operator sets "
        "TT_PERF_TRACE=0, which is outside your control).",
        {},
    )
    async def run_perf_test(args):  # noqa: ANN001
        text = await asyncio.to_thread(_run_and_format, node_abs, state)
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(PERF_RUN_SERVER, tools=[run_perf_test])


_SYSTEM = (
    "You write ONE single-component performance test for a TTNN model, then stop. Workflow: write "
    "the test file with Write/Edit, RUN it by calling the run_perf_test tool, READ the raw output "
    "(the real traceback AND the input the test built), fix your file, and repeat until run_perf_test "
    "returns VERDICT=PASS_TRACE — the ONLY success. VERDICT=WEDGE means the trace hung because the "
    "captured region touched the host: RESTRUCTURE it to be host-free (build inputs/constants resident "
    "once before the capture) and keep attempting the trace. There is NO eager fallback — do NOT print "
    "TRACE_NOT_TRACE_CAPABLE or drop the trace block; eager mode is only ever enabled by the operator via "
    "TT_PERF_TRACE=0 and is outside your control. Edit ONLY the one perf test file named in the task. "
    "NEVER run pytest, tt-smi, kill, fuser, or open/close a device yourself — the run_perf_test tool "
    "and the harness own all device execution and recovery; doing it yourself breaks the run. Do not "
    "repeat an approach that already failed the same way — change the approach. Keep your final "
    "message to one short line."
)


def build_component_perf_test(root: str | Path, task: str, out_rel: str, prompt_body: str, max_turns: int = 48) -> bool:
    root = Path(root)
    node_abs = f"{root / out_rel}::test_{task}_perf"
    state = {"wedges": 0, "passed": False}
    server = _make_perf_run_server(node_abs, state)
    if server is None:
        return False
    try:
        from .config import agent_effort, apply_agent_env, get_edit_model

        resolved = apply_agent_env(Path(__file__).parent.parent / ".env.agent")
    except Exception:  # noqa: BLE001
        return False

    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    from .sdk_retry import run_with_retry

    try:
        from .structural_agent import _DEVICE_CALL_TIMEOUT_S as _agent_timeout
    except Exception:  # noqa: BLE001
        _agent_timeout = None

    prompt = (
        prompt_body + f"\n\nWrite the test file at `{out_rel}` (relative to your working directory) with the Write "
        "tool. Then CALL run_perf_test, read its raw output, and iterate (Edit -> run_perf_test) until it "
        "returns VERDICT=PASS_TRACE — the ONLY success. On VERDICT=WEDGE the trace hung — restructure the "
        "captured region to be host-free and keep attempting the trace; there is no eager fallback. Do NOT "
        "finish until run_perf_test returns PASS_TRACE."
    )
    opts = ClaudeAgentOptions(
        model=get_edit_model(0, resolved),
        system_prompt=_SYSTEM,
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", PERF_RUN_TOOL],
        permission_mode="bypassPermissions",
        setting_sources=[],
        mcp_servers={PERF_RUN_SERVER: server},
        max_turns=max_turns,
        max_buffer_size=50 * 1024 * 1024,
        effort=agent_effort(resolved),
        cwd=str(root),
    )
    chunks: list[str] = []

    async def _go() -> None:
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)

    try:
        run_with_retry(_go, lambda: chunks.clear(), timeout=_agent_timeout)
    except Exception:  # noqa: BLE001
        return bool(state.get("passed"))
    return bool(state.get("passed"))
