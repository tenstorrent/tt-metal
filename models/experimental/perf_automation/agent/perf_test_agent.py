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

_WEDGE_GUIDANCE = (
    "The trace capture HUNG the device (it timed out; the harness already reset the device). This is a "
    "DEVICE WEDGE, not a code bug you can fix by editing the trace: this module's forward has an "
    "irreducible host op that a trace cannot record. STOP trying to make the trace work. Make the test "
    "EAGER-ONLY: drop the trace-replay block (or guard it so it does not run) and print "
    "TRACE_NOT_TRACE_CAPABLE=1, so the eager FORWARD_WALL_MS is the result. That returns VERDICT=PASS_EAGER."
)


def _judge_output(rc, out: str) -> str:
    from .perf_test_gen import _is_eager_terminal, _parse_trace_path

    text = out or ""
    if rc in _TIMEOUT_CODES or "WEDGE" in text:
        return "WEDGE"
    has_eager = "FORWARD_WALL_MS=" in text
    traced = ("TRACE_PER_TOKEN_MS=" in text) and bool(_parse_trace_path(text))
    if rc == 0 and traced:
        return "PASS_TRACE"
    if rc == 0 and has_eager and _is_eager_terminal(text):
        return "PASS_EAGER"
    return "FAIL"


def _bound_output(out: str, limit: int = 16000) -> str:
    if not out:
        return "(no output captured)"
    for marker in ("=== FAILURES ===", "=== ERRORS ==="):
        idx = out.rfind(marker)
        if idx != -1:
            return out[idx:][:limit]
    return out[-limit:]


def _run_and_format(node_abs: str) -> str:
    from .perf_test_gen import _run_perf_node

    rc, out = _run_perf_node(
        node_abs, {"TT_PERF_TRACE": "1", "TT_PERF_NUM_CQ": "1"}, timeout_s=_COMPONENT_RUN_TIMEOUT_S
    )
    verdict = _judge_output(rc, out)
    if verdict == "WEDGE":
        eager_note = (
            "\n(the eager FORWARD_WALL_MS already printed — you only need to remove the trace block.)"
            if "FORWARD_WALL_MS=" in (out or "")
            else ""
        )
        return f"VERDICT=WEDGE\nrc={rc}\n{_WEDGE_GUIDANCE}{eager_note}"
    return f"VERDICT={verdict}\nrc={rc}\n----- raw test output -----\n{_bound_output(out)}"


def _make_perf_run_server(node_abs: str):
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except Exception:  # noqa: BLE001
        return None

    @tool(
        "run_perf_test",
        "Run the perf test you have written on the device and return its RAW output. It routes "
        "through the harness runner, which handles ALL device execution and recovery (reset, "
        "cooldown) for you — you must NEVER run pytest/tt-smi/kill or open a device yourself. Read "
        "the returned output: the first line is VERDICT=PASS_TRACE / PASS_EAGER / WEDGE / FAIL. On FAIL "
        "the raw traceback + the input the test built follow — fix your file and call again. On WEDGE the "
        "trace HUNG the device (no fixable error) — go eager-only per the instructions and call again.",
        {},
    )
    async def run_perf_test(args):  # noqa: ANN001
        text = await asyncio.to_thread(_run_and_format, node_abs)
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(PERF_RUN_SERVER, tools=[run_perf_test])


_SYSTEM = (
    "You write ONE single-component performance test for a TTNN model, then stop. Workflow: write "
    "the test file with Write/Edit, RUN it by calling the run_perf_test tool, READ the raw output "
    "(the real traceback AND the input the test built), fix your file, and repeat until run_perf_test "
    "returns VERDICT=PASS_TRACE — or VERDICT=PASS_EAGER when the module's own forward has an "
    "irreducible host op that cannot be traced. If run_perf_test returns VERDICT=WEDGE, the trace HUNG "
    "the device (not a fixable error): immediately go eager-only (drop the trace block, print "
    "TRACE_NOT_TRACE_CAPABLE=1) — do NOT keep editing the trace. Edit ONLY the one perf test file named in the task. "
    "NEVER run pytest, tt-smi, kill, fuser, or open/close a device yourself — the run_perf_test tool "
    "and the harness own all device execution and recovery; doing it yourself breaks the run. Do not "
    "repeat an approach that already failed the same way — change the approach. Keep your final "
    "message to one short line."
)


def build_component_perf_test(root: str | Path, task: str, out_rel: str, prompt_body: str, max_turns: int = 48) -> bool:
    root = Path(root)
    node_abs = f"{root / out_rel}::test_{task}_perf"
    server = _make_perf_run_server(node_abs)
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
        "tool. Then CALL run_perf_test, read its raw output, and iterate (Edit -> run_perf_test) until "
        "it returns VERDICT=PASS_TRACE, or VERDICT=PASS_EAGER if the module genuinely cannot be "
        "trace-captured (VERDICT=WEDGE = the trace hung the device: switch to eager-only immediately, do "
        "not keep editing the trace). Do NOT finish until run_perf_test passes."
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
        return False
    return True
