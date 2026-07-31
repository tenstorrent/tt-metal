"""perftest-mcp — an EXTERNAL stdio MCP server that exposes the single-component perf-test RUN tool
to the claude-code CLI (`claude -p`) that AUTHORS the test.

Twin of perf_mcp.py: FastMCP over stdio, spawned by claude via --mcp-config, NO claude SDK. The
authoring agent calls run_perf_test; all device execution + self-heal live in agent.perf_test_gen
(reached through agent.perf_test_agent._run_and_format), so the agent never touches the device. The
target test node and a status-file path arrive via env (PERF_TEST_NODE / PERF_TEST_STATUS_FILE); the
success flag is persisted to the status file after every run so the parent can read the verdict once
the agent exits.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent  # the perf_automation dir
sys.path.insert(0, str(_PKG.parent.parent.parent))  # repo root, so `models...` imports resolve
sys.path.insert(0, str(_PKG))  # the perf_automation dir, so `agent` imports resolve

from agent.perf_test_agent import _run_and_format  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("perftest-mcp")

_NODE = os.environ.get("PERF_TEST_NODE", "")
_STATUS_FILE = os.environ.get("PERF_TEST_STATUS_FILE", "")
_STATE = {"wedges": 0, "passed": False}


def _persist() -> None:
    if not _STATUS_FILE:
        return
    try:
        Path(_STATUS_FILE).write_text(json.dumps({"passed": bool(_STATE.get("passed"))}))
    except OSError:
        pass


@mcp.tool()
def run_perf_test() -> str:
    """Run the perf test you have written on the device and return its RAW output. It routes through
    the harness runner, which handles ALL device execution and recovery (reset, cooldown) for you — you
    must NEVER run pytest/tt-smi/kill or open a device yourself. Read the returned output: the first line
    is VERDICT=PASS_TRACE / WEDGE / FAIL. On FAIL the raw traceback + the input the test built follow —
    fix and call again. On WEDGE the trace hung — restructure the captured region to be host-free and
    keep attempting the trace. There is NO eager fallback; PASS_TRACE is the only success (eager mode
    exists only when the operator sets TT_PERF_TRACE=0, which is outside your control)."""
    text = _run_and_format(_NODE, _STATE)
    _persist()
    return text


if __name__ == "__main__":
    mcp.run()
