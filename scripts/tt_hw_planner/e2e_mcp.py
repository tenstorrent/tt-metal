# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""e2e-mcp — an external stdio MCP server that exposes the emit-e2e DETERMINISTIC gates (G1–G4) to a
free-roaming Claude Code agent, so `emit-e2e --engine cc` can drive the build→gate→fix loop through
the shared cc harness.

It REUSES ``commands.emit_e2e._run_deterministic_gates`` verbatim — the gate logic is NOT
reimplemented or altered here; this file only adapts its ``(ok, reasons)`` return into the harness's
``termination_check`` verdict shape. Removing this file fully reverts.

Config via env (set in the --mcp-config):
  E2E_MCP_DEMO_DIR   demo dir to gate (required)
  E2E_MCP_PCC        required e2e PCC threshold (default 0.99)
  E2E_MCP_TIMEOUT    per-gate pytest timeout seconds (default 1800)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from scripts.tt_hw_planner.commands.emit_e2e import _run_deterministic_gates  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("e2e-mcp")

_DEMO_DIR = os.environ.get("E2E_MCP_DEMO_DIR", "")
_PCC = float(os.environ.get("E2E_MCP_PCC", "0.99"))
_TIMEOUT = int(os.environ.get("E2E_MCP_TIMEOUT", "1800"))


@mcp.tool()
def termination_check() -> dict:
    """THE deterministic stop gate for emit-e2e. Runs G1–G4 (native stubs, tests/e2e pass on device,
    PCC>=threshold, demo/ structure) via the SAME `_run_deterministic_gates` the legacy path uses, and
    reports the failing gates as `next_target` so the agent fixes exactly those. `can_stop` is true
    ONLY when zero gate reasons remain. The agent may NOT declare done — this gate is the authority."""
    if not _DEMO_DIR:
        return {"can_stop": False, "halt": True, "halt_reason": "E2E_MCP_DEMO_DIR not set", "next_target": None}
    ok, reasons = _run_deterministic_gates(Path(_DEMO_DIR), _PCC, _TIMEOUT)
    return {
        "can_stop": bool(ok),
        "halt": False,
        "halt_reason": None,
        "blocking": reasons,
        "next_target": (
            None
            if ok
            else {"unit": "e2e_gates", "rung": "fix", "reason": " | ".join(reasons)[:2000]}
        ),
    }


if __name__ == "__main__":
    mcp.run()
