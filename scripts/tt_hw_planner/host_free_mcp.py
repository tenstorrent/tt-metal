# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""host-free-mcp — the deterministic stop gate for emit-e2e PHASE 4 (`--host-free`).

Model-agnostic: it does not know any architecture. It runs ``_trace_capture_probe`` (a generic scan for
the host ops that block a device trace — per-layer weight streaming, host token feed, missing KV cache,
no fixed-shape decode_step) and reports the next unmet rung. The expensive checks are bookended: the
cheap static probe drives the ladder every round, and the full correctness gate
(``_run_deterministic_gates``) runs ONLY once the decode looks trace-ready, to confirm no host-free edit
regressed PCC. Removing this file + the PHASE-4 hook fully reverts.

Config via env (set in the --mcp-config), shared with e2e-mcp:
  E2E_MCP_DEMO_DIR   demo dir to gate (required)
  E2E_MCP_PCC        required e2e PCC threshold
  E2E_MCP_TIMEOUT    per-gate pytest timeout seconds
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_THP = Path(__file__).resolve().parent
_REPO = _THP.parents[1]
sys.path.insert(0, str(_REPO))

from scripts.tt_hw_planner.commands.emit_e2e import _run_deterministic_gates  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("host-free-mcp")

_DEMO_DIR = os.environ.get("E2E_MCP_DEMO_DIR", "")
_PCC = float(os.environ.get("E2E_MCP_PCC", "0.99"))
_TIMEOUT = int(os.environ.get("E2E_MCP_TIMEOUT", "1800"))


def _run_probe(demo_dir: Path) -> dict:
    probe = _THP / "_trace_capture_probe.py"
    try:
        r = subprocess.run(
            [sys.executable, str(probe), str(demo_dir)],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(_REPO),
        )
    except Exception as e:  # noqa: BLE001
        return {"trace_ready": False, "static_blockers": [{"rung": "probe", "guidance": "probe failed: %s" % e}]}
    for line in (r.stdout or "").splitlines():
        if line.startswith("TRACE_PROBE="):
            try:
                return json.loads(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                break
    tail = (r.stderr or r.stdout or "")[-400:]
    return {"trace_ready": False, "static_blockers": [{"rung": "probe", "guidance": "no probe output: %s" % tail}]}


@mcp.tool()
def termination_check() -> dict:
    """THE stop gate for emit-e2e PHASE 4 (host-free / trace-capturable decode). Reports can_stop ONLY
    when the decode is trace-capturable (no per-layer weight streaming, no host token loop, KV cache
    present, fixed-shape decode_step) AND e2e correctness (PCC) still passes. next_target names the next
    unmet rung with GENERIC guidance derived from the host ops actually present — the ladder adapts to
    any model (dense / MoE / SSM); it is not a per-model template. The agent may NOT declare done."""
    if not _DEMO_DIR:
        return {"can_stop": False, "halt": True, "halt_reason": "E2E_MCP_DEMO_DIR not set", "next_target": None}
    demo = Path(_DEMO_DIR)
    p = _run_probe(demo)
    blockers = p.get("static_blockers") or []
    if not p.get("trace_ready"):
        if blockers:
            nt = blockers[0]
        else:
            reason = ((p.get("device_capture") or {}) or {}).get("reason", "trace capture did not succeed")
            nt = {"rung": "trace_capture", "guidance": reason}
        return {
            "can_stop": False,
            "halt": False,
            "halt_reason": None,
            "blocking": [b.get("rung") for b in blockers],
            "next_target": {"unit": "decode", "rung": nt.get("rung"), "reason": nt.get("guidance", "")[:2000]},
        }
    ok, reasons = _run_deterministic_gates(demo, _PCC, _TIMEOUT)
    if not ok:
        return {
            "can_stop": False,
            "halt": False,
            "halt_reason": None,
            "blocking": reasons,
            "next_target": {
                "unit": "correctness",
                "rung": "pcc_regression",
                "reason": (
                    "a host-free edit regressed correctness — restore PCC WITHOUT reintroducing host ops: "
                    + " | ".join(reasons)
                )[:2000],
            },
        }
    return {"can_stop": True, "halt": False, "halt_reason": None, "blocking": [], "next_target": None}


if __name__ == "__main__":
    mcp.run()
