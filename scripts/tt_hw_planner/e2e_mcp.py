# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""e2e-mcp — the SINGLE combined deterministic stop gate for emit-e2e.

can_stop is true ONLY when BOTH pass, checked in order:
  (1) CORRECTNESS — G1-G4 + e2e PCC>=threshold, via the SAME `_run_deterministic_gates` (UNCHANGED,
      reused verbatim). Tool-run, NOT agent-reported: the tool runs tests/e2e and measures PCC itself,
      so the agent cannot self-declare done, fake the number, or xfail/skip past it.
  (2) HOST-FREE — the pipeline is everything-on-device / trace-capturable (no per-layer weight
      streaming, no host token loop, real ttnn.begin_trace_capture succeeds), via `_trace_capture_probe`
      (cheap static ladder + a real device capture). Model-agnostic (class-aware, no per-model logic).

Correctness runs FIRST every round; host-free is only checked once correct — so any edit that regresses
PCC is caught the next round before host-free progress is accepted. This merges the old PHASE 3
(correctness) and PHASE 4 (host-free) into one gate: no build-then-teardown. Host-free is required
unless E2E_SKIP_HOST_FREE=1 (correctness-only escape for a genuinely host-bound model).

Config via env (set in the --mcp-config):
  E2E_MCP_DEMO_DIR   demo dir to gate (required)
  E2E_MCP_PCC        required e2e PCC threshold (default 0.99)
  E2E_MCP_TIMEOUT    per-gate pytest timeout seconds (default 1800)
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

mcp = FastMCP("e2e-mcp")

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
    """THE combined stop gate for emit-e2e. can_stop=true ONLY when BOTH pass: (1) CORRECTNESS — G1-G4 +
    e2e PCC>=threshold via the SAME `_run_deterministic_gates` (tool-run, not agent-reported — the agent
    cannot self-declare done, fake PCC, or xfail/skip past it), AND (2) HOST-FREE — everything-on-device
    / trace-capturable (no weight streaming, no host token loop, real begin_trace_capture succeeds) so
    trace + 2CQ can run. Checked in order: correctness first, host-free only once correct. next_target
    names the single next failing thing (a correctness gate OR a host op). Host-free required unless
    E2E_SKIP_HOST_FREE=1. The agent may NOT declare done — this gate is the authority."""
    if not _DEMO_DIR:
        return {"can_stop": False, "halt": True, "halt_reason": "E2E_MCP_DEMO_DIR not set", "next_target": None}
    demo = Path(_DEMO_DIR)

    ok, reasons = _run_deterministic_gates(demo, _PCC, _TIMEOUT)
    if not ok:
        return {
            "can_stop": False,
            "halt": False,
            "halt_reason": None,
            "blocking": reasons,
            "next_target": {"unit": "e2e_gates", "rung": "correctness", "reason": " | ".join(reasons)[:2000]},
        }

    if os.environ.get("E2E_SKIP_HOST_FREE") == "1":
        return {"can_stop": True, "halt": False, "halt_reason": None, "blocking": [], "next_target": None}

    p = _run_probe(demo)
    if not p.get("trace_ready"):
        blockers = p.get("static_blockers") or []
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
            "correctness_ok": True,
            "next_target": {"unit": "decode", "rung": nt.get("rung"), "reason": nt.get("guidance", "")[:2000]},
        }

    return {"can_stop": True, "halt": False, "halt_reason": None, "blocking": [], "next_target": None}


if __name__ == "__main__":
    mcp.run()
