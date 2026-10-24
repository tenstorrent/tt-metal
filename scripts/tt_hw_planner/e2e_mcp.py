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
  E2E_MCP_PCC        required e2e PCC threshold (default: pcc_targets.E2E_PCC)
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

from scripts.tt_hw_planner.commands.emit_e2e import (  # noqa: E402
    E2E_MCP_BATCH_ENV,
    _recover_if_wedged,
    _run_deterministic_gates,
    _source_fingerprint,
    _thermal_step,
    run_stamp,
)
from scripts.tt_hw_planner.pcc_targets import E2E_PCC  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP  # noqa: E402
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402

mcp = FastMCP("e2e-mcp")

# THE CLIENT MUST BE ABLE TO TELL WORK FROM A WEDGE. These gates hold the device for a full pytest
# run while emitting nothing, and a client aborts a call that "sent no response or progress" for its
# silence window -- a window this server's own per-gate timeout sits right on top of. Installed once
# at registration, so no gate signature changes and a new one cannot forget.
try:
    from .mcp_progress import install as _install_progress

    _install_progress(mcp)
except Exception:  # noqa: BLE001 -- a server that cannot report progress must still serve
    try:
        from scripts.tt_hw_planner.mcp_progress import install as _install_progress

        _install_progress(mcp)
    except Exception:  # noqa: BLE001
        pass

_DEMO_DIR = os.environ.get("E2E_MCP_DEMO_DIR", "")
_PCC = float(os.environ.get("E2E_MCP_PCC", str(E2E_PCC)))
_TIMEOUT = int(os.environ.get("E2E_MCP_TIMEOUT", "1800"))
_BATCH = int(os.environ.get(E2E_MCP_BATCH_ENV, "1") or "1")  # the emit-e2e --batch request

# A ROUND THAT CHANGED NOTHING MUST NOT BE BOUGHT TWICE.
#
# The loop's contract is: the gate says what is failing, the agent edits code, the gate re-checks.
# When the agent edits nothing, the next round spends the whole gate again to re-derive the same
# verdict. A Qwen-Image-Edit bring-up did exactly that -- rounds of ~3.5 h each, the same blocker
# reported every time, the pipeline byte-identical throughout -- and nothing in the loop noticed,
# because nothing compared the code between rounds. "Retry" without a change is not a retry.
#
# So each DRIVER-side failing check records the demo's source fingerprint (`_source_fingerprint`, the
# same content walk the correctness cache keys on -- one definition, not two). An identical
# fingerprint on the next driver-side failing check is proof the round in between changed no code the
# gate can see. The count is put in the reason, so the agent is TOLD it is repeating itself, and at
# `_NOOP_LIMIT_ENV` consecutive such rounds the gate halts rather than buy the same answer again.
# The default tolerates one -- a round spent reading the code is legitimate -- and 0 disables it.
#
# DRIVER-SIDE ONLY. The agent calls this same tool mid-round, before it has edited anything, and
# counting those calls would halt a round that was about to do the work. Only the stdio server runs
# `mcp.run()`, so that is what distinguishes the two callers.
_NOOP_STATE_FILE = ".e2e_unchanged_rounds.json"
_NOOP_LIMIT_ENV = "E2E_NOOP_ROUND_LIMIT"
_NOOP_ROUND_LIMIT = 2
_NOOP_QUOTE_CHARS = 200
_SERVING = False


def _noop_round_limit() -> int:
    """Consecutive no-edit rounds tolerated before the gate halts. 0 disables the halt."""
    try:
        return max(0, int(os.environ.get(_NOOP_LIMIT_ENV, str(_NOOP_ROUND_LIMIT))))
    except ValueError:
        return _NOOP_ROUND_LIMIT


def _count_unchanged_round(demo_dir: Path) -> int:
    """Record this driver-side failure; return how many in a row saw byte-identical sources."""
    if _SERVING:
        return 0  # the agent's own mid-round call: it has not had its turn to edit yet
    fp = _source_fingerprint(demo_dir)
    if not fp:
        return 0  # no fingerprint -> no evidence either way -> never accuse
    # SCOPED TO THIS RUN. "Nothing changed since the last round" is a statement about this run's
    # rounds; a restart can have any number of reasons that are not the agent sitting still, and a
    # count carried over from the previous run would halt the new one for the old one's inaction.
    stamp = run_stamp()
    path = demo_dir / _NOOP_STATE_FILE
    try:
        prev = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - unreadable state is simply no history
        prev = {}
    same = prev.get("fingerprint") == fp and prev.get("run") == stamp
    unchanged = (int(prev.get("unchanged") or 0) + 1) if same else 0
    try:
        path.write_text(json.dumps({"fingerprint": fp, "run": stamp, "unchanged": unchanged}))
    except OSError:
        pass
    return unchanged


def _unchanged_round_note(unchanged: int, blockers: list) -> str:
    """Say it in the reason field: this report is not new, and nothing was edited after the last one."""
    return (
        "NO CODE CHANGED: the last %d round(s) ended with this demo's sources byte-identical to the "
        "round before, and the gate is still failing on: %s. What follows was already delivered and "
        "nothing was edited after it. Edit the code these blockers name -- or, if they cannot be "
        "acted on, say what is missing instead of re-running."
        % (unchanged, (blockers[0] if blockers else "the same blocker")[:_NOOP_QUOTE_CHARS])
    )


def _run_probe(demo_dir: Path) -> dict:
    probe = _THP / "_trace_capture_probe.py"
    try:
        r = _thermal_step(
            "trace-capture probe",
            lambda: subprocess.run(
                [sys.executable, str(probe), str(demo_dir)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                cwd=str(_REPO),
            ),
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
    _rst = _recover_if_wedged((r.stdout or "") + "\n" + (r.stderr or ""))
    if _rst:
        tail = "%s [%s]" % (tail, _rst)
    return {"trace_ready": False, "static_blockers": [{"rung": "probe", "guidance": "no probe output: %s" % tail}]}


def _failed(blockers: list, target: dict, demo_dir: Path) -> dict:
    """The gate's failing verdict. ONE builder, so the no-edit check cannot be wired into some
    branches and forgotten in others, and so `reason` is assembled the same way whatever failed."""
    blockers = [b for b in blockers if b]
    unchanged = _count_unchanged_round(demo_dir)
    limit = _noop_round_limit()
    note = _unchanged_round_note(unchanged, blockers) if unchanged else ""
    # The note goes FIRST: `reason` is capped, and "you already had this" outranks a repeat of it.
    reason = " | ".join(([note] if note else []) + blockers)[:2000]
    halting = bool(note) and bool(limit) and unchanged >= limit
    return {
        "can_stop": False,
        "halt": halting,
        "halt_reason": note if halting else None,
        "blocking": blockers,
        "next_target": {**target, "reason": reason},
    }


@mcp.tool()
def termination_check() -> dict:
    """THE combined stop gate for emit-e2e. can_stop=true ONLY when BOTH pass: (1) CORRECTNESS — G1-G4 +
    e2e PCC>=threshold via the SAME `_run_deterministic_gates` (tool-run, not agent-reported — the agent
    cannot self-declare done, fake PCC, or xfail/skip past it), AND (2) HOST-FREE — everything-on-device
    / trace-capturable (no weight streaming, no host token loop, real begin_trace_capture succeeds) so
    trace can run. Checked in order: correctness first, host-free only once correct. next_target
    names the single next failing thing (a correctness gate OR a host op). Host-free required unless
    E2E_SKIP_HOST_FREE=1. The agent may NOT declare done — this gate is the authority."""
    if not _DEMO_DIR:
        return {"can_stop": False, "halt": True, "halt_reason": "E2E_MCP_DEMO_DIR not set", "next_target": None}
    demo = Path(_DEMO_DIR)

    os.environ.pop("E2E_REQUIRE_ON_DEVICE", None)

    ok, reasons = _run_deterministic_gates(demo, _PCC, _TIMEOUT, batch=_BATCH)
    if not ok:
        return _failed(reasons, {"unit": "e2e_gates", "rung": "correctness"}, demo)

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
        out = _failed([nt.get("guidance", "")], {"unit": "decode", "rung": nt.get("rung")}, demo)
        out["blocking"] = [b.get("rung") for b in blockers]
        out["correctness_ok"] = True
        return out

    return {"can_stop": True, "halt": False, "halt_reason": None, "blocking": [], "next_target": None}


if __name__ == "__main__":
    _SERVING = True  # see _count_unchanged_round: only the agent's server reaches this line
    mcp.run()
