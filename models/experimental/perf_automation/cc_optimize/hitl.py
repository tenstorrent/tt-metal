# SPDX-License-Identifier: Apache-2.0
"""Human-in-the-loop (--hitl) gate for the cc optimizer.

The agent applies ONE lever, then calls hitl_gate() instead of committing. That MCP tool blocks and
hands a proposal to the orchestrator (run.py, which owns the terminal) over a file handshake — the
perf-mcp subprocess can't touch the operator's stdin/stdout (its streams belong to the agent), so it
posts a proposal file and polls for a decision file. run.py renders a block-level timing + rationale
pause screen, reads commit|revert|try, performs the git action, and posts the decision back. Same
long-lived agent throughout (no re-launch): hitl_gate returns the decision and the agent continues.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_PROPOSAL = "hitl_proposal.json"
_DECISION = "hitl_decision.json"
_VALID_ACTIONS = ("commit", "revert", "try")


def _p(run_dir, name):
    return Path(run_dir) / name


def post_proposal(run_dir, proposal: dict) -> None:
    """Agent side: publish the lever proposal and clear any stale decision, so the next read blocks
    until the operator answers THIS proposal."""
    try:
        _p(run_dir, _DECISION).unlink()
    except OSError:
        pass
    _p(run_dir, _PROPOSAL).write_text(json.dumps(proposal))


def await_decision(run_dir, poll: float = 0.5, timeout: float | None = None) -> dict:
    """Agent side: block until the orchestrator posts a decision (or timeout -> conservative revert so
    an unattended --hitl never silently banks an unreviewed edit)."""
    d = _p(run_dir, _DECISION)
    start = time.monotonic()
    while True:
        if d.exists():
            try:
                dec = json.loads(d.read_text())
                d.unlink()
                if dec.get("action") in _VALID_ACTIONS:
                    return dec
            except (OSError, ValueError):
                pass
        if timeout is not None and (time.monotonic() - start) > timeout:
            return {"action": "revert", "note": "hitl timeout — no operator decision; reverted (nothing banked)"}
        time.sleep(poll)


def read_proposal(run_dir) -> dict | None:
    """Orchestrator side: consume a pending proposal (None if none waiting)."""
    p = _p(run_dir, _PROPOSAL)
    if not p.exists():
        return None
    try:
        prop = json.loads(p.read_text())
        p.unlink()
        return prop
    except (OSError, ValueError):
        return None


def post_decision(run_dir, action: str, note: str = "", knob: str = "") -> None:
    """Orchestrator side: answer the blocked agent."""
    _p(run_dir, _DECISION).write_text(json.dumps({"action": action, "note": note, "knob": knob}))


def _bar(ms, peak, width=22):
    if not peak or peak <= 0:
        return ""
    n = max(0, min(width, round(width * float(ms) / float(peak))))
    return "█" * n + "·" * (width - n)


def render_pause_screen(proposal: dict) -> str:
    """Pure render of the HITL pause screen from a proposal dict. Sections: block-level timing (hottest
    flagged), the lever tried + WHY, the win/no-win result + WHY, the next target + WHY, and the prompt."""
    stages = list(proposal.get("stages") or [])
    tried = proposal.get("tried") or {}
    result = proposal.get("result") or {}
    nxt = proposal.get("next") or {}
    lines = []
    title = "optimize --hitl · %s · step %s" % (proposal.get("model", "?"), proposal.get("step", "?"))
    lines.append(title)
    lines.append("=" * len(title))

    lines.append("BLOCK-LEVEL TIMING (per-stage trace):")
    if stages:
        peak = max((s.get("ms") or 0) for s in stages)
        hot = max(stages, key=lambda s: s.get("ms") or 0)
        for s in stages:
            ms = s.get("ms") or 0
            mark = "  <- hottest" if s is hot else ""
            dom = (" · %s" % s["dominant"]) if s.get("dominant") else ""
            lines.append("  %-10s %9.2f ms  %s%s%s" % (s.get("name", "?"), ms, _bar(ms, peak), dom, mark))
    else:
        lines.append("  (no per-stage timing captured)")
    if proposal.get("timing_note"):
        lines.append("  note: %s" % proposal["timing_note"])

    lines.append("")
    lines.append("TRIED : %s  on  %s" % (tried.get("lever", "?"), tried.get("op", "?")))
    if tried.get("why"):
        lines.append("  why : %s" % tried["why"])

    lines.append("")
    won = result.get("win")
    verdict = "WIN" if won else ("NO WIN" if won is not None else "?")
    before, after = result.get("before_ms"), result.get("after_ms")
    delta = ""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
        delta = "  (%.2f -> %.2f ms, %+.1f%%)" % (before, after, (after - before) / before * 100.0)
    lines.append("RESULT: %s%s" % (verdict, delta))
    checks = result.get("checks")
    if checks:
        lines.append("  %s" % checks)
    if not won and result.get("why_not"):
        lines.append("  why not: %s" % result["why_not"])

    lines.append("")
    lines.append("NEXT (if you proceed): %s" % (nxt.get("target", "?")))
    if nxt.get("why"):
        lines.append("  why : %s" % nxt["why"])

    lines.append("")
    lines.append("  [c] commit   [r] revert   [t] try another knob")
    return "\n".join(lines)
