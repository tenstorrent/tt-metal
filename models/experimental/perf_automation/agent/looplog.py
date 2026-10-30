"""Concise terminal logging for the Agent Loop — one line per stage transition.

No extra packages. Appends a terse line per stage (history preserved while
debugging). TODO(in-place): swap print(...) for end="\\r" to overwrite a single
status line once the loop is trusted and the verbose history is no longer wanted.
"""

from __future__ import annotations

import sys


def _extra(ctx, state: str) -> str:
    st = ctx.state
    d = st.get("last_decision") or {}
    v = st.get("last_verdict") or {}
    if state == "ROUTE":
        return f"bucket={st.get('current_bucket')} candidates={len(st.get('candidates') or [])}"
    if state == "SELECT":
        return f"lever={st.get('selected_lever')}"
    if state == "APPLY":
        return f"edited={len((st.get('last_edit') or {}).get('files', []))} file(s)"
    if state == "VERIFY":
        return v.get("status", "")
    if state == "GATE_PCC":
        return f"{v.get('status','')} pcc={v.get('pcc')}"
    if state == "REMEASURE":
        return f"{d.get('before')}->{d.get('after')} ms (spread {d.get('spread')})"
    if state == "DECIDE":
        return d.get("result", "") + (f" ({d.get('reason')})" if d.get("reason") else "")
    if state == "CHECK_EXIT":
        m = st.get("metric", {})
        return f"{m.get('current')}/{m.get('target')}"
    return ""


def make_logger(stream=sys.stderr):
    def log(ctx, state, next_state):
        it = ctx.state.get("iteration", 0)
        print(f"  [iter {it}] {state:12} -> {next_state:12} {_extra(ctx, state)}", file=stream, flush=True)

    return log
