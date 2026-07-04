"""SELECT handler (PLAN 8.3) — REAL. The lead agent's one judgment: pick a lever.

A query() stage (not part of ROUTE): reads ROUTE's brief, commits ONE untried
lever id from the closed candidate list. The picker is injectable
(ctx.deps["select_runner"]) so this tests without a key; default is the live
LEAD-model picker. Any invalid pick or API error falls back to untried[0].

In:  ctx.state["candidates"], ["tried"], ["route_brief_id"].
Out: ctx.state["selected_lever"], ["select_reasoning"]; counters reset. -> APPLY
"""

from __future__ import annotations

import json

from .. import states


def select(ctx) -> str:
    candidates = ctx.state.get("candidates") or []
    tried = set(ctx.state.get("tried") or [])
    untried = [c for c in candidates if c not in tried]

    ctx.state["code_fix_attempts"] = 0  # counters reset per NEW lever
    ctx.state["pcc_fix_attempts"] = 0
    ctx.state["inert_fix_attempts"] = 0
    ctx.state.pop("inert_repair_error", None)
    ctx.state.pop("prev_fixer_sig", None)
    ctx.state.pop("repair_history", None)  # per-lever: prior failed repair approaches fed back to the editor
    ctx.state.pop("last_edit_summary", None)

    if not untried:
        # exhausted bucket — let CHECK_EXIT's no-untried-levers floor stop the run.
        ctx.state["selected_lever"] = candidates[0] if candidates else None
        ctx.state["select_reasoning"] = "no untried candidates (fallback)"
        ctx.log_event(states.SELECT, "info", "no untried candidates")
        return states.PLAN

    brief = _read_brief(ctx)
    runner = ctx.deps.get("select_runner") or _default_runner()

    chosen, reasoning, model, usage = untried[0], "fallback: first untried", "?", None
    prompt_text, response_text = None, None
    try:
        result = runner(brief=brief, candidates=untried, tried=sorted(tried))
        if result.get("lever") in untried:
            chosen = result["lever"]
            reasoning = result.get("reasoning", "")
            model = result.get("model", "?")
            usage = result.get("usage")
            prompt_text, response_text = result.get("prompt"), result.get("response")
            skipped = [s for s in (result.get("skip") or []) if s in untried and s != chosen]
            if skipped:
                tlist = ctx.state.setdefault("tried", [])
                for s in skipped:
                    if s not in tlist:
                        tlist.append(s)
                ctx.log_event(states.SELECT, "info", f"pruned (judged irrelevant to bottleneck): {skipped}")
        else:
            ctx.log_event(states.SELECT, "warn", f"invalid pick {result.get('lever')!r}; fallback {untried[0]}")
    except Exception as exc:  # graceful fallback (PLAN 8.3)
        ctx.log_event(states.SELECT, "warn", f"select error; fallback {untried[0]}: {exc}")

    ctx.state["selected_lever"] = chosen
    ctx.state["select_reasoning"] = reasoning
    ctx.record_agent_call(states.SELECT, "select", model, usage, prompt=prompt_text, response=response_text)
    ctx.log_event(states.SELECT, "info", f"lever={chosen}")
    return states.PLAN


def _read_brief(ctx) -> str:
    from ..events import read_jsonl_last

    rid = ctx.state.get("route_brief_id")
    if not rid:
        return ""
    row = read_jsonl_last(ctx.run.dir / "route_briefs.jsonl", route_brief_id=rid)
    return json.dumps(row, indent=2, sort_keys=True) if row else ""


def _default_runner():
    from ..select_agent import make_select_runner

    return make_select_runner()
