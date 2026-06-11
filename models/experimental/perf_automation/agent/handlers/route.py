"""ROUTE handler (Member 1 / lead) — REAL, deterministic. No agent, no device.

Picks the slowest bucket of the CURRENT profile, asks the router which playbook
levers are tagged for it, and writes a **route brief** — a table of candidates
plus the full extracted text of each section. That brief is the decision
material SELECT (the agent) reads to choose one lever.

In:  ctx.current_profile() buckets + ctx.index (playbook).
Out: ctx.state["current_bucket"], ["candidates"] (ids), ["route_brief"] (file path). -> SELECT
"""

from __future__ import annotations

from typing import Any, Callable

from .. import router, states


def _select_bucket(buckets: list[dict[str, Any]], exhausted: set[str]) -> dict[str, Any]:
    """TBD(bucket-select-policy): top remaining bucket by device time."""
    pool = [b for b in buckets if b["id"] not in exhausted] or buckets
    return max(pool, key=lambda b: b.get("device_ms", 0.0))


def _bucket_query(tags: dict[str, Any]) -> dict[str, Any]:
    """Project a bucket's tags onto the router's 8 dimensions (drop anything else)."""
    return {k: v for k, v in tags.items() if k in router.DIMENSIONS}


def build_route_brief(bucket: dict[str, Any], hits: list[dict[str, Any]], read_section: Callable[[str], str]) -> str:
    """Assemble the human/agent-readable decision brief: bottleneck + table + section texts."""
    t = bucket.get("tags", {})
    out: list[str] = [
        f"# ROUTE brief — bottleneck: {bucket['id']}",
        "",
        f"**{bucket['id']}** — {bucket.get('device_ms')} ms " f"({bucket.get('pct')}%), {bucket.get('count')} calls",
        "tags: " + ", ".join(f"{k}={v}" for k, v in sorted(t.items())),
        "",
        f"## {len(hits)} candidate levers (choose ONE)",
        "",
        "| id | lever_type | file | title |",
        "|----|-----------|------|-------|",
    ]
    for h in hits:
        out.append(f"| {h['id']} | {h.get('lever_type', '')} | {h['file']} | {h['title']} |")
    out += ["", "## Playbook sections — the full text to decide from", ""]
    for h in hits:
        out += [f"### {h['id']}  ({h['file']})", ""]
        try:
            out.append(read_section(h["id"]))
        except KeyError:
            out.append("_(section text unavailable in this index)_")
        out += ["", "---", ""]
    return "\n".join(out)


def route(ctx) -> str:
    profile = ctx.current_profile()
    bucket = _select_bucket(profile["buckets"], set(ctx.state.get("exhausted_buckets", [])))
    query = _bucket_query(bucket.get("tags", {}))
    hits = router.route(ctx.index, query)

    ctx.state["current_bucket"] = bucket["id"]
    ctx.state["candidates"] = [h["id"] for h in hits]

    # persist the decision material for SELECT (and for a human to inspect)
    rel = f"route_brief_{ctx.state.get('iteration', 0):02d}.md"
    (ctx.run.dir / rel).write_text(build_route_brief(bucket, hits, router.read_section))
    ctx.state["route_brief"] = rel

    ctx.log_event(states.ROUTE, "info", f"bucket={bucket['id']} candidates={len(hits)} brief={rel}")
    return states.SELECT
