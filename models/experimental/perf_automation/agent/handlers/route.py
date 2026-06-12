"""ROUTE handler (Member 1 / lead) — REAL, deterministic. No agent, no device.

Picks the slowest bucket of the CURRENT profile, asks the router which playbook
levers are tagged for it, and appends a **route brief row** to route_briefs.jsonl — candidate metadata,
model map, and full extracted section texts. SELECT reads that row back (by
route_brief_id) as its decision material.

In:  ctx.current_profile() buckets + ctx.index (playbook).
Out: ctx.state["current_bucket"], ["candidates"] (ids), ["route_brief_id"]. -> SELECT
"""

from __future__ import annotations

from typing import Any, Callable

from .. import model_map, router, states


def _select_bucket(buckets: list[dict[str, Any]], exhausted: set[str]) -> dict[str, Any]:
    """TBD(bucket-select-policy): top remaining bucket by device time."""
    pool = [b for b in buckets if b["id"] not in exhausted] or buckets
    return max(pool, key=lambda b: b.get("device_ms", 0.0))


def _bucket_query(tags: dict[str, Any]) -> dict[str, Any]:
    """Project a bucket's tags onto the router's 8 dimensions (drop anything else)."""
    return {k: v for k, v in tags.items() if k in router.DIMENSIONS}


def build_route_brief(
    bucket: dict[str, Any], hits: list[dict[str, Any]], read_section: Callable[[str], str], skeleton: str = ""
) -> dict[str, Any]:
    """Assemble JSON decision material: bottleneck + candidates + section texts."""
    sections = []
    for h in hits:
        try:
            text = read_section(h["id"])
        except KeyError:
            text = ""
        sections.append({"id": h["id"], "file": h["file"], "title": h["title"], "text": text})
    return {
        "row_type": "route_brief",
        "bucket": bucket,
        "candidate_count": len(hits),
        "candidates": [
            {
                "id": h["id"],
                "lever_type": h.get("lever_type", ""),
                "file": h["file"],
                "title": h["title"],
            }
            for h in hits
        ],
        "model_map": skeleton,
        "sections": sections,
    }


def route(ctx) -> str:
    profile = ctx.current_profile()
    bucket = _select_bucket(profile["buckets"], set(ctx.state.get("exhausted_buckets", [])))
    query = _bucket_query(bucket.get("tags", {}))
    hits = router.route(ctx.index, query)

    ctx.state["current_bucket"] = bucket["id"]
    ctx.state["candidates"] = [h["id"] for h in hits]

    # deterministic model map, filtered to this bucket's op_class — where its ops live
    op_class = bucket.get("tags", {}).get("op_class")
    subs = model_map.OP_CLASS_SUBSTRINGS.get(op_class)
    try:
        mm = model_map.build_model_map(ctx.model_files(), root=ctx.model_root())
        skeleton = model_map.render_skeleton(mm, op_substrings=subs)
    except Exception:
        skeleton = ""

    # append structured decision material to the single route_briefs.jsonl stream;
    # SELECT reads its row back by route_brief_id (state below points at it).
    from ..events import append_jsonl

    route_brief_id = f"{ctx.run.run_id}:{ctx.state.get('iteration', 0)}:ROUTE"
    payload = build_route_brief(bucket, hits, router.read_section, skeleton)
    payload.update(
        {
            "route_brief_id": route_brief_id,
            "run_id": ctx.run.run_id,
            "iteration": ctx.state.get("iteration", 0),
            "stage": states.ROUTE,
            "query": query,
        }
    )
    append_jsonl(ctx.run.dir / "route_briefs.jsonl", payload)
    ctx.state["route_brief_id"] = route_brief_id

    ctx.log_event(states.ROUTE, "info", f"bucket={bucket['id']} candidates={len(hits)} brief_id={route_brief_id}")
    return states.SELECT
