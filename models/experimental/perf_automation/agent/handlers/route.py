"""ROUTE handler (Member 1) — REAL, deterministic. No agent, no device.

In:  ctx.baseline_profile() buckets + ctx.index (playbook).
Out: ctx.state["current_bucket"], ctx.state["candidates"] (section ids). -> SELECT

This one is essentially complete; the only open knob is TBD(bucket-select-policy)
(currently top-by-device_ms). Use it as the template for the other M1 handlers.
"""

from __future__ import annotations

from typing import Any

from .. import router, states


def _select_bucket(buckets: list[dict[str, Any]], exhausted: set[str]) -> dict[str, Any]:
    """TBD(bucket-select-policy): top remaining bucket by device time."""
    pool = [b for b in buckets if b["id"] not in exhausted] or buckets
    return max(pool, key=lambda b: b.get("device_ms", 0.0))


def _bucket_query(tags: dict[str, Any]) -> dict[str, Any]:
    """Project a bucket's tags onto the router's 8 dimensions (drop anything else)."""
    return {k: v for k, v in tags.items() if k in router.DIMENSIONS}


def route(ctx) -> str:
    profile = ctx.current_profile()  # latest committed model, NOT the frozen baseline
    bucket = _select_bucket(profile["buckets"], set(ctx.state.get("exhausted_buckets", [])))
    query = _bucket_query(bucket.get("tags", {}))
    sections = router.route(ctx.index, query)
    ctx.state["current_bucket"] = bucket["id"]
    ctx.state["candidates"] = [s["id"] for s in sections]
    ctx.log_event(states.ROUTE, "info", f"bucket={bucket['id']} candidates={len(sections)}")
    return states.SELECT
