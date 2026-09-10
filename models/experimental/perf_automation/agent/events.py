"""Shared JSONL logging + run-artifact helpers — the ONE home for all streams.

A run writes APPEND-ONLY jsonl streams, never per-iteration files:
  events.jsonl        transition timeline                 (EVENT_FIELDS)
  agent_calls.jsonl   one lean cost/token row per query() (AGENT_CALL_FIELDS)
  prompts.jsonl       one row per query() w/ full prompt+response payload
  route_briefs.jsonl  one row per ROUTE (SELECT reads the current one back)

Join keys (the "follow-along" scheme):
  iteration       coarse join — filter EVERY stream by it to see one lap.
  agent_call_id   primary key of a query(); SAME value in agent_calls.jsonl
                  (cost) and prompts.jsonl (payload) -> 1:1 join.
                  = run_id:phase:iteration:stage:role:seq
  route_brief_id  primary key of a ROUTE brief = run_id:iteration:ROUTE;
                  state["route_brief_id"] points SELECT at the current row.
"""

from __future__ import annotations

import json
from pathlib import Path

from .clock import utc_ts

EVENT_FIELDS = ("ts", "phase", "stage", "event", "detail", "seconds", "iteration")

# agent_calls.jsonl: the LEAN cost stream. No prompt text here — the full
# prompt/response live in prompts.jsonl, linked by agent_call_id.
AGENT_CALL_FIELDS = (
    "agent_call_id",
    "run_id",
    "ts",
    "phase",
    "iteration",
    "stage",
    "role",
    "model",
    "tokens_in",
    "tokens_cached",
    "tokens_out",
    "cost_usd",
    "latency_s",
    "prompt_sha",
    "response_sha",
)


# --- generic jsonl I/O ------------------------------------------------------
def append_jsonl(path, row: dict) -> None:
    """Append one dict as a line to an append-only jsonl stream."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl_last(path, **match):
    """Last row whose fields all equal `match`, or None.

    Streams line-by-line (never builds the whole file in memory). Last-wins so a
    resumed run that re-appends a row supersedes the stale one.
    """
    p = Path(path)
    if not p.is_file():
        return None
    hit = None
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if all(row.get(k) == v for k, v in match.items()):
                hit = row
    return hit


# --- events.jsonl (unchanged schema) ---------------------------------------
def event_row(phase, stage, event, detail="", seconds=None, iteration=None):
    return {
        "ts": utc_ts(),
        "phase": phase,
        "stage": str(stage),
        "event": event,
        "detail": detail,
        "seconds": seconds,
        "iteration": iteration,
    }


def write_event(path, **kw):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event_row(**kw)) + "\n")


# --- agent_calls.jsonl ------------------------------------------------------
def next_agent_call_seq(path):
    """Next append sequence = count of existing rows (monotonic, unique per run)."""
    p = Path(path)
    if not p.is_file():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def make_agent_call_id(*, run_id, phase, iteration, stage, role, seq):
    iter_part = "none" if iteration is None else str(iteration)
    return f"{run_id}:{phase}:{iter_part}:{stage}:{role}:{seq:03d}"


def make_agent_call_row(
    *,
    run_id,
    phase,
    iteration,
    stage,
    role,
    model,
    usage,
    seq,
    prompt_sha=None,
    response_sha=None,
    ts=None,
):
    """Build a normalized agent_calls row with exactly AGENT_CALL_FIELDS."""
    usage = usage or {}
    row = {
        "agent_call_id": make_agent_call_id(
            run_id=run_id, phase=phase, iteration=iteration, stage=stage, role=role, seq=seq
        ),
        "run_id": run_id,
        "ts": ts or utc_ts(),
        "phase": phase,
        "iteration": iteration,
        "stage": stage,
        "role": role,
        "model": model,
        "tokens_in": usage.get("tokens_in"),
        "tokens_cached": usage.get("tokens_cached"),
        "tokens_out": usage.get("tokens_out"),
        "cost_usd": usage.get("cost_usd"),
        "latency_s": usage.get("latency_s"),
        "prompt_sha": prompt_sha,
        "response_sha": response_sha,
    }
    return {field: row[field] for field in AGENT_CALL_FIELDS}
