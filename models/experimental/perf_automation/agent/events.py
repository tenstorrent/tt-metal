"""Single events.jsonl schema, shared by the Before Loop and the Agent Loop.

Every row has the SAME keys, so the format never drifts across phases (the bug:
before_loop wrote {stage:int, name, event}, the loop wrote {stage:str, status,
iteration}). One shape:

  ts         UTC ISO-8601 with Z (clock.utc_ts)
  phase      "before_loop" | "loop"
  stage      stage/state NAME, always a string (e.g. "resolve_signposts", "ROUTE")
  event      "start" | "done" | "info" | "warn" | "note"
  detail     human-readable string
  seconds    duration when known, else null (before-loop "done" events)
  iteration  loop iteration when applicable, else null
"""

from __future__ import annotations

import json
from pathlib import Path

from .clock import utc_ts

EVENT_FIELDS = ("ts", "phase", "stage", "event", "detail", "seconds", "iteration")


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
