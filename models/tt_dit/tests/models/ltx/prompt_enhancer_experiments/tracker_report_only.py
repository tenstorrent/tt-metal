# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: make the trace allocation tracker report-only for one diagnostic run.

The tracker raises at the first replay that finds a post-capture buffer alive, so a run can only
surface one site at a time. Patched to log instead, deduplicated by buffer id, it collects every
flagged allocation across all generations in a single run; the full reports (with allocation
tracebacks when TT_METAL_TRACE_ALLOC_TRACEBACKS=1) land in TRACKER_REPORT_PATH as JSON.
Load with ``-p models.tt_dit.tests.models.ltx.prompt_enhancer_experiments.tracker_report_only``.
"""

import json
import os
import re
import time

import ttnn.tools.trace_allocation_tracker as _tat
from loguru import logger

_SEEN: set[str] = set()
_RECORDS: list[dict] = []
_ORIG = _tat.TraceAllocationTracker.verify_before_replay.__func__
_REPLAYS = {"checked": 0, "flagged": 0}


def _report_only(cls, mesh_device, trace_id):
    _REPLAYS["checked"] += 1
    try:
        _ORIG(cls, mesh_device, trace_id)
    except RuntimeError as e:
        _REPLAYS["flagged"] += 1
        msg = str(e)
        found = re.findall(r"Buffer (\d+) \[op: ([^\]\n]*)\]", msg)
        new = [(bid, op) for bid, op in found if bid not in _SEEN]
        _SEEN.update(bid for bid, _ in new)
        if new:
            logger.warning(
                f"[tracker-report-only] trace {trace_id}: {len(new)} new unsafe buffer(s) "
                f"({len(found)} total alive): " + ", ".join(f"{bid}[{op[:50]}]" for bid, op in new)
            )
            _RECORDS.append({"time": time.time(), "trace_id": str(trace_id), "new": new, "report": msg})
            _flush()


_tat.TraceAllocationTracker.verify_before_replay = classmethod(_report_only)
logger.warning("[tracker-report-only] trace allocation tracker patched to report-only for this run")


def _flush():
    # Written on every new flag, not at session end: a run killed from outside keeps what it found.
    out = os.environ.get("TRACKER_REPORT_PATH", "tracker_report.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"replays": _REPLAYS, "unique_unsafe_buffers": len(_SEEN), "records": _RECORDS}, f, indent=1)
    os.replace(tmp, out)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    _flush()
    out = os.environ.get("TRACKER_REPORT_PATH", "tracker_report.json")
    logger.warning(
        f"[tracker-report-only] replays checked={_REPLAYS['checked']} flagged={_REPLAYS['flagged']} "
        f"unique unsafe buffers={len(_SEEN)} -> {out}"
    )
