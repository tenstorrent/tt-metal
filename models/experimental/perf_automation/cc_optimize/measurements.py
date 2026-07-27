# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measurement ledger: the ONE place a reported number comes from.

Every headline defect this tool has produced came from the same shape -- the report needing a
"before" number and not having one recorded, so it searched a chain of files and found something
with different provenance:

    eager per-op device time (all layers):  0.06 ms -> 648.17 ms  (-1062476.1%)
        a sub-millisecond anchor from an unrelated model's run, against a real 648 ms reading

    baseline 832.93 ms -> final 1088.15 ms  (-30.6%)
        a 2-layer profile paired with a 16-layer one; a regression that never happened

    before 47.10 ms [eager] -> after 100.00 ms [trace+1cq]
        two different units subtracted from each other

Each was fixed by hardening one link of the chain. The chain is the defect: a number reached the
report carrying no statement of WHAT it measured, so nothing could tell that it did not belong.

Here a measurement is APPENDED when it is taken, with its provenance, and the report reads only
this. There is nowhere to fall back to, so a foreign or stale value cannot be promoted into an
anchor; and two rows are subtracted only when they describe the same work.

DURABILITY IS THE POINT. The ledger is keyed by (model, task) and is never truncated -- not by a
rerun, not by a fresh ladder, not by clearing the kernel log. `first("eager_per_op", "before")` is
the earliest before-reading ever taken for this model, so re-running optimize on an
already-optimized model still reports against the TRUE original. Without that, the second run
measures the optimized model, calls that its baseline, and the 2464 -> 648 result becomes
unreportable the moment you restart.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

_SCHEMA = 1

KIND_EAGER = "eager_per_op"
KIND_TRACE_PASS = "trace_pass"
KIND_FULLPIPE = "fullpipe_e2e"

PHASE_BEFORE = "before"
PHASE_AFTER = "after"


def ledger_path(model: str = "", task: str = "") -> Path:
    """Keyed by (model, task), like every other per-run artifact. An unkeyed file is how another
    run's number became this run's baseline."""
    override = os.environ.get("PERF_MCP_LEDGER")
    if override:
        return Path(override)
    model = (
        model
        or os.environ.get("PERF_MCP_MODEL_NAME")
        or Path(os.environ.get("PERF_MCP_MODEL_ROOT", "") or "model").name
    )
    task = task or os.environ.get("PERF_MCP_TASK", "main")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (model, task))
    return Path(tempfile.gettempdir()) / ("perf_measurements_%s.jsonl" % safe)


def record(
    kind: str,
    phase: str,
    value_ms,
    *,
    depth: str = "",
    mode: str = "",
    stage: str = "",
    source: str = "",
    model: str = "",
    task: str = "",
) -> bool:
    """Append one measurement. Returns False when it is not worth recording.

    A reading with no depth or no mode is REFUSED rather than stored blank: an unlabelled number is
    exactly what the report cannot safely use, and storing it would rebuild the guessing problem
    inside the ledger.
    """
    try:
        v = float(value_ms)
    except (TypeError, ValueError):
        return False
    if v <= 0:
        return False
    if not str(depth).strip() or not str(mode).strip():
        return False
    row = {
        "schema": _SCHEMA,
        "kind": str(kind),
        "phase": str(phase),
        "value_ms": round(v, 4),
        "depth": str(depth).strip(),
        "mode": str(mode).strip(),
        "stage": str(stage or "").strip(),
        "source": str(source or "").strip(),
    }
    try:
        p = ledger_path(model, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def rows(kind: str = "", phase: str = "", model: str = "", task: str = "") -> list:
    """Every matching row, oldest first. A malformed line is skipped, never fatal -- a corrupt
    ledger must degrade to 'not measured', not crash the report."""
    out = []
    try:
        text = ledger_path(model, task).read_text()
    except Exception:  # noqa: BLE001
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict):
            continue
        if kind and r.get("kind") != kind:
            continue
        if phase and r.get("phase") != phase:
            continue
        out.append(r)
    return out


def first(kind: str, phase: str = PHASE_BEFORE, model: str = "", task: str = ""):
    """The EARLIEST matching reading -- the true original, surviving every rerun."""
    rs = rows(kind, phase, model, task)
    return rs[0] if rs else None


def last(kind: str, phase: str = PHASE_AFTER, model: str = "", task: str = ""):
    """The most recent matching reading -- the current state."""
    rs = rows(kind, phase, model, task)
    return rs[-1] if rs else None


def comparable(a, b) -> tuple:
    """(ok, why). Two readings may be subtracted only when they describe the SAME work: same depth,
    same mode, same stage. This is the structural version of the checks that were previously spread
    across the renderer as magnitude heuristics and mode string comparisons."""
    if not a or not b:
        return False, "not measured"
    for axis in ("depth", "mode", "stage"):
        av, bv = str(a.get(axis) or ""), str(b.get(axis) or "")
        if av != bv:
            return False, "%s differs: %s vs %s" % (axis, av or "unknown", bv or "unknown")
    return True, "comparable"


def delta_pct(a, b):
    """Percent improvement from a -> b, or None when the pair may not be compared."""
    ok, _ = comparable(a, b)
    if not ok:
        return None
    before, after = float(a.get("value_ms") or 0.0), float(b.get("value_ms") or 0.0)
    if before <= 0 or after <= 0:
        return None
    return (before - after) / before * 100.0
