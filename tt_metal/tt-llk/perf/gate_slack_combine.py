# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One Slack post for all arches of a gate run, built from the per-arch parts.

Each arch's gate job writes a part with ``gate_slack_notify.py --part-out``.
A regression on one arch also shows how the same points moved on the others,
read from their ``.points.csv``: a move under that arch's threshold is easy to
miss, and the reader should not have to open a second report to see it.
"""

import csv
import os

from gate_slack_notify import (
    MODULE_KEY,
    _config_of,
    _cycles,
    _escape,
    _pct,
    _subject,
    regression_body,
)

# A regression first, then a skip, then a pass: the post leads with what needs action.
_ORDER = {"regressed": 0, "skipped": 1, "clean": 2}


def _point_key(row):
    return (
        row.get(MODULE_KEY) or "",
        row.get("marker") or "",
        row.get("run_type") or "",
        frozenset(_config_of(row).items()),
    )


def find_points(path, rows):
    """The rows of ``path`` (a ``.points.csv``) that share a key with ``rows``."""
    wanted = {_point_key(r) for r in rows}
    found = {}
    if not wanted or not path or not os.path.exists(path):
        return found
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = _point_key(row)
            if key in wanted:
                found[key] = row
    return found


def threshold_for(ctx, module):
    """The relative threshold the arch's gate applied to ``module``, or None."""
    for pair in (ctx.get("module_thresholds") or "").split(","):
        name, _, value = pair.partition("=")
        if name.strip() and name.strip() == module.split("+")[0]:
            try:
                return float(value)
            except ValueError:
                break
    try:
        return float(ctx.get("threshold") or "")
    except ValueError:
        return None


def _pct_short(value):
    return f"{value * 100:g}%"


def _rule(ctx):
    """The threshold line of an arch's section; empty when no threshold was given."""
    base = threshold_for(ctx, "")
    if base is None:
        return []
    rule = f"slower by more than {_pct_short(base)}"
    if ctx.get("min_cycles"):
        rule += f" and by more than {_escape(ctx['min_cycles'])} cycles"
    pairs = (ctx.get("module_thresholds") or "").split(",")
    names = sorted({p.partition("=")[0].strip() for p in pairs} - {""})
    if names:
        rule += ", except " + ", ".join(
            f"{_escape(m)} {_pct_short(threshold_for(ctx, m))}" for m in names
        )
    return [f"_Threshold: a point regresses when it is {rule}._"]


def _elsewhere(row, others):
    """How the same point moved on the other arches, as one line."""
    bits = []
    for part, points in others:
        ctx = part["ctx"]
        arch = ctx.get("arch") or "?"
        hit = points.get(_point_key(row))
        if hit is None:
            continue
        what = f"{_pct(hit.get('delta_pct'))} ({_cycles(hit.get('delta_cycles'))})"
        if part["status"] == "regressed":
            bits.append(f"{arch}: {what}")
            continue
        limit = threshold_for(ctx, row.get(MODULE_KEY) or "")
        under = (
            f", under its {_pct_short(limit)} threshold" if limit is not None else ""
        )
        bits.append(f"{arch}: {what}{under}")
    return "same point on " + "; ".join(bits) if bits else ""


def _section(part, others):
    ctx = part["ctx"]
    arch = (ctx.get("arch") or "?").capitalize()
    head = f"*{_escape(arch)}* · {_escape(ctx.get('run_types') or '?')}"
    note = [f":hourglass: {_escape(ctx['note'])}"] if ctx.get("note") else []
    rule = _rule(ctx)
    status = part["status"]
    if status == "clean":
        return [f"{head}: :white_check_mark: no point regressed.", *rule, *note]
    if status == "skipped":
        reason = _escape(ctx.get("reason") or "The gate compared nothing.")
        return [f"{head}: :warning: *SKIPPED.* {reason}", *note]
    return [
        f"{head}: :rotating_light: regression",
        *rule,
        *note,
        *regression_body(part["rows"], ctx, lambda row: _elsewhere(row, others)),
    ]


def build_text(parts, points_by_arch):
    """The one post. ``points_by_arch`` maps an arch to its ``.points.csv`` path."""
    parts = sorted(parts, key=lambda p: (_ORDER.get(p["status"], 9), p["ctx"]["arch"]))
    ctx = parts[0]["ctx"]
    subject, byline, _ = _subject(ctx)
    gate = _escape(ctx.get("gate_name") or "LLK perf gate")
    worst = parts[0]["status"]
    if worst == "regressed":
        arches = ", ".join(
            p["ctx"]["arch"] for p in parts if p["status"] == "regressed"
        )
        head = f":rotating_light: *{gate}: regression on {arches}* {subject}"
    elif worst == "skipped":
        head = f":warning: *{gate} SKIPPED* {subject}"
    else:
        head = f":white_check_mark: *{gate} passed* {subject}"
    lines = [head, byline]
    if ctx.get("pr_title"):
        lines.append(f"_{_escape(ctx['pr_title'])}_")
    if worst == "skipped":
        lines.append("*A green check here does not mean anything was checked.*")

    for part in parts:
        others = []
        if part["status"] == "regressed":
            for other in parts:
                if other is part:
                    continue
                path = points_by_arch.get(other["ctx"].get("arch"))
                others.append((other, find_points(path, part["rows"])))
        lines += ["", *_section(part, others)]
    return "\n".join(lines)
