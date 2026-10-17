# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the team overview as a Cursor canvas: canvas_template.tsx with the data inlined."""

import json
import sqlite3
import time
from pathlib import Path

from . import report, triage, validate
from .signatures import CATEGORIES

TEMPLATE = Path(__file__).with_name("canvas_template.tsx")


def _day(ts: float | None) -> str:
    return triage.fmt_ts(ts)[:10] if ts else "-"


def _evidence(ev: triage.Evidence) -> dict:
    return {
        "signal": ev.signal,
        "severity": ev.severity,
        "when": triage.fmt_ts(ev.ts),
        "runs": ev.runs,
        "chips": ev.chips or "",
        "detail": (ev.detail or "")[:220],
        "where": f"{ev.path}:{ev.line}" if ev.line else ev.path or "",
    }


def _fleet_row(r: report.FleetRow) -> dict:
    return {
        "host": r.host,
        "status": r.status,
        "likely": r.likely,
        "confidence": r.confidence,
        "where": r.where or "",
        "scores": ", ".join(f"{c}={s:.1f}" for c, s in r.scores),
        "lastRun": triage.fmt_ts(r.last_mhc),
        "open": r.open_tickets,
        "auto": r.auto_tickets,
    }


def payload(conn: sqlite3.Connection, days: int = 14) -> dict:
    rows = report.fleet(conn, days=days)
    untracked = report.untracked(rows)
    cats = report.categories_data(conn, window_days=days)
    val_cases = validate.cases(conn, window_days=days)
    device = [c for c in val_cases if c.ticket_cats]
    (machine_runs,) = conn.execute("SELECT COUNT(*) FROM runs WHERE source = 'machine_hc'").fetchone()
    (rebuilt_runs,) = conn.execute("SELECT COUNT(*) FROM units WHERE source = 'dcamp_hc'").fetchone()
    (cluster_records,) = conn.execute("SELECT COUNT(*) FROM units WHERE source = 'cluster_report'").fetchone()
    (logs,) = conn.execute("SELECT COUNT(*) FROM units WHERE source = 'cluster_log'").fetchone()
    (events,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    (tickets,) = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()
    return {
        "generated": f"{triage.fmt_ts(time.time())} UTC",
        "windowDays": days,
        "index": {
            "machineRuns": machine_runs,
            "rebuiltRuns": rebuilt_runs,
            "clusterRecords": cluster_records,
            "logs": logs,
            "events": events,
            "tickets": tickets,
        },
        "fleetCounts": {s: sum(r.status == s for r in rows) for s in report.STATUS_ORDER},
        "untracked": [_fleet_row(r) for r in untracked],
        "fleet": [_fleet_row(r) for r in rows],
        "categories": [
            {
                "category": c.category,
                "label": CATEGORIES[c.category],
                "jql": c.jql,
                "tickets": c.n_tickets,
                "jqlTickets": c.n_jql,
                "titleTickets": c.n_title,
                "pairs": len(c.cases),
                "corroborated": len(c.corroborated),
                "weak": len(c.weak),
                "none": len(c.cases) - len(c.corroborated) - len(c.weak),
                "examples": [
                    {
                        "key": x.key,
                        "summary": x.summary,
                        "host": x.host,
                        "score": round(x.score, 2),
                        "evidence": [_evidence(ev) for ev in x.evidence],
                    }
                    for x in c.corroborated
                ],
            }
            for c in cats
        ],
        "validation": [dict(row, label=CATEGORIES[row["category"]]) for row in validate.by_category(val_cases)],
        "validationTotal": {
            "cases": len(device),
            "evidence": sum(c.evidence for c in device),
            "top1": sum(c.hit for c in device),
            "family": sum(c.family_hit for c in device),
        },
        "postReset": [
            {
                "host": p["host"],
                "device": p["device"] or "",
                "chip": p["chip"],
                "runs": p["runs"],
                "first": _day(p["first"]),
                "last": _day(p["last"]),
                "tickets": [t["key"] for t in p["tickets"]],
            }
            for p in report.post_reset_offenders(conn)
        ],
    }


def write(conn: sqlite3.Connection, path: Path, days: int = 14) -> dict:
    data = payload(conn, days)
    text = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, indent=1, ensure_ascii=False), 1)
    path.write_text(text)
    return data
