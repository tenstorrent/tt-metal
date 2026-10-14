# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay triage at each ticket's creation time and compare with the ticket's category.

A ticket is evaluated when it names 1-4 hosts unambiguously and those hosts have
indexed HC data in the window before it was filed. "hit" means the predicted
category is one of the ticket's device categories; "evidence" means at least one
of them scored, even if something else scored higher.
"""

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from . import tickets as ticket_store
from . import triage
from .signatures import CATEGORIES

DEVICE_CATEGORIES = [c for c in CATEGORIES if c not in triage.NON_DEVICE]
EVIDENCE_MIN = 0.3
# A host that is down produces no HC runs, so these tickets are out of reach by construction.
HOST_DOWN_RE = re.compile(
    r"unreachable|not reachable|can'?t (?:reach|ping)|cannot (?:reach|ping)|not coming back|"
    r"crash|going down|\bdown\b|offline|won'?t boot|not booting|fails to post",
    re.I,
)


@dataclass
class Case:
    key: str
    host: str
    created: float
    status: str
    summary: str
    ticket_cats: list[str]
    predicted: str
    confidence: str
    scores: dict[str, float]
    chips: str
    host_status: str

    @property
    def hit(self) -> bool:
        return self.predicted in self.ticket_cats

    @property
    def family_hit(self) -> bool:
        return any(triage.same_family(self.predicted, c) for c in self.ticket_cats)

    @property
    def evidence(self) -> bool:
        return any(self.scores.get(c, 0) >= EVIDENCE_MIN for c in self.ticket_cats)


def ticket_categories(conn: sqlite3.Connection, key: str) -> list[str]:
    """Device categories from the title; JQL (description) membership only when the title has none."""
    rows = conn.execute("SELECT category, source FROM ticket_cats WHERE key = ?", (key,)).fetchall()
    text = [c for c in DEVICE_CATEGORIES if (c, "text") in rows]
    if text:
        return text
    return [c for c in DEVICE_CATEGORIES if (c, "jql") in rows and c in {"gddr", "tensix", "pcie"}]


def cases(conn: sqlite3.Connection, window_days: int = 14, lead_hours: float = 12.0) -> list[Case]:
    stats = triage.fleet_stats(conn)
    out = []
    tickets = conn.execute(
        "SELECT t.key, t.created, t.status, t.summary FROM tickets t WHERE t.project = 'DIIM' AND t.created IS NOT NULL "
        "ORDER BY t.created"
    ).fetchall()
    for key, created, status, summary in tickets:
        hosts = ticket_store.subject_hosts(conn, key)
        if not 1 <= len(hosts) <= 4:
            continue
        cats = [] if HOST_DOWN_RE.search(summary) else ticket_categories(conn, key)
        for host in hosts:
            as_of = created + lead_hours * 3600
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM events WHERE host = ? AND ts BETWEEN ? AND ?",
                (host, as_of - window_days * triage.DAY, as_of),
            ).fetchone()
            if not n:
                continue
            report = triage.triage(conn, host, as_of=as_of, window_days=window_days, stats=stats)
            out.append(
                Case(
                    key,
                    host,
                    created,
                    status,
                    summary,
                    cats,
                    report.likely,
                    report.confidence,
                    {s.category: s.score for s in report.scores},
                    report.likely_chips,
                    report.status,
                )
            )
    return out


def by_category(all_cases: list[Case]) -> list[dict]:
    device = [c for c in all_cases if c.ticket_cats]
    rows = []
    for cat in DEVICE_CATEGORIES:
        sub = [c for c in device if cat in c.ticket_cats]
        if sub:
            rows.append(
                {
                    "category": cat,
                    "cases": len(sub),
                    "top1": sum(c.predicted == cat for c in sub),
                    "family": sum(triage.same_family(c.predicted, cat) for c in sub),
                    "evidence": sum(c.scores.get(cat, 0) >= EVIDENCE_MIN for c in sub),
                    "none": sum(c.predicted == "none" for c in sub),
                }
            )
    return rows


def summarize(all_cases: list[Case]) -> str:
    device = [c for c in all_cases if c.ticket_cats]
    lines = [
        "# Validation: triage prediction vs DIIM ticket category",
        "",
        "Triage replayed as of 12h after each ticket was created, 14-day window. Only tickets naming 1-4 hosts",
        "unambiguously, with indexed events for the host in the window.",
        "",
        f"- cases (ticket x host): {len(all_cases)}; with a device category in the ticket: {len(device)}",
        f"- top-1 hit (predicted category is one of the ticket's): {sum(c.hit for c in device)}/{len(device)}",
        f"- top-1 hit counting PCIe and reset as one family: {sum(c.family_hit for c in device)}/{len(device)}",
        f"- evidence (a ticket category scored >= {EVIDENCE_MIN}): {sum(c.evidence for c in device)}/{len(device)}",
        f"- predicted 'none' on device tickets: {sum(c.predicted == 'none' for c in device)}",
        "",
    ]
    lines += [
        "## By ticket category",
        "",
        "| category | cases | top-1 hit | top-1 (PCIe/reset family) | evidence | predicted none |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in by_category(all_cases):
        lines.append(
            f"| {row['category']} | {row['cases']} | {row['top1']} | {row['family']} | {row['evidence']} | "
            f"{row['none']} |"
        )
    confusion = Counter((c.ticket_cats[0], c.predicted) for c in device)
    lines += ["", "## Confusion (ticket primary device category -> predicted)", ""]
    for (t, p), n in confusion.most_common():
        lines.append(f"- {t} -> {p}: {n}")
    lines += [
        "",
        "## Cases",
        "",
        "| ticket | host | created | ticket cats | predicted (conf) | where | top scores | ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in all_cases:
        top = ", ".join(f"{k}={v:.1f}" for k, v in sorted(c.scores.items(), key=lambda kv: -kv[1])[:3] if v > 0)
        ok = "-" if not c.ticket_cats else ("hit" if c.hit else "evidence" if c.evidence else "miss")
        lines.append(
            f"| {c.key} | {c.host} | {triage.fmt_ts(c.created)[:10]} | {','.join(c.ticket_cats) or '-'} | "
            f"{c.predicted} ({c.confidence}) | {c.chips or '-'} | {top or '-'} | {ok} |"
        )
    return "\n".join(lines) + "\n"
