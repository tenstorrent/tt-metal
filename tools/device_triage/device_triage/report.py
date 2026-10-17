# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fleet-wide and per-category reports built on `triage`.

fleet:      every host with recent HC data -> status, likely issue, open tickets.
categories: per DIIM category (GDDR, PCIe, Tensix, BMC, Resets): the DIIM
            tickets in it, and whether Cluster HC / Machine HC logs corroborate
            each one at the time it was filed, with the evidence line.
"""

import sqlite3
import time
from dataclasses import dataclass

from . import tickets, triage
from .signatures import CATEGORIES, PRIMARY_CATEGORIES

DAY = triage.DAY
STATUS_ORDER = {"FAILING": 0, "FLAKY": 1, "DEGRADED": 2, "UNKNOWN": 3, "HEALTHY": 4}
UNTRACKED_MIN_SCORE = 1.0


@dataclass
class FleetRow:
    host: str
    status: str
    reason: str
    likely: str
    confidence: str
    where: str
    scores: list[tuple[str, float]]
    last_mhc: float | None
    open_tickets: list[str]
    open_same_category: bool
    auto_tickets: list[str]


def fleet_hosts(conn: sqlite3.Connection, as_of: float, days: int) -> list[str]:
    return [
        h
        for (h,) in conn.execute(
            "SELECT DISTINCT h.host FROM run_hosts h JOIN runs r USING (run_id) WHERE r.source = 'machine_hc' "
            "AND r.ts_start BETWEEN ? AND ? ORDER BY h.host",
            (as_of - days * DAY, as_of),
        )
    ]


def fleet(conn: sqlite3.Connection, days: int = 14, as_of: float | None = None) -> list[FleetRow]:
    as_of = as_of or time.time()
    stats = triage.fleet_stats(conn)
    rows = []
    for host in fleet_hosts(conn, as_of, days):
        r = triage.triage(conn, host, as_of=as_of, window_days=days, stats=stats)
        open_about = [t for t in r.tickets if t["open"] and t["subject"]]
        human = [t for t in open_about if not t["bot"]]
        rows.append(
            FleetRow(
                host,
                r.status,
                r.status_reason,
                r.likely,
                r.confidence,
                r.likely_chips,
                [(s.category, s.score) for s in r.scores if s.score > 0 and s.category not in triage.NON_DEVICE][:3],
                r.machine_hc.get("last_ts"),
                [t["key"] for t in human],
                any(triage.same_family(r.likely, c) for t in human for c in t["categories"]),
                [t["key"] for t in open_about if t["bot"]],
            )
        )
    top = {row.host: (row.scores[0][1] if row.scores else 0.0) for row in rows}
    return sorted(rows, key=lambda row: (STATUS_ORDER.get(row.status, 9), -top[row.host], row.host))


def untracked(rows: list[FleetRow]) -> list[FleetRow]:
    """Failing or flaky hosts with solid evidence for a device issue and no open human ticket about it."""
    return [
        r
        for r in rows
        if r.status in ("FAILING", "FLAKY")
        and r.likely != "none"
        and not r.open_same_category
        and dict(r.scores).get(r.likely, 0.0) >= UNTRACKED_MIN_SCORE
    ]


def fleet_markdown(rows: list[FleetRow], days: int) -> str:
    counts = {s: sum(r.status == s for r in rows) for s in STATUS_ORDER}
    todo = untracked(rows)
    lines = [
        f"# Fleet status ({triage.fmt_ts(time.time())} UTC, {days}-day window)",
        "",
        "Hosts with a machine HC run in the window. " + ", ".join(f"{k}: {v}" for k, v in counts.items() if v),
        "",
        f"## Failing or flaky with no open ticket about the likely issue ({len(todo)})",
        "",
        f"Only hosts whose likely category scores >= {UNTRACKED_MIN_SCORE:g}. Open tickets are human tickets about the",
        "host (named in the title, or in a short description list); PCIe and reset count as one issue. HC bot",
        "auto-tickets are listed but don't count: nobody has triaged them.",
        "",
        "| host | status | likely issue | where | scores | open tickets | HC bot auto-ticket |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in todo:
        lines.append(
            f"| {r.host} | {r.status} | {r.likely} ({r.confidence}) | {r.where or '-'} | "
            f"{', '.join(f'{c}={s:.1f}' for c, s in r.scores)} | {', '.join(r.open_tickets) or '-'} | "
            f"{', '.join(r.auto_tickets) or '-'} |"
        )
    lines += [
        "",
        "## All hosts",
        "",
        "| host | status | likely issue | where | scores | last machine HC | open tickets | HC bot auto-ticket |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.host} | {r.status} | {r.likely} ({r.confidence}) | {r.where or '-'} | "
            f"{', '.join(f'{c}={s:.1f}' for c, s in r.scores) or '-'} | {triage.fmt_ts(r.last_mhc)} | "
            f"{', '.join(r.open_tickets) or '-'} | {', '.join(r.auto_tickets) or '-'} |"
        )
    return "\n".join(lines) + "\n"


@dataclass
class CategoryCase:
    key: str
    summary: str
    created: float
    status: str
    host: str
    in_title: bool
    score: float
    logs_top: str
    evidence: list[triage.Evidence]


@dataclass
class CategoryResult:
    category: str
    jql: str
    n_tickets: int
    n_jql: int
    n_title: int
    n_title_only: int
    cases: list[CategoryCase]

    @property
    def corroborated(self) -> list[CategoryCase]:
        return [c for c in self.cases if c.score >= 1]

    @property
    def weak(self) -> list[CategoryCase]:
        return [c for c in self.cases if 0.3 <= c.score < 1]


def categories_data(conn: sqlite3.Connection, window_days: int = 14, lead_hours: float = 12.0) -> list[CategoryResult]:
    stats = triage.fleet_stats(conn)
    spec = tickets.jql_membership()
    results = []
    for cat in PRIMARY_CATEGORIES:
        jql = spec.get(cat, {})
        jql_keys = set(jql.get("keys", []))
        title_keys = {
            k for (k,) in conn.execute("SELECT key FROM ticket_cats WHERE category = ? AND source = 'text'", (cat,))
        }
        keys = jql_keys | title_keys
        rows = (
            conn.execute(
                f"SELECT t.key, t.created, t.status, t.summary FROM tickets t WHERE t.key IN ({','.join('?' * len(keys))}) "
                "ORDER BY t.created DESC",
                sorted(keys),
            ).fetchall()
            if keys
            else []
        )
        cases = []
        for key, created, status, summary in rows:
            for host in tickets.subject_hosts(conn, key)[:4]:
                r = triage.triage(conn, host, as_of=created + lead_hours * 3600, window_days=window_days, stats=stats)
                score = next((s for s in r.scores if s.category == cat), None)
                cases.append(
                    CategoryCase(
                        key,
                        summary,
                        created,
                        status,
                        host,
                        key in title_keys,
                        score.score if score else 0.0,
                        r.likely,
                        score.evidence[:2] if score else [],
                    )
                )
        cases.sort(key=lambda c: (-min(c.score, 1.0), -c.created))
        results.append(
            CategoryResult(
                cat, jql.get("jql", ""), len(rows), len(jql_keys), len(title_keys), len(title_keys - jql_keys), cases
            )
        )
    return results


def categories(results: list[CategoryResult], window_days: int = 14, lead_hours: float = 12.0) -> str:
    out = [
        "# DIIM categories vs Cluster HC / Machine HC logs",
        "",
        "For each category: the JQL that selects its DIIM tickets (summary + description), how many tickets name an",
        "exabox host, and for each such ticket whether the logs show that category for the host in the "
        f"{window_days} days before it was filed (triage replayed at creation + {lead_hours:g}h). `score` is the",
        "category's recency- and rarity-weighted score; >= 1 is solid, 0.3-1 weak, < 0.3 nothing. Evidence",
        "lines link the exact file and line.",
        "",
    ]
    for res in results:
        n_none = len(res.cases) - len(res.corroborated) - len(res.weak)
        out += [
            f"## {CATEGORIES[res.category]}",
            "",
            f"JQL: `{res.jql or '(title keywords only)'}`",
            "",
            f"- tickets: {res.n_tickets} ({res.n_jql} from JQL, {res.n_title_only} more from title keywords; "
            f"{res.n_title} have the category in the title)",
            f"- ticket x host pairs naming an exabox host: {len(res.cases)}; corroborated by logs (score >= 1): "
            f"{len(res.corroborated)}; weak (0.3-1): {len(res.weak)}; no log evidence: {n_none}",
            "",
            "| ticket | in title | filed | status | host | score | logs' top category |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in res.cases:
            out.append(
                f"| {c.key} {c.summary[:70].replace('|', '/')} | {'yes' if c.in_title else ''} | "
                f"{triage.fmt_ts(c.created)[:10]} | {c.status} | {c.host} | {c.score:.1f} | {c.logs_top} |"
            )
        if res.corroborated or res.weak:
            out += ["", "Evidence for corroborated and weak matches:", ""]
            for c in res.corroborated + res.weak:
                for ev in c.evidence:
                    where = f"{ev.path}:{ev.line}" if ev.line else ev.path
                    out.append(
                        f"- {c.key} {c.host}: `{ev.signal}` ({ev.severity}, {triage.fmt_ts(ev.ts)}, {ev.runs} run(s))"
                        f"{' chips ' + ev.chips if ev.chips else ''}: {ev.detail[:160]} - `{where}`"
                    )
        out.append("")
    return "\n".join(out) + "\n"


def post_reset_offenders(conn: sqlite3.Connection, limit: int = 12) -> list[dict]:
    """Chips that failed POST_RESET in the most recover runs, with the human tickets about their host."""
    out = []
    rows = conn.execute(
        "SELECT host, chips, COUNT(*), MIN(ts), MAX(ts) FROM events WHERE signal = 'post_reset_failed' "
        "GROUP BY host, chips ORDER BY COUNT(*) DESC LIMIT ?",
        (limit,),
    ).fetchall()
    for host, chips, runs, first, last in rows:
        ordinals = triage.ordinal_map(conn, host)
        resolved = [
            ordinals.get(int(c[3:]), c) if c.startswith("dev") and c[3:].isdigit() and c != "dev0" else c
            for c in (chips or "").split(",")
            if c
        ]
        named = conn.execute(
            "SELECT t.key, t.summary FROM ticket_hosts h JOIN tickets t USING (key) WHERE h.host = ? AND h.subject = 1 "
            "AND substr(t.summary, 1, ?) != ? ORDER BY t.created",
            (host, len(tickets.HC_BOT_PREFIX), tickets.HC_BOT_PREFIX),
        ).fetchall()
        out.append(
            {
                "host": host,
                "device": chips,
                "chip": ",".join(resolved),
                "runs": runs,
                "first": first,
                "last": last,
                "tickets": [{"key": k, "summary": s} for k, s in named],
            }
        )
    return out
