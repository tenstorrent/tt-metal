# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-device status and likely issue, from the index.

Scoring, per category over a time window ending at `as_of`:
  event weight = severity x source factor x rarity(signal) x recency
  rarity  = how few hosts in the fleet show the signal (IDF): a WARN that every
            host has says nothing about this one
  recency = exponential decay, half-life 7 days
Within one run (or log file) a category counts once, at its strongest event, so
a log repeating one error 500 times is one piece of evidence, not 500.
"""

import bisect
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import hosts
from .signatures import CATEGORIES, SEVERITY_WEIGHT
from .tickets import HC_BOT_PREFIX

DAY = 86400.0
HALF_LIFE_DAYS = 7.0
RECENT_DAYS = 7
# A FAIL that many hosts have had (ports down, query mappings) is still a failure.
FAIL_RARITY_FLOOR = 0.3
SOURCE_FACTOR = {"machine_hc": 1.0, "bmc_reboot": 1.0, "cluster_hc": 0.8}
# Categories that describe the tooling or the host OS rather than the device.
NON_DEVICE = {"infra", "unknown", "host"}
# A chip dropping off PCIe and a reset that can't bring it back are one incident seen from two sides;
# tickets are titled by whichever the filer saw first.
CATEGORY_FAMILY = {"pcie": "pcie_reset", "reset": "pcie_reset"}


def same_family(a: str, b: str) -> bool:
    return CATEGORY_FAMILY.get(a, a) == CATEGORY_FAMILY.get(b, b)


# Machine HC runs are per host, so the same failure on several hosts within minutes is an environment
# event (neighbours resetting their ETH links, a driver rollout), not a fault on each host.
SWEEP_SOURCES = ("machine_hc", "bmc_reboot")
SWEEP_WINDOW_S = 300.0
SWEEP_MIN_HOSTS = 3
SWEEP_FACTOR = 0.25


def fmt_ts(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "?"


@dataclass
class Evidence:
    ts: float | None
    source: str
    origin: str
    category: str
    signal: str
    severity: str
    chips: str
    detail: str
    path: str
    line: int | None
    count: int
    weight: float = 0.0
    runs: int = 1
    first_ts: float | None = None


@dataclass
class CategoryScore:
    category: str
    score: float
    runs: int
    days: int
    chips: str
    evidence: list[Evidence]


@dataclass
class Report:
    host: str
    as_of: float
    window_days: int
    status: str
    status_reason: str
    machine_hc: dict
    cluster_hc: dict
    scores: list[CategoryScore]
    likely: str
    confidence: str
    likely_chips: str = ""
    tickets: list[dict] = field(default_factory=list)
    chip_ordinals: dict[str, str] = field(default_factory=dict)


def base_signal(signal: str) -> str:
    """`eth_link_up:query_mappings_failed` -> `query_mappings_failed`."""
    return signal.rsplit(":", 1)[-1]


def signal_rarity(conn: sqlite3.Connection) -> dict[str, float]:
    """IDF per base signal over the whole index, in [0, 1]."""
    (n_hosts,) = conn.execute("SELECT COUNT(DISTINCT host) FROM events WHERE host IS NOT NULL").fetchone()
    hosts_by_signal: dict[str, set[str]] = defaultdict(set)
    for sig, host in conn.execute("SELECT DISTINCT signal, host FROM events WHERE host IS NOT NULL"):
        hosts_by_signal[base_signal(sig)].add(host)
    denom = math.log(n_hosts + 1) or 1.0
    return {sig: math.log((n_hosts + 1) / (len(h) + 1)) / denom for sig, h in hosts_by_signal.items()}


@dataclass
class FleetStats:
    """Fleet-wide statistics that scoring a single host depends on. Compute once per report."""

    rarity: dict[str, float]
    sweeps: dict[str, tuple[list[float], list[str]]]

    def sweep_hosts(self, signal: str, ts: float) -> int:
        """Distinct hosts with the same signal within SWEEP_WINDOW_S of ts."""
        times, host_list = self.sweeps.get(signal, ([], []))
        lo = bisect.bisect_left(times, ts - SWEEP_WINDOW_S)
        hi = bisect.bisect_right(times, ts + SWEEP_WINDOW_S)
        return len(set(host_list[lo:hi]))


def fleet_stats(conn: sqlite3.Connection) -> FleetStats:
    sweeps: dict[str, tuple[list[float], list[str]]] = {}
    rows = conn.execute(
        f"SELECT ts, host, signal FROM events WHERE source IN ({','.join('?' * len(SWEEP_SOURCES))}) "
        "AND severity = 'fail' AND ts IS NOT NULL AND host IS NOT NULL ORDER BY ts",
        SWEEP_SOURCES,
    )
    for ts, host, sig in rows:
        times, host_list = sweeps.setdefault(base_signal(sig), ([], []))
        times.append(ts)
        host_list.append(host)
    return FleetStats(signal_rarity(conn), sweeps)


def event_weight(
    severity: str, source: str, origin: str, signal: str, age_days: float, rarity: dict[str, float]
) -> float:
    idf = rarity.get(base_signal(signal), 0.5)
    if severity == "fail":
        idf = max(idf, FAIL_RARITY_FLOOR)
    factor = SOURCE_FACTOR.get(source, 0.8) * (0.8 if origin.startswith("log:") else 1.0)
    return SEVERITY_WEIGHT[severity] * factor * idf * 0.5 ** (age_days / HALF_LIFE_DAYS)


def ordinal_map(conn: sqlite3.Connection, host: str) -> dict[int, str]:
    """Driver ordinal -> chip, as most often observed in this host's triage tables."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for ordinal, chip in conn.execute(
        "SELECT ordinal, chip FROM chip_obs WHERE host = ? AND ordinal IS NOT NULL", (host,)
    ):
        votes[ordinal][chip] += 1
    return {o: c.most_common(1)[0][0] for o, c in votes.items()}


def _resolve_chips(chips: str, ordinals: dict[int, str]) -> list[str]:
    # Device 0 is the first one opened, so host-wide failures get reported against it: keep it unresolved.
    out = []
    for c in filter(None, chips.split(",")):
        if c.startswith("dev") and c[3:].isdigit() and int(c[3:]) in ordinals and int(c[3:]) != 0:
            out.append(ordinals[int(c[3:])])
        else:
            out.append(c)
    return out


def score_host(
    conn: sqlite3.Connection, host: str, as_of: float, window_days: int, stats: FleetStats, ordinals: dict[int, str]
) -> list[CategoryScore]:
    rows = conn.execute(
        "SELECT ts, source, origin, category, signal, severity, chips, detail, path, line, count, unit "
        "FROM events WHERE host = ? AND ts BETWEEN ? AND ? AND severity != 'info'",
        (host, as_of - window_days * DAY, as_of),
    ).fetchall()
    unit_best: dict[tuple[str, str], Evidence] = {}
    signal_best: dict[tuple[str, str], Evidence] = {}
    signal_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    signal_first: dict[tuple[str, str], float] = {}
    chip_weight: dict[str, Counter] = defaultdict(Counter)
    for ts, source, origin, cat, sig, sev, chips, detail, path, line, count, unit in rows:
        weight = event_weight(sev, source, origin, sig, max(0.0, (as_of - (ts or as_of)) / DAY), stats.rarity)
        if ts and sev == "fail" and source in SWEEP_SOURCES:
            n_hosts = stats.sweep_hosts(base_signal(sig), ts)
            if n_hosts >= SWEEP_MIN_HOSTS:
                weight *= SWEEP_FACTOR
                detail = f"{detail} [same failure on {n_hosts} hosts within {SWEEP_WINDOW_S / 60:.0f} min]"
        ev = Evidence(
            ts,
            source,
            origin,
            cat,
            sig,
            sev,
            ",".join(_resolve_chips(chips or "", ordinals)),
            detail,
            path,
            line,
            count,
            weight,
        )
        for chip in filter(None, ev.chips.split(",")):
            chip_weight[cat][chip] += weight
        if (unit, cat) not in unit_best or weight > unit_best[(unit, cat)].weight:
            unit_best[(unit, cat)] = ev
        key = (cat, sig)
        signal_units[key].add(unit)
        if ts:
            signal_first[key] = min(signal_first.get(key, ts), ts)
        if key not in signal_best or weight > signal_best[key].weight:
            signal_best[key] = ev
    scores = []
    for cat in {c for _, c in unit_best}:
        best = [ev for (_, c), ev in unit_best.items() if c == cat]
        evidence = sorted((ev for (c, _), ev in signal_best.items() if c == cat), key=lambda e: -e.weight)
        for ev in evidence:
            ev.runs = len(signal_units[(cat, ev.signal)])
            ev.first_ts = signal_first.get((cat, ev.signal))
        scores.append(
            CategoryScore(
                category=cat,
                score=round(sum(ev.weight for ev in best), 3),
                runs=len(best),
                days=len({int((ev.ts or 0) // DAY) for ev in best}),
                chips=hosts.summarize_chips(chip_weight[cat]),
                evidence=evidence[:8],
            )
        )
    return sorted(scores, key=lambda s: -s.score)


def _machine_hc_status(conn: sqlite3.Connection, host: str, as_of: float, window_days: int) -> dict:
    runs = conn.execute(
        "SELECT r.run_id, r.ts_start, r.status, r.summary, r.path, r.extra FROM runs r JOIN run_hosts h USING (run_id) "
        "WHERE h.host = ? AND r.source = 'machine_hc' AND r.ts_start BETWEEN ? AND ? ORDER BY r.ts_start",
        (host, as_of - window_days * DAY, as_of),
    ).fetchall()
    if not runs:
        return {"runs": 0}
    last = runs[-1]
    return {
        "runs": len(runs),
        "counts": dict(Counter(r[2] for r in runs)),
        "last_ts": last[1],
        "last_status": last[2],
        "last_summary": last[3],
        "last_path": last[4],
        "last_pass": max((r[1] for r in runs if r[2] == "PASS"), default=None),
        "last_fail": max((r[1] for r in runs if r[2] == "FAIL"), default=None),
        "history": "".join(r[2][0] for r in runs[-20:]),
        "recent": "".join(r[2][0] for r in runs if r[1] >= as_of - RECENT_DAYS * DAY),
        "overall_status": json.loads(last[5] or "{}").get("overall_status"),
    }


def _cluster_hc_status(conn: sqlite3.Connection, host: str, as_of: float, window_days: int) -> dict:
    out: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT r.kind, r.status, r.ts_end, (SELECT COUNT(*) FROM run_hosts x WHERE x.run_id = r.run_id) "
        "FROM runs r JOIN run_hosts h USING (run_id) WHERE h.host = ? AND r.source = 'cluster_hc' "
        "AND r.ts_end BETWEEN ? AND ? ORDER BY r.ts_end",
        (host, as_of - window_days * DAY, as_of),
    ).fetchall()
    for kind, status, ts, n_hosts in rows:
        k = out.setdefault(kind, {"runs": 0, "failed": 0, "single_host_runs": 0, "single_host_failed": 0})
        k["runs"] += 1
        k["failed"] += status == "FAIL"
        if n_hosts == 1:
            k["single_host_runs"] += 1
            k["single_host_failed"] += status == "FAIL"
            k["single_last_status"], k["single_last_ts"] = status, ts
        k["last_status"], k["last_ts"] = status, ts
    return out


def host_tickets(conn: sqlite3.Connection, host: str) -> list[dict]:
    """Tickets about the host first (`subject`), then ones that only mention it; humans' before the HC bot's."""
    rows = conn.execute(
        "SELECT t.key, t.created, t.status, t.is_open, t.summary, t.primary_category, h.how, h.subject, "
        "(SELECT group_concat(DISTINCT category) FROM ticket_cats c WHERE c.key = t.key) "
        "FROM tickets t JOIN ticket_hosts h USING (key) WHERE h.host = ? "
        "ORDER BY h.subject DESC, substr(t.summary, 1, ?) = ?, t.is_open DESC, t.created DESC",
        (host, len(HC_BOT_PREFIX), HC_BOT_PREFIX),
    ).fetchall()
    out = []
    for k, c, s, o, su, p, how, subject, cats in rows:
        evidence = conn.execute(
            "SELECT category, signal, severity, chips, detail, path, line, count FROM ticket_evidence "
            "WHERE key = ? AND (host = ? OR host IS NULL) AND severity != 'info' "
            "ORDER BY severity = 'fail' DESC, count DESC LIMIT 3",
            (k, host),
        ).fetchall()
        out.append(
            {
                "key": k,
                "created": c,
                "status": s,
                "open": bool(o),
                "summary": su,
                "primary": p,
                "how": how,
                "subject": bool(subject),
                "bot": su.startswith(HC_BOT_PREFIX),
                "categories": sorted(set((cats or "").split(",")) - {""}),
                "attachment_evidence": [
                    dict(zip(("category", "signal", "severity", "chips", "detail", "path", "line", "count"), e))
                    for e in evidence
                ],
            }
        )
    return out


def _status(mhc: dict, chc: dict, top: CategoryScore | None, as_of: float) -> tuple[str, str]:
    if mhc.get("runs"):
        last = f"last machine HC {fmt_ts(mhc['last_ts'])} {mhc['last_status']}: {mhc['last_summary']}"
        recent_fails = mhc["recent"].count("F")
        if mhc["last_status"] == "FAIL":
            return "FAILING", last
        if recent_fails >= 2:
            return "FLAKY", f"{last}; but {recent_fails} of {len(mhc['recent'])} runs in the last {RECENT_DAYS}d failed"
        if (as_of - mhc["last_ts"]) / DAY <= 3:
            if mhc["last_status"] == "PASS" or top is None:
                return "HEALTHY", last + ("" if mhc["last_status"] == "PASS" else " (fleet-common warnings only)")
            return "DEGRADED", last
    for kind, v in chc.items():
        if v.get("single_last_status") == "FAIL" and as_of - v["single_last_ts"] <= RECENT_DAYS * DAY:
            return "FAILING", f"last single-host cluster {kind} {fmt_ts(v['single_last_ts'])} failed"
    if top and top.score >= 3:
        return "DEGRADED", f"recent {CATEGORIES[top.category]} evidence (score {top.score:g})"
    if not mhc.get("runs") and not chc:
        return "UNKNOWN", "no HC runs in the window"
    return "HEALTHY", "no recent failures"


def _likely(scores: list[CategoryScore]) -> tuple[str, str, str]:
    device = [s for s in scores if s.category not in NON_DEVICE]
    if not device or device[0].score < 0.5:
        return "none", "-", ""
    top = device[0]
    second = device[1].score if len(device) > 1 else 0.0
    ratio = top.score / second if second else math.inf
    if ratio >= 2 and top.runs >= 2:
        confidence = "high"
    elif ratio >= 1.3 or top.runs >= 3:
        confidence = "medium"
    else:
        confidence = "low"
    return top.category, confidence, top.chips


def triage(
    conn: sqlite3.Connection,
    host: str,
    as_of: float | None = None,
    window_days: int = 30,
    stats: FleetStats | None = None,
) -> Report:
    host = hosts.normalize_host(host) or host
    as_of = as_of or time.time()
    stats = stats or fleet_stats(conn)
    ordinals = ordinal_map(conn, host)
    scores = score_host(conn, host, as_of, window_days, stats, ordinals)
    likely, confidence, chips = _likely(scores)
    mhc = _machine_hc_status(conn, host, as_of, window_days)
    chc = _cluster_hc_status(conn, host, as_of, window_days)
    status, reason = _status(mhc, chc, next((s for s in scores if s.category == likely), None), as_of)
    tickets = host_tickets(conn, host) if _has_tickets(conn) else []
    return Report(
        host,
        as_of,
        window_days,
        status,
        reason,
        mhc,
        chc,
        scores,
        likely,
        confidence,
        chips,
        tickets,
        {f"dev{o}": c for o, c in sorted(ordinals.items())},
    )


def _has_tickets(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'tickets'").fetchone())


def render(report: Report, verbose: bool = False) -> str:
    r = report
    lines = [
        f"{r.host}   status: {r.status}   (as of {fmt_ts(r.as_of)} UTC, window {r.window_days}d)",
        f"  why: {r.status_reason}",
    ]
    m = r.machine_hc
    if m.get("runs"):
        lines.append(
            f"  machine HC: {m['runs']} runs {m['counts']}; last PASS {fmt_ts(m['last_pass'])}, "
            f"last FAIL {fmt_ts(m['last_fail'])}; oldest->newest {m['history']}"
        )
        lines.append(f"    latest run: {m['last_path']}  (suite verdict {m['overall_status']})")
    else:
        lines.append("  machine HC: no runs in window")
    for kind, v in sorted(r.cluster_hc.items()):
        lines.append(
            f"  cluster HC {kind}: {v['failed']}/{v['runs']} failed (single-host {v['single_host_failed']}/"
            f"{v['single_host_runs']}); last {v['last_status']} {fmt_ts(v['last_ts'])}"
        )
    lines.append("")
    if r.likely == "none":
        lines.append("LIKELY ISSUE: none (no device-level failure evidence in the window)")
    else:
        lines.append(
            f"LIKELY ISSUE: {CATEGORIES[r.likely]}   confidence {r.confidence}   "
            f"where: {r.likely_chips or 'not localized to chips'}"
        )
        if r.likely_chips == "all":
            lines.append(
                "  note: every chip is affected, which points at firmware, telemetry or the check itself "
                "rather than one bad part"
            )
    lines.append("SCORES: " + "  ".join(f"{s.category}={s.score:g} ({s.runs} runs, {s.days}d)" for s in r.scores[:8]))
    for s in r.scores[: (6 if verbose else 3)]:
        if s.score <= 0:
            continue
        lines.append(f"  [{CATEGORIES[s.category]}] chips: {s.chips or '-'}")
        for ev in s.evidence[: (6 if verbose else 3)]:
            span = fmt_ts(ev.ts) if ev.runs == 1 else f"{fmt_ts(ev.first_ts)} .. {fmt_ts(ev.ts)}, {ev.runs} runs"
            lines.append(f"    {ev.signal} ({ev.severity}, {ev.source}/{ev.origin}) {span}")
            chips = f" [{hosts.summarize_chips(dict.fromkeys(ev.chips.split(','), 1.0))}]" if ev.chips else ""
            lines.append(f"        {ev.detail[:200]}{chips}")
            if verbose:
                lines.append(f"        {ev.path}" + (f":{ev.line}" if ev.line else ""))
    about = [t for t in r.tickets if t["subject"] and not t["bot"]]
    bot = [t for t in r.tickets if t["subject"] and t["bot"]]
    mentioned = [t for t in r.tickets if not t["subject"]]
    if about:
        lines += ["", "TICKETS about this host:"]
        for t in about[:10]:
            match = "  <- same issue" if any(same_family(r.likely, c) for c in t["categories"]) else ""
            how = "" if t["how"] == "full" else f" ({t['how']})"
            lines.append(
                f"  {'OPEN' if t['open'] else 'done'} {t['key']:<10} {fmt_ts(t['created'])[:10]} [{t['status']}] "
                f"{t['summary'][:100]}{how}  cats={','.join(t['categories']) or '-'}{match}"
            )
            for e in t["attachment_evidence"]:
                chips = f" [{e['chips']}]" if e["chips"] else ""
                lines.append(
                    f"      attachment: {e['category']}/{e['signal']}{chips} {e['detail'][:120]}"
                    f"  ({e['path'].rsplit('/', 1)[-1]}:{e['line']})"
                )
    if bot:
        open_bot = [t["key"] for t in bot if t["open"]]
        lines.append(
            f"HC bot auto-tickets: {len(bot)} (latest {bot[0]['key']} {fmt_ts(bot[0]['created'])[:10]}"
            + (f"; open: {', '.join(open_bot)}" if open_bot else "")
            + ")"
        )
    if mentioned:
        lines.append(
            f"Also mentioned in: {', '.join(t['key'] for t in mentioned[:10])}"
            + (f" +{len(mentioned) - 10}" if len(mentioned) > 10 else "")
        )
    return "\n".join(lines)


def report_json(report: Report) -> str:
    return json.dumps(asdict(report), indent=1, default=str)
