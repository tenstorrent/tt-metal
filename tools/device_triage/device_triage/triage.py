# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-device status and likely issue, from one host's health-check runs.

Scoring, per category over a time window ending at `as_of`:
  event weight = severity x source factor x rarity(signal) x recency
  rarity  = how few hosts in the fleet show the signal (IDF, from the shipped snapshot): a WARN that every
            host has says nothing about this one
  recency = exponential decay, half-life 7 days
Within one run (or log file) a category counts once, at its strongest event, so
a log repeating one error 500 times is one piece of evidence, not 500.
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import hosts
from .model import Event, HostData, Run
from .signatures import CATEGORIES, SEVERITY_WEIGHT

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
RARITY_SNAPSHOT = Path(__file__).with_name("signal_rarity.json")


def same_family(a: str, b: str) -> bool:
    return CATEGORY_FAMILY.get(a, a) == CATEGORY_FAMILY.get(b, b)


def fmt_ts(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "?"


def load_rarity() -> dict[str, float]:
    return json.loads(RARITY_SNAPSHOT.read_text())["rarity"]


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


def event_weight(
    severity: str, source: str, origin: str, signal: str, age_days: float, rarity: dict[str, float]
) -> float:
    idf = rarity.get(base_signal(signal), 0.5)
    if severity == "fail":
        idf = max(idf, FAIL_RARITY_FLOOR)
    factor = SOURCE_FACTOR.get(source, 0.8) * (0.8 if origin.startswith("log:") else 1.0)
    return SEVERITY_WEIGHT[severity] * factor * idf * 0.5 ** (age_days / HALF_LIFE_DAYS)


def ordinal_map(chip_map: list[tuple[str, str, str, int | None]], host: str) -> dict[int, str]:
    """Driver ordinal -> chip, as most often observed in this host's triage tables."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for h, _bdf, chip, ordinal in chip_map:
        if h == host and ordinal is not None:
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


def _in_window(ts: float | None, as_of: float, window_days: int) -> bool:
    return ts is not None and as_of - window_days * DAY <= ts <= as_of


def score_host(
    events: list[tuple[str, Event]],
    host: str,
    as_of: float,
    window_days: int,
    rarity: dict[str, float],
    ordinals: dict[int, str],
) -> list[CategoryScore]:
    unit_best: dict[tuple[str, str], Evidence] = {}
    signal_best: dict[tuple[str, str], Evidence] = {}
    signal_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    signal_first: dict[tuple[str, str], float] = {}
    chip_weight: dict[str, Counter] = defaultdict(Counter)
    for unit, e in events:
        if e.host != host or e.severity == "info" or not _in_window(e.ts, as_of, window_days):
            continue
        weight = event_weight(e.severity, e.source, e.origin, e.signal, (as_of - e.ts) / DAY, rarity)
        ev = Evidence(
            e.ts,
            e.source,
            e.origin,
            e.category,
            e.signal,
            e.severity,
            ",".join(_resolve_chips(e.chips or "", ordinals)),
            e.detail,
            e.path,
            e.line,
            e.count,
            weight,
        )
        for chip in filter(None, ev.chips.split(",")):
            chip_weight[ev.category][chip] += weight
        if (unit, ev.category) not in unit_best or weight > unit_best[(unit, ev.category)].weight:
            unit_best[(unit, ev.category)] = ev
        key = (ev.category, ev.signal)
        signal_units[key].add(unit)
        signal_first[key] = min(signal_first.get(key, e.ts), e.ts)
        if key not in signal_best or weight > signal_best[key].weight:
            signal_best[key] = ev
    scores = []
    for cat in {c for _, c in unit_best}:
        best = [ev for (_, c), ev in unit_best.items() if c == cat]
        evidence = sorted((ev for (c, _), ev in signal_best.items() if c == cat), key=lambda e: -e.weight)
        for ev in evidence:
            ev.runs = len(signal_units[(cat, ev.signal)])
            ev.first_ts = signal_first.get((cat, ev.signal))
        # Where = the chips of the strongest evidence; summing every event would let chips that checks merely
        # list (the first BDF of a table) outweigh the tray that is actually missing. Unless the category hits
        # every tray: a check that lists only its first few failing chips must still read "all".
        lead = next((ev for ev in evidence if ev.chips), None)
        overall = hosts.summarize_chips(chip_weight[cat])
        chips = (
            overall
            if overall == "all" or not lead
            else hosts.summarize_chips(dict.fromkeys(lead.chips.split(","), 1.0))
        )
        scores.append(
            CategoryScore(
                category=cat,
                score=round(sum(ev.weight for ev in best), 3),
                runs=len(best),
                days=len({int((ev.ts or 0) // DAY) for ev in best}),
                chips=chips,
                evidence=evidence[:8],
            )
        )
    return sorted(scores, key=lambda s: -s.score)


def _machine_hc_status(runs: list[Run], host: str, as_of: float, window_days: int) -> dict:
    mhc = sorted(
        (
            r
            for r in runs
            if r.source == "machine_hc" and host in r.hosts and _in_window(r.ts_start, as_of, window_days)
        ),
        key=lambda r: r.ts_start,
    )
    if not mhc:
        return {"runs": 0}
    last = mhc[-1]
    return {
        "runs": len(mhc),
        "counts": dict(Counter(r.status for r in mhc)),
        "last_ts": last.ts_start,
        "last_status": last.status,
        "last_summary": last.summary,
        "last_path": last.path,
        "last_pass": max((r.ts_start for r in mhc if r.status == "PASS"), default=None),
        "last_fail": max((r.ts_start for r in mhc if r.status == "FAIL"), default=None),
        "history": "".join(r.status[0] for r in mhc[-20:]),
        "recent": "".join(r.status[0] for r in mhc if r.ts_start >= as_of - RECENT_DAYS * DAY),
        "overall_status": last.extra.get("overall_status"),
    }


def _cluster_hc_status(runs: list[Run], host: str, as_of: float, window_days: int) -> dict:
    out: dict[str, dict] = {}
    chc = sorted(
        (r for r in runs if r.source == "cluster_hc" and host in r.hosts and _in_window(r.ts_end, as_of, window_days)),
        key=lambda r: r.ts_end,
    )
    for r in chc:
        k = out.setdefault(r.kind, {"runs": 0, "failed": 0, "single_host_runs": 0, "single_host_failed": 0})
        k["runs"] += 1
        k["failed"] += r.status == "FAIL"
        if len(r.hosts) == 1:
            k["single_host_runs"] += 1
            k["single_host_failed"] += r.status == "FAIL"
            k["single_last_status"], k["single_last_ts"] = r.status, r.ts_end
        k["last_status"], k["last_ts"] = r.status, r.ts_end
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
    host: str,
    data: HostData,
    tickets: list[dict],
    as_of: float,
    window_days: int = 30,
    rarity: dict[str, float] | None = None,
) -> Report:
    rarity = rarity if rarity is not None else load_rarity()
    ordinals = ordinal_map(data.chip_map, host)
    scores = score_host(data.events, host, as_of, window_days, rarity, ordinals)
    likely, confidence, chips = _likely(scores)
    mhc = _machine_hc_status(data.runs, host, as_of, window_days)
    chc = _cluster_hc_status(data.runs, host, as_of, window_days)
    status, reason = _status(mhc, chc, next((s for s in scores if s.category == likely), None), as_of)
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


def render(report: Report, verbose: bool = False) -> str:
    r = report
    rule = "=" * 100
    lines = [
        rule,
        f"{r.host}   status: {r.status}   (as of {fmt_ts(r.as_of)} UTC, window {r.window_days}d)",
        rule,
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
