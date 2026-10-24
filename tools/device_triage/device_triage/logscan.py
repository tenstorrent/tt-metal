# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Line-oriented log scanning shared by every unstructured source."""

import re
from collections.abc import Iterable
from datetime import datetime, timezone

from . import hosts
from .config import MAX_EVIDENCE_CHARS, MAX_LINES_PER_FILE
from .model import Event
from .signatures import TARGETED_SIGNALS, match_line

# `[bh-glx-110-d03u02][16:25:40] ...` and `[bh-glx-110-d03u02:00000][16:28:20] ...`
PREFIX_RE = re.compile(r"^\[([A-Za-z0-9.-]+)(?::\d+)?\]\[(\d\d):(\d\d):(\d\d)\]\s?")
TTLOG_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)")
KLOG_TS_RE = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def utc_ts(text: str, fmt: str) -> float | None:
    try:
        return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def iso_ts(text: str | None) -> float | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_lines(path: str) -> list[str]:
    with open(path, errors="replace") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= MAX_LINES_PER_FILE:
                break
            lines.append(line.rstrip("\n"))
    return lines


def resolve_chips(line: str, match: re.Match, bdf_map: dict[str, str], ordinals: dict[int, str]) -> set[str]:
    chips = {chip for _, chip in hosts.bdf_chip_pairs(line)}
    chips |= {hosts.chip_label(m.group(1), m.group(2)) for m in hosts.UC_RE.finditer(line)}
    chips |= {bdf_map[b.lower()] for b in hosts.BDF_RE.findall(line) if b.lower() in bdf_map}
    dev = match.groupdict().get("dev")
    if dev is not None:
        chips.add(ordinals.get(int(dev), f"dev{int(dev)}"))
    return chips


def scan_lines(
    lines: Iterable[str],
    *,
    default_host: str | None,
    base_ts: float | None,
    source: str,
    origin: str,
    path: str,
    run_id: str | None = None,
    bdf_map: dict[str, str] | None = None,
    ordinals: dict[int, str] | None = None,
) -> list[Event]:
    """Signature hits in `lines`, aggregated per (host, signal, chips) with a count.

    Host attribution: hosts named inside the line (validation endpoints) win, then
    the `[host][time]` prefix, then `default_host`.
    """
    bdf_map = bdf_map or {}
    ordinals = ordinals or {}
    day0 = (base_ts // 86400) * 86400 if base_ts else None
    day_offset, last_sod = 0, None
    agg: dict[tuple, Event] = {}
    for lineno, raw in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw)
        host, ts = default_host, base_ts
        m = PREFIX_RE.match(line)
        if m:
            host = hosts.normalize_host(m.group(1)) or default_host
            sod = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            if last_sod is not None and sod < last_sod - 3600:
                day_offset += 86400
            last_sod = sod
            if day0 is not None:
                ts = day0 + day_offset + sod
            line = line[m.end() :]
        hit = match_line(line)
        if hit is None:
            continue
        sig, match = hit
        tm = TTLOG_TS_RE.match(line)
        if tm:
            ts = utc_ts(f"{tm.group(1)} {tm.group(2)}", "%Y-%m-%d %H:%M:%S") or ts
        else:
            km = KLOG_TS_RE.match(line)
            if km:
                ts = utc_ts(" ".join(km.group(1).split()), "%a %b %d %H:%M:%S %Y") or ts
        targets: dict[str | None, set[str]] = {}
        for ep_host, chip in hosts.endpoint_chips(line):
            targets.setdefault(ep_host, set()).add(chip)
        if not targets:
            named = hosts.find_hosts(line) - {host}
            if sig.name in TARGETED_SIGNALS and len(named) == 1:
                host = named.pop()
            targets[host] = resolve_chips(line, match, bdf_map, ordinals)
        for target, chips in targets.items():
            chip_str = ",".join(sorted(chips))
            key = (target, sig.name, chip_str)
            ev = agg.get(key)
            if ev is None:
                agg[key] = Event(
                    host=target,
                    ts=ts,
                    source=source,
                    origin=origin,
                    category=sig.category,
                    signal=sig.name,
                    severity=sig.severity,
                    detail=line.strip()[:MAX_EVIDENCE_CHARS],
                    chips=chip_str,
                    scope=hosts.scope_of(chips),
                    run_id=run_id,
                    path=path,
                    line=lineno,
                )
            else:
                ev.count += 1
    return list(agg.values())
