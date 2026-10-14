# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Triage one host straight from the logs, without the index.

Only that host's runs are parsed, into an in-memory database with the index schema, so scoring and rendering are
the same code as `triage --index`. Live mode sees one host, so signal weights come from a fleet snapshot
(`signal_rarity.json`) and failures shared by many hosts at once (sweeps) aren't recognised. Tickets come live from
Jira when a token is configured, else from a `jira-sync` cache if there is one.
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from . import index, jira_sync, tickets, triage
from .config import (
    BMC_REBOOT_ROOT,
    CLUSTER_REPORT_ROOTS,
    DCAMP_HC_ROOT,
    JIRA_ENV,
    MACHINE_HC_ROOT,
    RECOVER_LOG_ROOT,
    STACKSTORM_ROOT,
)

RARITY_SNAPSHOT = Path(__file__).with_name("signal_rarity.json")
LIVE_TICKET_LIMIT = 100
RECOVER_DIR_DATE_RE = re.compile(r"-(\d{8})T\d{6}Z")
# Unit patterns of the cluster HC sources (as the index discovers them); files are found by the host name in them.
CLUSTER_PATTERNS = (
    ("cluster_report", [str(root / "*/*/*.json") for root in CLUSTER_REPORT_ROOTS]),
    (
        "cluster_log",
        [
            str(RECOVER_LOG_ROOT / "*/recover_*.log"),
            str(STACKSTORM_ROOT / "*/logs/recover-*.log"),
            str(STACKSTORM_ROOT / "nightly-health-check/*/physical_validation/*/*/cluster_validation_iteration_*.log"),
        ],
    ),
)


def fleet_stats() -> triage.FleetStats:
    return triage.FleetStats(json.loads(RARITY_SNAPSHOT.read_text())["rarity"], {})


def _run_dirs(root: Path, host: str, since_day: str) -> list[str]:
    """`<root>/<host>/<date>/<run>/` directories with a diag_report.json, dated `since_day` or later."""
    return [
        str(p.parent) for p in sorted((root / host).glob("*/*/diag_report.json")) if p.parent.parent.name >= since_day
    ]


def _files_naming(host: str, roots: list[Path], since_ts: float) -> list[Path]:
    roots = [str(r) for r in roots if r.exists()]
    if not roots:
        return []
    if shutil.which("rg"):
        cmd = ["rg", "-l", "-F", "--no-messages", "-g", "*.json", "-g", "*.log", host, *roots]
    else:
        cmd = ["grep", "-rlF", "--include=*.json", "--include=*.log", host, *roots]
    found = subprocess.run(["nice", "-n", "10", *cmd], capture_output=True, text=True).stdout.splitlines()
    out = []
    for path in map(Path, found):
        try:
            if path.stat().st_mtime >= since_ts and os.access(path, os.R_OK):
                out.append(path)
        except OSError:
            continue
    return out


def _cluster_search_roots(since_day: str, since_ts: float) -> list[Path]:
    """Only directories that can hold files from the window: report `<kind>/<date>/`, recover `<user>-<date>T...`."""
    roots = [d for root in CLUSTER_REPORT_ROOTS for d in root.glob("*/*") if d.name >= since_day]
    compact = since_day.replace("-", "")
    for d in RECOVER_LOG_ROOT.iterdir() if RECOVER_LOG_ROOT.is_dir() else []:
        m = RECOVER_DIR_DATE_RE.search(d.name)
        if not m or m.group(1) >= compact:
            roots.append(d)
    for d in STACKSTORM_ROOT.glob("*/logs"):
        try:
            if d.stat().st_mtime >= since_ts:
                roots.append(d)
        except OSError:
            continue
    return roots + [STACKSTORM_ROOT / "nightly-health-check"]


def host_units(host: str, as_of: float, days: int) -> list[tuple[str, str]]:
    """(source, unit) for every run of `host` that can fall in the window."""
    since_ts = as_of - (days + 1) * triage.DAY
    since_day = datetime.fromtimestamp(since_ts, timezone.utc).strftime("%Y-%m-%d")
    units = [("machine_hc", d) for d in _run_dirs(MACHINE_HC_ROOT, host, since_day)]
    units += [("bmc_reboot", d) for d in _run_dirs(BMC_REBOOT_ROOT, host, since_day)]
    units += [
        ("dcamp_hc", str(p.parent))
        for p in sorted((DCAMP_HC_ROOT / host).glob("*/*/ticket.json"))
        if p.parent.parent.name >= since_day
    ]
    for path in _files_naming(host, _cluster_search_roots(since_day, since_ts), since_ts):
        source = next((s for s, patterns in CLUSTER_PATTERNS if any(path.match(p) for p in patterns)), None)
        if source:
            units.append((source, str(path)))
    return units


def _siblings(host: str) -> list[str]:
    """The same rack/slot in the other superpod: a short name (`b02u08`) in a ticket may mean either."""
    m = re.fullmatch(r"bh-glx-(\d{3})-([a-d]\d{2}u\d{2})", host)
    return [f"bh-glx-{pod}-{m.group(2)}" for pod in ("110", "120") if pod != m.group(1)] if m else []


def host_tickets(host: str) -> list[dict]:
    """Tickets mentioning the host: live from Jira with a token, else from a jira-sync cache."""
    suffix = host.rsplit("-", 1)[-1].lower()
    if not JIRA_ENV.exists():
        return [t for t in tickets.load_cached_tickets() if suffix in t["text"].lower()]
    client = jira_sync.Client(*jira_sync.load_credentials())
    # Jira doesn't split `bh-glx-120-b03u02` into words: search the full name and the short one (`b03u02`).
    jql = f'(project = DIIM OR project = DCAMP) AND (text ~ "{host}" OR text ~ "{suffix}") ORDER BY created DESC'
    out = []
    try:
        for issue in client.search(jql, jira_sync.FIELDS):
            comments = (issue["fields"].get("comment") or {}).get("comments", [])
            out.append(tickets.ticket_from_issue(jira_sync.normalize(issue, comments)))
            if len(out) >= LIVE_TICKET_LIMIT:
                break
    except jira_sync.JiraError as exc:
        print(f"Jira unavailable, tickets not shown: {exc}", file=sys.stderr)
    return out


def connect(host: str, as_of: float, days: int, workers: int = 8) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(index.SCHEMA)
    units = host_units(host, as_of, days)
    source_of = dict((unit, source) for source, unit in units)
    with Pool(max(1, min(workers, len(units) // 8))) as pool:
        for res in pool.imap_unordered(index.parse_unit, units, chunksize=4):
            index.store_unit(conn, source_of[res.unit], res, 0.0, 0, "live")
    conn.executescript(tickets.SCHEMA)
    tickets.insert_tickets(conn, host_tickets(host), {host, *_siblings(host)}, tickets.load_jql_categories())
    return conn
