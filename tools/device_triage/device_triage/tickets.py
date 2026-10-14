# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Jira tickets in the index: hosts they name, categories they belong to.

Tickets come from `cache/jira/issues/*.json` (`jira-sync`, full text) or an older summary-only listing in
`cache/jira/mcp/*.tsv`. Category membership by JQL (summary + description) comes from `jql_membership.json`.
"""

import itertools
import json
import re
import sqlite3
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from . import hosts, index, logscan
from .config import JIRA_CACHE, JQL_MEMBERSHIP, JQL_QUERIES, MAX_LINES_PER_FILE

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (key TEXT PRIMARY KEY, project TEXT, created REAL, resolved REAL, status TEXT,
    is_open INTEGER, summary TEXT, text TEXT, primary_category TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS ticket_hosts (key TEXT, host TEXT, how TEXT, subject INTEGER, PRIMARY KEY (key, host));
CREATE TABLE IF NOT EXISTS ticket_cats (key TEXT, category TEXT, source TEXT, PRIMARY KEY (key, category, source));
CREATE TABLE IF NOT EXISTS ticket_refs (key TEXT, ref TEXT, PRIMARY KEY (key, ref));
CREATE TABLE IF NOT EXISTS ticket_evidence (key TEXT, host TEXT, category TEXT, signal TEXT, severity TEXT,
    chips TEXT, detail TEXT, path TEXT, line INTEGER, count INTEGER);
CREATE INDEX IF NOT EXISTS ticket_hosts_host ON ticket_hosts(host);
CREATE TABLE IF NOT EXISTS attachment_scans (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, version TEXT,
    events TEXT);
"""
ATTACHMENTS_DIR = JIRA_CACHE / "attachments"
TEXT_SUFFIXES = {".log", ".txt", ".out", ".json", ".jsonl", ".csv", ".yaml", ".yml", ""}

CLOSED_STATUSES = {"done", "closed", "resolved", "won't do", "wont do", "cancelled", "duplicate"}

# Ticket text -> categories. Order is priority: the first match is the primary
# category, so specific causes (GDDR, Tensix) win over symptoms (reset, down).
TEXT_CATEGORIES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (cat, re.compile(rx, re.I))
    for cat, rx in (
        ("gddr", r"gddr|\bdram\b(?! ?machine)|\bddr\b|memory training|\bbist\b|bad dram"),
        ("tensix", r"tensix|non-?determin|\bND\b|\bpcc\b|aiclk|\bnoc ?0?\b|\bL1\b|compute failure|corrupts"),
        (
            "pcie",
            r"pcie|\bpci\b|enumerat|drops? off|\b\d{1,2} ?/ ?32\b|only \d+ chips|missing \d+ chips|chips? missing|"
            r"chip not working|not detected|no asics|no tenstorrent devices|detected chips|devices? (?:missing|not found)|"
            r"detecting devices|board should be reset|pciehang|31 chips|24 chips|fewer chips",
        ),
        ("bmc", r"\bbmc\b|ipmi|redfish|bmc[ _-]?bots?"),
        ("reset", r"reset|post_reset|\bstuck (?:in|on) post\b|power[ -]?cycle|recycle|cpld"),
        (
            "power_thermal",
            r"power issues|\bpsu\b|thermal|breaker|\bpdu\b|\btdp\b|temp_|fans?\b|cooling|power[- ]off|"
            r"won'?t (?:turn on|power)|pgood",
        ),
        (
            "eth",
            r"\beth(?:ernet)?\b|\blinks?\b|qsfp|cables?\b|cabling|connections?\b|fabric|torus|topology|link training|\bfec\b",
        ),
        ("arc_fw", r"\barc\b|firmware|\bfw\b|tt-flash|\bflash\b"),
        (
            "host",
            r"\bssh\b|unreachable|not reachable|\bdown\*?|offline|disk|\bnfs\b|space|\bmce\b|kernel|\bdns\b|network|"
            r"login|weka|out of memory|\boom\b|nvme|reboots?|crash",
        ),
        ("infra", r"slurm|provision|access\b|install|script|weights|\bci\b|runner|descriptor|recabl|request"),
    )
)
# JQL membership whose terms are specific enough to trust as a category label.
TRUSTED_JQL = {"gddr", "tensix", "pcie"}
# ticket_hosts.how for a host the title names unambiguously. Hosts only in the description are "description"
# (they often list a whole quad around the one affected host); hosts only in comments are "comment".
NAMED_HOWS = ("full", "bracket", "quad", "short")
# A ticket is about (`subject`) the hosts its title names, else the hosts its description names if at most
# this many; longer lists are fleet-wide tickets (KMD versions, Weka, SP-wide logs), not about each host.
MAX_DESCRIPTION_SUBJECTS = 4
# exabox-ci-jira-bot's machine HC auto-tickets: shown per host, but nobody has triaged them yet.
HC_BOT_PREFIX = "[Fabric Health Check]"

BRACKET_HOST_RE = re.compile(r"(?:bh-glx-)?(110|120)-([a-d][0-9\[\]/,]*u[0-9\[\]/,]+)", re.I)
SUPERPOD_HOST_RE = re.compile(r"(?<![\w-])(110|120)-([a-d]\d{2}u\d{2})(?![\w])", re.I)
PREFIX_SHORT_RE = re.compile(r"bh-glx-([a-d]\d{2}u\d{2})(?![\w])", re.I)
# "bh-glx-120-c01u02/u08/u20": one rack, several slots.
SLASH_SLOTS_RE = re.compile(r"(?:bh-glx-)?(110|120)-([a-d]\d{2})u(\d{2})((?:/u\d{2})+)(?![\w])", re.I)


def expand_brackets(token: str) -> list[str]:
    """`a0[6/7]u0[2,8]` -> a06u02, a06u08, a07u02, a07u08."""
    parts = re.split(r"(\[[^\]]*\])", token)
    choices = [re.split(r"[,/]", p[1:-1]) if p.startswith("[") else [p] for p in parts]
    return ["".join(c) for c in itertools.product(*choices)]


def text_categories(text: str) -> list[str]:
    return [cat for cat, rx in TEXT_CATEGORIES if rx.search(text)]


def ticket_host_map(text: str, known: set[str]) -> dict[str, str]:
    found = hosts.ticket_hosts(text, known)
    for m in BRACKET_HOST_RE.finditer(text):
        if "[" not in m.group(2):
            continue
        for suffix in expand_brackets(m.group(2).lower()):
            if re.fullmatch(r"[a-d]\d{2}u\d{2}", suffix):
                found.setdefault(f"bh-glx-{m.group(1)}-{suffix}", "bracket")
    for m in SUPERPOD_HOST_RE.finditer(text):
        found[f"bh-glx-{m.group(1)}-{m.group(2).lower()}"] = "full"
    for m in SLASH_SLOTS_RE.finditer(text):
        pod, rack = m.group(1), m.group(2).lower()
        for slot in [m.group(3), *re.findall(r"\d{2}", m.group(4))]:
            found.setdefault(f"bh-glx-{pod}-{rack}u{slot}", "full")
    # "bh-glx-c08u02" (no superpod): treat like a short name.
    stripped = PREFIX_SHORT_RE.sub(lambda m: " " + m.group(1), text)
    if stripped != text:
        for h, how in hosts.ticket_hosts(stripped, known).items():
            found.setdefault(h, how)
    return found


def _parse_jira_ts(text: str | None) -> float | None:
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def load_cached_tickets() -> list[dict]:
    """Full issues from a REST sync if present, else the MCP summary listing."""
    issues_dir = JIRA_CACHE / "issues"
    tickets: dict[str, dict] = {}
    for path in sorted((JIRA_CACHE / "mcp").glob("*_p*.tsv")):
        project = path.name.split("_", 1)[0]
        for line in path.read_text().splitlines():
            key, created, status, summary = (line.split("\t", 3) + ["", "", ""])[:4]
            tickets[key] = {
                "key": key,
                "project": project,
                "created": _parse_jira_ts(created),
                "resolved": None,
                "status": status,
                "summary": summary,
                "description": "",
                "comments": "",
                "text": summary,
                "source": "mcp",
            }
    if issues_dir.is_dir():
        for path in issues_dir.glob("*.json"):
            issue = json.loads(path.read_text())
            summary, description = issue.get("summary", ""), issue.get("description", "")
            comments = "\n".join(c.get("body", "") for c in issue.get("comments", []))
            tickets[issue["key"]] = {
                "key": issue["key"],
                "project": issue["key"].split("-")[0],
                "created": _parse_jira_ts(issue.get("created")),
                "resolved": _parse_jira_ts(issue.get("resolved")),
                "status": issue.get("status", ""),
                "summary": summary,
                "description": description,
                "comments": comments,
                "text": "\n".join([summary, description, comments]),
                "source": "rest",
            }
    return list(tickets.values())


def ticket_hosts_by_part(t: dict, known: set[str]) -> dict[str, str]:
    found = ticket_host_map(t["summary"], known)
    for part, tag in (("description", "description"), ("comments", "comment")):
        for h, how in ticket_host_map(t[part], known).items():
            if how in NAMED_HOWS and h not in found:
                found[h] = tag
    return found


def subjects(found: dict[str, str]) -> set[str]:
    named = {h for h, how in found.items() if how in NAMED_HOWS}
    described = {h for h, how in found.items() if how == "description"}
    return named or (described if len(described) <= MAX_DESCRIPTION_SUBJECTS else set())


def subject_hosts(conn: sqlite3.Connection, key: str) -> list[str]:
    return [
        h for (h,) in conn.execute("SELECT host FROM ticket_hosts WHERE key = ? AND subject = 1 ORDER BY host", (key,))
    ]


def jql_membership() -> dict[str, dict]:
    """category -> {"jql", "keys"}; keys are empty until jira-sync has run the queries."""
    if JQL_MEMBERSHIP.exists():
        return json.loads(JQL_MEMBERSHIP.read_text())["categories"]
    legacy = JIRA_CACHE / "mcp" / "DIIM_jql_categories.json"
    if legacy.exists():
        return {
            cat: {"jql": e["jql"], "keys": [f"DIIM-{n}" for n in e["keys"]]}
            for cat, e in json.loads(legacy.read_text())["categories"].items()
        }
    return {cat: {"jql": jql, "keys": []} for cat, jql in json.loads(JQL_QUERIES.read_text())["queries"].items()}


def load_jql_categories() -> dict[str, set[str]]:
    return {cat: set(entry["keys"]) for cat, entry in jql_membership().items()}


def _attachment_texts(path: Path) -> Iterator[tuple[str, list[str]]]:
    """(display path, lines) for a text attachment or each text member of a zip."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith("/") and Path(name).suffix.lower() in TEXT_SUFFIXES:
                    text = zf.read(name).decode("utf-8", "replace")
                    yield f"{path}:{name}", text.splitlines()[:MAX_LINES_PER_FILE]
    elif path.suffix.lower() in TEXT_SUFFIXES:
        yield str(path), path.read_text(errors="replace").splitlines()[:MAX_LINES_PER_FILE]


def _scan_attachment(path: Path) -> list[list]:
    """[host, category, signal, severity, chips, detail, display path, line, count] per hit; host only if a line names it."""
    return [
        [e.host, e.category, e.signal, e.severity, e.chips, e.detail, e.path, e.line, e.count]
        for display, lines in _attachment_texts(path)
        for e in logscan.scan_lines(
            lines, default_host=None, base_ts=None, source="jira_attachment", origin="log:attachment", path=display
        )
    ]


def scan_attachments(conn: sqlite3.Connection) -> int:
    """Signature hits in downloaded ticket attachments, attributed to the ticket (and host when known).

    Kept apart from `events` so validation never scores a ticket with its own logs. Scans are cached per file
    (mtime, size, signature version) in attachment_scans, which survives ticket reloads.
    """
    n = 0
    if not ATTACHMENTS_DIR.is_dir():
        return n
    version = index.parser_version("jira_attachment")
    cache = {p: (m, s, v, ev) for p, m, s, v, ev in conn.execute("SELECT * FROM attachment_scans")}
    for ticket_dir in sorted(p for p in ATTACHMENTS_DIR.iterdir() if p.is_dir()):
        key = ticket_dir.name
        named = subject_hosts(conn, key)
        default_host = named[0] if len(named) == 1 else None
        for path in sorted(p for p in ticket_dir.rglob("*") if p.is_file()):
            st = path.stat()
            hit = cache.get(str(path))
            if hit and hit[:3] == (st.st_mtime, st.st_size, version):
                hits = json.loads(hit[3])
            else:
                hits = _scan_attachment(path)
                conn.execute(
                    "INSERT OR REPLACE INTO attachment_scans VALUES (?,?,?,?,?)",
                    (str(path), st.st_mtime, st.st_size, version, json.dumps(hits)),
                )
            conn.executemany(
                "INSERT INTO ticket_evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(key, h[0] or default_host, *h[1:]) for h in hits],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO ticket_cats VALUES (?,?,?)",
                [(key, h[1], "attachment") for h in hits if h[3] == "fail"],
            )
            n += len(hits)
    return n


def build(conn: sqlite3.Connection) -> int:
    for table in ("tickets", "ticket_hosts", "ticket_cats", "ticket_refs", "ticket_evidence"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(SCHEMA)
    known = {h for (h,) in conn.execute("SELECT DISTINCT host FROM run_hosts")}
    jql = load_jql_categories()
    tickets = load_cached_tickets()
    for t in tickets:
        cats = text_categories(t["summary"])
        jql_cats = [c for c, keys in jql.items() if t["key"] in keys]
        primary = cats[0] if cats else next((c for c in ("gddr", "tensix", "pcie") if c in jql_cats), "unknown")
        is_open = int(t["status"].strip().lower() not in CLOSED_STATUSES)
        conn.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                t["key"],
                t["project"],
                t["created"],
                t["resolved"],
                t["status"],
                is_open,
                t["summary"],
                t["text"],
                primary,
                t["source"],
            ),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO ticket_cats VALUES (?,?,?)",
            [(t["key"], c, "text") for c in cats] + [(t["key"], c, "jql") for c in jql_cats],
        )
        found = ticket_hosts_by_part(t, known)
        about = subjects(found)
        conn.executemany(
            "INSERT OR IGNORE INTO ticket_hosts VALUES (?,?,?,?)",
            [(t["key"], h, how, int(h in about)) for h, how in found.items()],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO ticket_refs VALUES (?,?)", [(t["key"], r) for r in hosts.ticket_chip_refs(t["text"])]
        )
    scan_attachments(conn)
    conn.commit()
    return len(tickets)
