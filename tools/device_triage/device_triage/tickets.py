# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Jira tickets as seen from one host: which hosts a ticket names, how, and which categories it is in."""

import itertools
import re
from datetime import datetime

from . import hosts

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
# How a ticket names a host: in the title unambiguously (NAMED_HOWS); hosts only in the description are
# "description" (they often list a whole quad around the one affected host); only in comments, "comment".
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


def ticket_from_issue(issue: dict) -> dict:
    """An issue (`jira.normalize`) as a ticket: text split into title, description and comments."""
    summary, description = issue.get("summary", ""), issue.get("description", "")
    return {
        "key": issue["key"],
        "created": _parse_jira_ts(issue.get("created")),
        "status": issue.get("status", ""),
        "summary": summary,
        "description": description,
        "comments": "\n".join(c.get("body", "") for c in issue.get("comments", [])),
    }


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


def siblings(host: str) -> list[str]:
    """The same rack/slot in the other superpod: a short name (`b02u08`) in a ticket may mean either."""
    m = re.fullmatch(r"bh-glx-(\d{3})-([a-d]\d{2}u\d{2})", host)
    return [f"bh-glx-{pod}-{m.group(2)}" for pod in ("110", "120") if pod != m.group(1)] if m else []


def host_view(tickets: list[dict], host: str, jql_categories: dict[str, set[str]]) -> list[dict]:
    """The tickets that name `host`, as triage shows them: about the host first, humans' before the HC bot's,
    open first, newest first. `jql_categories` maps a ticket key to the categories whose JQL it matches."""
    known = {host, *siblings(host)}
    out = []
    for t in tickets:
        found = ticket_hosts_by_part(t, known)
        if host not in found:
            continue
        out.append(
            {
                "key": t["key"],
                "created": t["created"],
                "status": t["status"],
                "open": t["status"].strip().lower() not in CLOSED_STATUSES,
                "summary": t["summary"],
                "how": found[host],
                "subject": host in subjects(found),
                "bot": t["summary"].startswith(HC_BOT_PREFIX),
                "categories": sorted(set(text_categories(t["summary"])) | jql_categories.get(t["key"], set())),
            }
        )
    return sorted(out, key=lambda v: (not v["subject"], v["bot"], not v["open"], -(v["created"] or 0)))
