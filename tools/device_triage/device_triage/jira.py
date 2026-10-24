# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Live, read-only Jira lookups (GET only). The token comes from JIRA_ENV (0600) and is never printed or logged."""

import base64
import itertools
import json
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from . import tickets
from .config import JIRA_ENV, JIRA_SITE, JQL_QUERIES

FIELDS = "summary,status,created,description,comment"
LIVE_TICKET_LIMIT = 100
RETRY_STATUSES = {429, 500, 502, 503, 504}
BLOCK_NODES = {
    "paragraph",
    "heading",
    "codeBlock",
    "blockquote",
    "listItem",
    "tableRow",
    "panel",
    "rule",
    "mediaSingle",
    "decisionItem",
    "taskItem",
    "expand",
    "nestedExpand",
}


class JiraError(Exception):
    pass


def load_credentials(path: Path = JIRA_ENV) -> tuple[str, str]:
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise SystemExit(f"{path} must be readable only by you: chmod 600 {path}")
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    if not env.get("JIRA_EMAIL") or not env.get("JIRA_TOKEN"):
        raise SystemExit(f"{path} needs JIRA_EMAIL and JIRA_TOKEN")
    return env["JIRA_EMAIL"], env["JIRA_TOKEN"]


class Client:
    def __init__(self, email: str, token: str, site: str = JIRA_SITE):
        self.site = site.rstrip("/")
        self._auth = "Basic " + base64.b64encode(f"{email}:{token}".encode()).decode()

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.site}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        req = urllib.request.Request(url, headers={"Authorization": self._auth, "Accept": "application/json"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES or attempt == 5:
                    raise JiraError(f"GET {path} failed: HTTP {exc.code}") from None
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt)
            except urllib.error.URLError as exc:
                if attempt == 5:
                    raise JiraError(f"GET {path} failed: {exc.reason}") from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def search(self, jql: str, fields: str, page_size: int = 100):
        token = None
        while True:
            params = {"jql": jql, "fields": fields, "maxResults": page_size}
            if token:
                params["nextPageToken"] = token
            page = self.get("/rest/api/3/search/jql", params)
            yield from page.get("issues", [])
            token = page.get("nextPageToken")
            if page.get("isLast", True) or not token:
                return


def adf_text(node) -> str:
    """Atlassian Document Format -> plain text (enough for host and regex matching)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_text(n) for n in node)
    kind = node.get("type")
    attrs = node.get("attrs") or {}
    if kind == "text":
        return node.get("text", "")
    if kind == "hardBreak":
        return "\n"
    if kind in ("mention", "status", "emoji"):
        return attrs.get("text") or attrs.get("shortName") or ""
    if kind in ("inlineCard", "blockCard", "embedCard"):
        return attrs.get("url", "")
    if kind == "media":
        return f"[attachment {attrs.get('alt') or attrs.get('id', '')}]"
    if kind == "tableCell" or kind == "tableHeader":
        return adf_text(node.get("content")).strip() + " | "
    body = adf_text(node.get("content"))
    return body + "\n" if kind in BLOCK_NODES else body


def normalize(issue: dict) -> dict:
    f = issue["fields"]
    return {
        "key": issue["key"],
        "summary": f.get("summary") or "",
        "status": (f.get("status") or {}).get("name", ""),
        "created": f.get("created"),
        "description": adf_text(f.get("description")).strip(),
        "comments": [{"body": adf_text(c.get("body")).strip()} for c in (f.get("comment") or {}).get("comments", [])],
    }


def _jql_categories(client: Client, keys: list[str]) -> dict[str, set[str]]:
    """Which of the shipped category queries (JQL_QUERIES) each of `keys` matches."""
    out: dict[str, set[str]] = defaultdict(set)
    if not keys:
        return out
    for cat, jql in json.loads(JQL_QUERIES.read_text())["queries"].items():
        for issue in client.search(f"({jql}) AND key in ({','.join(keys)})", "summary"):
            out[issue["key"]].add(cat)
    return out


def host_tickets(host: str) -> list[dict]:
    """DIIM and DCAMP tickets naming `host`, live (`tickets.host_view`); none without a token."""
    if not JIRA_ENV.exists():
        return []
    client = Client(*load_credentials())
    short = host.rsplit("-", 1)[-1]
    # Jira doesn't split `bh-glx-120-b03u02` into words: search the full name and the short one (`b03u02`).
    jql = f'(project = DIIM OR project = DCAMP) AND (text ~ "{host}" OR text ~ "{short}") ORDER BY created DESC'
    try:
        issues = list(itertools.islice(client.search(jql, FIELDS), LIVE_TICKET_LIMIT))
        found = [tickets.ticket_from_issue(normalize(i)) for i in issues]
        categories = _jql_categories(client, [t["key"] for t in found])
    except JiraError as exc:
        print(f"Jira unavailable, tickets not shown: {exc}", file=sys.stderr)
        return []
    return tickets.host_view(found, host, categories)
