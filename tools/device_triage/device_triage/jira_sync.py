# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only Jira REST sync into cache/jira: issues (+ all comments), attachments, JQL category membership.

Only GET requests. The token comes from JIRA_ENV (0600) and is never printed or logged.
Incremental: each project re-fetches issues updated since its last sync (minus a day for timezone slack).
"""

import base64
import json
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DCAMP_HC_ROOT, JIRA_CACHE, JIRA_ENV, JIRA_SITE, JQL_MEMBERSHIP, JQL_QUERIES
from .signatures import RESET_CHECK_RE
from .tickets import HC_BOT_PREFIX

ISSUES_DIR = JIRA_CACHE / "issues"
ATTACHMENTS_DIR = JIRA_CACHE / "attachments"
STATE_PATH = JIRA_CACHE / "sync_state.json"
FIELDS = (
    "summary,status,created,updated,resolutiondate,description,comment,attachment,labels,components,"
    "issuetype,priority,reporter,assignee,issuelinks,parent"
)
# What tickets.scan_attachments can read; everything else (images, videos, binaries) is listed, not downloaded.
DOWNLOAD_SUFFIXES = {".log", ".txt", ".out", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".zip", ""}
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
HC_SKIP_PREFIXES = ("telemetry_", "qsfp_dump_", "console-", "triage_device_side")
# Attachments without a job suffix belong to the run whose files were attached closest in time.
HC_PAIR_WINDOW_S = 1800.0


class JiraError(Exception):
    pass


def load_credentials(path: Path = JIRA_ENV) -> tuple[str, str]:
    if not path.exists():
        raise SystemExit(f"no Jira token: create {path} (see README)")
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
        self.requests = 0

    def _open(self, path: str, params: dict | None = None, accept: str = "application/json"):
        url = f"{self.site}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        req = urllib.request.Request(url, headers={"Authorization": self._auth, "Accept": accept})
        for attempt in range(6):
            self.requests += 1
            try:
                return urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES or attempt == 5:
                    raise JiraError(f"GET {path} failed: HTTP {exc.code}") from None
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt)
            except urllib.error.URLError:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def get(self, path: str, params: dict | None = None) -> dict:
        with self._open(path, params) as resp:
            return json.load(resp)

    def download(self, attachment_id: str, dest: Path) -> int:
        part = dest.with_name(dest.name + ".part")
        size = 0
        with self._open(
            f"/rest/api/3/attachment/content/{attachment_id}", {"redirect": "false"}, accept="*/*"
        ) as resp, part.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                size += len(chunk)
        part.replace(dest)
        return size

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
    """Atlassian Document Format -> plain text (enough for grepping and host/regex matching)."""
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


def _person(p: dict | None) -> str:
    return (p or {}).get("displayName", "")


def normalize(issue: dict, comments: list[dict]) -> dict:
    f = issue["fields"]
    return {
        "key": issue["key"],
        "project": issue["key"].split("-")[0],
        "summary": f.get("summary") or "",
        "status": (f.get("status") or {}).get("name", ""),
        "created": f.get("created"),
        "updated": f.get("updated"),
        "resolved": f.get("resolutiondate"),
        "issuetype": (f.get("issuetype") or {}).get("name", ""),
        "priority": (f.get("priority") or {}).get("name", ""),
        "labels": f.get("labels") or [],
        "components": [c.get("name", "") for c in f.get("components") or []],
        "reporter": _person(f.get("reporter")),
        "assignee": _person(f.get("assignee")),
        "parent": (f.get("parent") or {}).get("key"),
        "links": [
            {
                "type": (link.get("type") or {}).get("name", ""),
                "key": (link.get("outwardIssue") or link.get("inwardIssue") or {}).get("key"),
            }
            for link in f.get("issuelinks") or []
        ],
        "description": adf_text(f.get("description")).strip(),
        "comments": [
            {"author": _person(c.get("author")), "created": c.get("created"), "body": adf_text(c.get("body")).strip()}
            for c in comments
        ],
        "attachments": [
            {
                "id": a["id"],
                "filename": a.get("filename", ""),
                "size": a.get("size", 0),
                "mimeType": a.get("mimeType", ""),
                "created": a.get("created"),
                "author": _person(a.get("author")),
            }
            for a in f.get("attachment") or []
        ],
    }


def _all_comments(client: Client, issue: dict) -> list[dict]:
    page = issue["fields"].get("comment") or {}
    comments = list(page.get("comments", []))
    while len(comments) < page.get("total", 0):
        more = client.get(f"/rest/api/3/issue/{issue['key']}/comment", {"startAt": len(comments), "maxResults": 100})
        if not more.get("comments"):
            break
        comments += more["comments"]
    return comments


@dataclass
class SyncStats:
    issues: dict[str, int] = field(default_factory=dict)
    comments: int = 0
    downloaded: int = 0
    downloaded_bytes: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    jql: dict[str, int] = field(default_factory=dict)
    hc_runs: int = 0
    diag: dict[str, int] = field(default_factory=dict)
    requests: int = 0


def _load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def sync_issues(client: Client, project: str, full: bool, stats: SyncStats) -> list[dict]:
    state = _load_state()
    since = None if full else state.get(project, {}).get("max_updated")
    jql = f"project = {project}"
    if since:
        start = datetime.fromisoformat(since) - timedelta(days=1)
        jql += f' AND updated >= "{start:%Y-%m-%d %H:%M}"'
    ISSUES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    max_updated = since
    for issue in client.search(jql + " ORDER BY key ASC", FIELDS):
        comments = _all_comments(client, issue)
        record = normalize(issue, comments)
        (ISSUES_DIR / f"{record['key']}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        stats.comments += len(comments)
        out.append(record)
        updated = datetime.strptime(record["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(timezone.utc).isoformat()
        max_updated = max(max_updated or updated, updated)
    stats.issues[project] = len(out)
    state = _load_state()
    state[project] = {"max_updated": max_updated, "synced_at": datetime.now(timezone.utc).isoformat()}
    STATE_PATH.write_text(json.dumps(state, indent=1))
    return out


def _attachment_path(key: str, att: dict) -> Path:
    name = att["filename"].replace("/", "_") or att["id"]
    return ATTACHMENTS_DIR / key / name


def sync_attachments(client: Client, records: list[dict], max_bytes: int, stats: SyncStats) -> None:
    for record in records:
        seen: set[str] = set()
        for att in record["attachments"]:
            dest = _attachment_path(record["key"], att)
            if dest.name in seen:
                dest = dest.with_name(f"{att['id']}_{dest.name}")
            seen.add(dest.name)
            suffix = Path(att["filename"]).suffix.lower()
            if suffix not in DOWNLOAD_SUFFIXES:
                stats.skipped["type"] = stats.skipped.get("type", 0) + 1
                continue
            if att["size"] > max_bytes:
                stats.skipped["size"] = stats.skipped.get("size", 0) + 1
                continue
            if dest.exists() and dest.stat().st_size == att["size"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                stats.downloaded_bytes += client.download(att["id"], dest)
            except JiraError:
                stats.skipped["error"] = stats.skipped.get("error", 0) + 1
                continue
            stats.downloaded += 1


def write_manifest() -> int:
    """Every attachment of every synced issue, and whether a copy is on disk."""
    rows = ["key\tid\tfilename\tsize\tmime\ton_disk"]
    for path in sorted(ISSUES_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        for att in record.get("attachments", []):
            dest = _attachment_path(record["key"], att)
            alt = dest.with_name(f"{att['id']}_{dest.name}")
            on_disk = dest.exists() or alt.exists()
            rows.append(
                f"{record['key']}\t{att['id']}\t{att['filename']}\t{att['size']}\t{att['mimeType']}\t"
                f"{'yes' if on_disk else 'no'}"
            )
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (ATTACHMENTS_DIR / "manifest.tsv").write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def _jira_ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()


def _job_of(stem: str, prefix: str) -> str | None:
    """`gddr_fast-24936` -> 24936; `<host>-131689-131689` -> 131689; `diag_report-29848680-tkt4q` -> 29848680-tkt4q."""
    job = stem[len(prefix) :].lstrip("-")
    head, sep, rest = job.partition("-")
    return head if sep and rest == head else job or None


def hc_attachment_role(filename: str, host: str) -> tuple[str, str | None]:
    """(role, job) of a health-check bot attachment: suite, diag, triage, check:<name>, or skip."""
    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    if suffix not in (".json", ".log", ".txt") or stem.startswith(HC_SKIP_PREFIXES):
        return "skip", None
    if stem.startswith(host):
        return ("suite", _job_of(stem, host)) if suffix == ".log" else ("skip", None)
    name = stem.split("-", 1)[0]
    if name == "diag_report":
        return ("diag", _job_of(stem, name)) if suffix == ".json" else ("skip", None)
    if name == "triage_host_side":
        return ("triage", _job_of(stem, name)) if suffix == ".txt" else ("skip", None)
    return (f"check:{name}", _job_of(stem, name)) if suffix == ".log" else ("skip", None)


def hc_runs(record: dict, host_named: str) -> list[dict]:
    """A bot ticket's attachments grouped into health-check runs, one per job."""
    groups: dict[str, dict] = {}
    loose = []
    for att in record["attachments"]:
        role, job = hc_attachment_role(att["filename"], host_named)
        if role == "skip":
            continue
        if job is None:
            loose.append((role, att))
        else:
            groups.setdefault(job, {}).setdefault(role, att)
    anchors = {
        job: _jira_ts((files.get("suite") or files.get("diag") or next(iter(files.values())))["created"])
        for job, files in groups.items()
    }
    for role, att in loose:
        ts = _jira_ts(att["created"])
        best = min(anchors, key=lambda j: abs(anchors[j] - ts), default=None)
        if best is not None and abs(anchors[best] - ts) <= HC_PAIR_WINDOW_S:
            groups[best].setdefault(role, att)
    return [{"job": job, "ts": anchors[job], "files": files} for job, files in groups.items()]


def _resolve_host(named: str, known: set[str]) -> str | None:
    """`bh-glx-b05u02` (early naming, no superpod) -> the one known `bh-glx-<pod>-b05u02`, else None."""
    m = re.fullmatch(r"bh-glx-([a-d]\d{2}u\d{2})", named)
    if not m:
        return named
    candidates = [h for h in known if re.fullmatch(rf"bh-glx-\d{{3}}-{m.group(1)}", h)]
    return candidates[0] if len(candidates) == 1 else None


def unwrap_base64(path: Path) -> None:
    """Some bot uploads (July-August) are the file base64-encoded inside quotes; store the decoded bytes."""
    data = path.read_bytes()
    body = data.strip().strip(b'"')
    if not data.startswith(b'"') or not re.fullmatch(rb"[A-Za-z0-9+/=\s]+", body[:4096]):
        return
    body = re.sub(rb"\s+", b"", body)
    path.write_bytes(base64.b64decode(body[: len(body) // 4 * 4]))


def load_truncated_json(text: str) -> tuple[dict | None, str]:
    """(document, "complete" | "repaired" | "unreadable"). A cut-off document keeps every complete value."""
    try:
        return json.loads(text), "complete"
    except ValueError:
        pass
    stack: list[str] = []
    in_str = escaped = False
    cut: tuple[int, tuple[str, ...]] | None = None
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack:
            stack.pop()
            if stack:
                cut = (i + 1, tuple(stack))
        elif ch == "," and stack:
            cut = (i, tuple(stack))
    if cut is None:
        return None, "unreadable"
    end, still_open = cut
    closing = "".join("]" if b == "[" else "}" for b in reversed(still_open))
    try:
        doc = json.loads(text[:end] + closing)
    except ValueError:
        return None, "unreadable"
    return (doc, "repaired") if isinstance(doc, dict) else (None, "unreadable")


def _needed_check_logs(report: dict) -> set[str]:
    """Checks whose logs machine_hc.parse_run reads: failed/warned tests and reset checks."""
    phases = report.get("phases") or {}
    if isinstance(phases, list):
        phases = {ph.get("name", str(i)): ph for i, ph in enumerate(phases)}
    return {
        c.get("name", "")
        for pn, ph in phases.items()
        if isinstance(ph, dict)
        for c in ph.get("checks") or []
        if isinstance(c, dict)
        and c.get("status") in ("FAIL", "WARN")
        and (pn == "tests" or RESET_CHECK_RE.match(c.get("name", "")))
    }


def _top_up_suite_log(client: Client, run_dir: Path, files: dict, stats: SyncStats) -> None:
    """A truncated diag report lost its tests phase: the console log has the test failures."""
    meta_path = run_dir / "ticket.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("diag") != "repaired" or "suite" not in files or (run_dir / "suite.log").exists():
        return
    try:
        client.download(files["suite"]["id"], run_dir / "suite.log")
    except JiraError:
        stats.skipped["hc_run_error"] = stats.skipped.get("hc_run_error", 0) + 1
        return
    unwrap_base64(run_dir / "suite.log")
    meta_path.write_text(json.dumps(meta | {"suite_log": True}, indent=1))
    stats.hc_runs += 1


def sync_hc_runs(client: Client, known_hosts: set[str], on_disk_jobs: set[str], stats: SyncStats) -> None:
    """Rebuild machine HC run directories under DCAMP_HC_ROOT from the bot tickets' attachments.

    Skips jobs whose run directory is still on disk. `ticket.json` is written last and marks a run complete.
    """
    for path in sorted(ISSUES_DIR.glob("DCAMP-*.json")):
        record = json.loads(path.read_text())
        if not record["summary"].startswith(HC_BOT_PREFIX):
            continue
        named = record["summary"].rsplit(" ", 1)[-1].strip().lower()
        host = _resolve_host(named, known_hosts)
        if not host:
            stats.skipped["hc_run_host"] = stats.skipped.get("hc_run_host", 0) + 1
            continue
        for run in hc_runs(record, named):
            if run["job"].split("_")[0] in on_disk_jobs:
                stats.skipped["hc_run_on_disk"] = stats.skipped.get("hc_run_on_disk", 0) + 1
                continue
            day = datetime.fromtimestamp(run["ts"], timezone.utc).strftime("%Y-%m-%d")
            run_dir = DCAMP_HC_ROOT / host / day / run["job"]
            files = run["files"]
            if (run_dir / "ticket.json").exists():
                _top_up_suite_log(client, run_dir, files, stats)
                continue
            if "diag" not in files and "suite" not in files:
                continue
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            diag_state = "absent"
            try:
                if "diag" in files:
                    diag_path = run_dir / "diag_report.json"
                    client.download(files["diag"]["id"], diag_path)
                    unwrap_base64(diag_path)
                    report, diag_state = load_truncated_json(diag_path.read_text(errors="replace"))
                    if report is None:
                        diag_path.unlink()
                    elif diag_state == "repaired":
                        diag_path.write_text(json.dumps(report, indent=1))
                if diag_state in ("complete", "repaired"):
                    wanted = _needed_check_logs(report)
                    downloads = (
                        [(files["triage"], run_dir / "logs" / "triage_host_side.txt")] if "triage" in files else []
                    )
                    downloads += [
                        (att, run_dir / "logs" / f"{role[6:]}.log")
                        for role, att in files.items()
                        if role.startswith("check:") and role[6:] in wanted
                    ]
                    if diag_state == "repaired" and "suite" in files:
                        downloads.append((files["suite"], run_dir / "suite.log"))
                elif "suite" in files:
                    downloads = [(files["suite"], run_dir / "suite.log")]
                else:
                    stats.skipped["hc_run_unreadable"] = stats.skipped.get("hc_run_unreadable", 0) + 1
                    continue
                for att, dest in downloads:
                    client.download(att["id"], dest)
                    unwrap_base64(dest)
            except JiraError:
                stats.skipped["hc_run_error"] = stats.skipped.get("hc_run_error", 0) + 1
                continue
            stats.diag[diag_state] = stats.diag.get(diag_state, 0) + 1
            meta = {
                "key": record["key"],
                "summary": record["summary"],
                "status": record["status"],
                "created": record["created"],
                "host": host,
                "job": run["job"],
                "ts": run["ts"],
                "diag": diag_state,
                "files": {role: att["filename"] for role, att in files.items()},
            }
            (run_dir / "ticket.json").write_text(json.dumps(meta, indent=1))
            stats.hc_runs += 1


def refresh_jql_categories(client: Client) -> dict[str, int]:
    """Run each category's JQL (JQL_QUERIES) and store the matching ticket keys (JQL_MEMBERSHIP)."""
    queries = json.loads(JQL_QUERIES.read_text())["queries"]
    categories = {
        cat: {"jql": jql, "keys": sorted(i["key"] for i in client.search(jql, "summary"))}
        for cat, jql in queries.items()
    }
    JQL_MEMBERSHIP.parent.mkdir(parents=True, exist_ok=True)
    JQL_MEMBERSHIP.write_text(
        json.dumps({"fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "categories": categories}, indent=1)
    )
    return {cat: len(entry["keys"]) for cat, entry in categories.items()}


def sync(
    projects: list[str],
    attachment_projects: list[str],
    full: bool = False,
    max_mb: float = 25.0,
    known_hosts: set[str] | None = None,
    on_disk_jobs: set[str] | None = None,
) -> SyncStats:
    """known_hosts/on_disk_jobs given -> also rebuild machine HC runs from the DCAMP bot tickets."""
    client = Client(*load_credentials())
    stats = SyncStats()
    for project in projects:
        records = sync_issues(client, project, full, stats)
        if project in attachment_projects:
            sync_attachments(client, records, int(max_mb * 1e6), stats)
    if "DCAMP" in projects and known_hosts is not None and on_disk_jobs is not None:
        sync_hc_runs(client, known_hosts, on_disk_jobs, stats)
    stats.jql = refresh_jql_categories(client)
    write_manifest()
    stats.requests = client.requests
    return stats
