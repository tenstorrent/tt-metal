#!/usr/bin/env python3
"""Keep a machine-readable record of who `/codeowners ping` notified.

The record lives in a hidden HTML comment inside the CodeOwners bot comment
(`<!-- CodeOwners-Workflow-Bot -->`) so a follow-up reminder can read it from the
PR without access to Slack:

    <!-- codeowners-ping-record
    {"pings": [
      {"at": "2026-10-19T17:02:11Z",
       "rules": [{"rule": "@tenstorrent/example-team",
                  "any_of": ["alice", "carol", "bob", "dave"],
                  "pinged": ["alice", "carol"]}]}
    ]}
    -->

Each ping appends one entry to `pings`; nothing is ever rewritten.

Subcommands:
  append  Append the entry written by select_owners_to_ping.py ($PING_RECORD_FILE)
          to the newest bot comment on $PR_NUMBER. Needs GITHUB_TOKEN and
          GITHUB_REPOSITORY.
  carry NEW_BODY_FILE OLD_BODY_FILE
          The summary step rewrites the bot comment on every run. Copy the record
          block from the old body into the new one (in place) so earlier pings are
          not lost.

Failures here only ever lose the record, never the ping: every error path logs a
warning and exits 0.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BOT_MARKER = "CodeOwners-Workflow-Bot"
BLOCK_OPEN = "<!-- codeowners-ping-record"
BLOCK_CLOSE = "-->"
BLOCK_RE = re.compile(re.escape(BLOCK_OPEN) + r"\n(.*?)\n" + re.escape(BLOCK_CLOSE), re.DOTALL)

# Old entries are dropped first; the reminder only ever needs the recent pings.
MAX_PINGS = 50
# GitHub caps a comment body at 65536 characters. Leave headroom so the record
# can never be what pushes the summary over the limit.
MAX_BODY_CHARS = 60000


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_pings(body: str) -> list[dict]:
    """The `pings` list from a comment body, or [] if absent or unreadable."""
    match = BLOCK_RE.search(body or "")
    if not match:
        return []
    try:
        pings = json.loads(match.group(1)).get("pings")
    except (ValueError, AttributeError):
        log("::warning::Existing codeowners ping record is not valid JSON; starting a new one.")
        return []
    return pings if isinstance(pings, list) else []


def render_block(pings: list[dict]) -> str:
    # One entry per line keeps the raw comment readable. `>` is escaped so no
    # value can close the HTML comment early; `>` is still valid JSON.
    lines = ",\n".join(json.dumps(p, separators=(", ", ": ")) for p in pings)
    payload = '{"pings": [\n' + lines + "\n]}"
    return f"{BLOCK_OPEN}\n{payload.replace('>', chr(92) + 'u003e')}\n{BLOCK_CLOSE}"


def with_pings(body: str, pings: list[dict], max_chars: int = MAX_BODY_CHARS) -> str:
    """`body` with its record block replaced by `pings` (appended if it had none).

    Oldest entries are dropped until the whole body fits in `max_chars`; if not
    even the newest one fits, the record is left out and the summary kept.
    """
    base = BLOCK_RE.sub("", body, count=1).rstrip("\n")
    pings = pings[-MAX_PINGS:]
    while pings:
        updated = base + "\n\n" + render_block(pings) + "\n"
        if len(updated) <= max_chars:
            return updated
        pings = pings[1:]
        log("::warning::Codeowners ping record too large for the comment; dropping its oldest entry.")
    log("::warning::Codeowners ping record does not fit in the comment; leaving it out.")
    return base + "\n"


# --------------------------------------------------------------------------- #
# carry
# --------------------------------------------------------------------------- #
def carry(new_path: str, old_path: str) -> int:
    try:
        with open(old_path, encoding="utf-8") as fh:
            pings = extract_pings(fh.read())
        if not pings:
            return 0
        with open(new_path, encoding="utf-8") as fh:
            new_body = fh.read()
        with open(new_path, "w", encoding="utf-8") as fh:
            fh.write(with_pings(new_body, pings))
        log(f"Carried {len(pings)} codeowners ping record entries into the updated comment")
    except OSError as exc:
        log(f"::warning::Could not carry the codeowners ping record forward ({exc})")
    return 0


# --------------------------------------------------------------------------- #
# append
# --------------------------------------------------------------------------- #
def github(method: str, url: str, token: str, payload: dict | None = None):
    request = urllib.request.Request(url, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "tt-metal-codeowners-ping")
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, data=data, timeout=30) as response:
        return json.load(response)


def newest_bot_comment(repo: str, pr_number: str, token: str) -> dict | None:
    """Same pick as the summary step: newest comment containing the marker."""
    comments: list[dict] = []
    for page in range(1, 11):
        batch = github(
            "GET",
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}",
            token,
        )
        comments.extend(batch)
        if len(batch) < 100:
            break
    ours = [c for c in comments if BOT_MARKER in (c.get("body") or "")]
    return max(ours, key=lambda c: c.get("created_at") or "") if ours else None


def append(entry_path: str) -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not (repo and pr_number and token and entry_path):
        log("::warning::Missing GITHUB_REPOSITORY, PR_NUMBER, GITHUB_TOKEN or PING_RECORD_FILE; ping not recorded.")
        return 0
    try:
        with open(entry_path, encoding="utf-8") as fh:
            rules = json.load(fh).get("rules") or []
    except (OSError, ValueError, AttributeError) as exc:
        log(f"::warning::Could not read {entry_path} ({exc}); ping not recorded.")
        return 0
    if not rules:
        log("No pinged rules to record")
        return 0

    entry = {"at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rules": rules}
    try:
        comment = newest_bot_comment(repo, pr_number, token)
        if comment is None:
            log("::warning::No CodeOwners bot comment on the PR; ping not recorded.")
            return 0
        body = comment.get("body") or ""
        updated = with_pings(body, extract_pings(body) + [entry])
        github(
            "PATCH",
            f"https://api.github.com/repos/{repo}/issues/comments/{comment['id']}",
            token,
            {"body": updated},
        )
    except (urllib.error.URLError, ValueError, TimeoutError, KeyError) as exc:
        log(f"::warning::Could not update the CodeOwners bot comment ({exc}); ping not recorded.")
        return 0
    log(f"Recorded ping on comment {comment['id']}: {json.dumps(entry)}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "append":
        return append(os.environ.get("PING_RECORD_FILE", ""))
    if len(argv) == 4 and argv[1] == "carry":
        return carry(argv[2], argv[3])
    log(__doc__ or "")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
