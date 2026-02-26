# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Measure average review response time per CODEOWNERS line.

For each CODEOWNERS line, computes the time between the first "/codeowners ping"
comment on a PR and the first response from an owner listed on that line.

Two separate metrics are reported:
  1. Time to first APPROVAL
  2. Time to first REVIEW ACTIVITY (approval, review comment, or changes requested)

Usage:
    python tools/codeowners_review_metrics.py [--days N] [--repo OWNER/REPO] [--codeowners PATH]

Requirements:
    pip install pathspec
    gh CLI must be authenticated (gh auth login)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import pathspec
except ImportError:
    print("ERROR: 'pathspec' package is required. Install it with: pip install pathspec", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# GitHub API helpers (via gh CLI)
# ---------------------------------------------------------------------------

# Lock to serialize console output from worker threads
_print_lock = threading.Lock()

# Disk cache configuration (set by main() before workers start)
_cache_dir: Path | None = None
_no_cache: bool = False


def safe_print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(path: str, paginate: bool) -> str:
    """Return a filename-safe hex digest for a given API call."""
    raw = json.dumps({"path": path, "paginate": paginate}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(path: str, paginate: bool) -> Any:
    """Return cached data for this call, or _MISS if not cached."""
    if _no_cache or _cache_dir is None:
        return _MISS
    cache_file = _cache_dir / f"{_cache_key(path, paginate)}.json"
    try:
        with cache_file.open() as f:
            return json.load(f)["data"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return _MISS


def _cache_put(path: str, paginate: bool, data: Any) -> None:
    """Write data to the disk cache atomically."""
    if _no_cache or _cache_dir is None:
        return
    cache_file = _cache_dir / f"{_cache_key(path, paginate)}.json"
    payload = json.dumps({"path": path, "paginate": paginate, "data": data})
    # Write to a temp file in the same dir then rename for atomicity
    fd, tmp = tempfile.mkstemp(dir=_cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp, cache_file)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# Sentinel for cache miss (cannot use None since None is a valid cached value)
class _MissType:
    pass
_MISS = _MissType()


_RATE_LIMIT_INITIAL_DELAY = 10  # seconds for first retry
_RATE_LIMIT_MAX_DELAY = 300    # cap at 5 minutes
_RATE_LIMIT_MAX_RETRIES = 30


def gh_api(path: str, method: str = "GET", paginate: bool = False) -> Any:
    """Call the GitHub API via the gh CLI and return parsed JSON.

    Results are transparently read from / written to the disk cache when
    _cache_dir is set and _no_cache is False.  Rate-limit errors trigger a
    sleep-and-retry loop with exponential backoff instead of aborting.
    """
    import time

    cached = _cache_get(path, paginate)
    if not isinstance(cached, _MissType):
        return cached

    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd += ["--method", method, path]

    delay = _RATE_LIMIT_INITIAL_DELAY
    for attempt in range(1, _RATE_LIMIT_MAX_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            break  # success
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            if "rate limit" in stderr.lower():
                safe_print(
                    f"  [rate-limited] attempt {attempt}/{_RATE_LIMIT_MAX_RETRIES}, "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY)
                continue
            raise RuntimeError(f"gh api {path} failed: {stderr}") from e
    else:
        safe_print("ERROR: GitHub API rate limit exceeded after max retries.", file=sys.stderr)
        raise RuntimeError(f"gh api {path} failed: rate limit exceeded after {_RATE_LIMIT_MAX_RETRIES} retries")

    text = result.stdout.strip()
    if not text:
        data: Any = []
    elif paginate:
        merged: list = []
        for chunk in _split_json_arrays(text):
            merged.extend(chunk)
        data = merged
    else:
        data = json.loads(text)

    _cache_put(path, paginate, data)
    return data


def _split_json_arrays(text: str) -> list[list]:
    """Split concatenated JSON arrays produced by gh --paginate."""
    decoder = json.JSONDecoder()
    results = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        obj, end = decoder.raw_decode(text, idx)
        results.append(obj)
        idx = end
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
    return results


def _urlencode(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


# ---------------------------------------------------------------------------
# CODEOWNERS parsing
# ---------------------------------------------------------------------------

BYPASS_TEAM = "codeowner-bypass"


def parse_codeowners(path: str) -> list[tuple[str, list[str]]]:
    """
    Parse a CODEOWNERS file into an ordered list of (pattern, owners) tuples.

    Filters out the `codeowner-bypass` team since it's not a real reviewer.
    Returns lines in file order (last match wins during lookup).
    """
    entries: list[tuple[str, list[str]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Strip inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()
            if not line:
                continue
            parts = line.split()
            pattern = parts[0]
            owners = [o for o in parts[1:] if BYPASS_TEAM not in o]
            if owners:
                entries.append((pattern, owners))
    return entries


def build_specs(codeowners: list[tuple[str, list[str]]]) -> list[tuple[str, list[str], pathspec.PathSpec]]:
    """Pre-build pathspec objects for all CODEOWNERS entries."""
    return [
        (pattern, owners, pathspec.PathSpec.from_lines("gitwildmatch", [pattern]))
        for pattern, owners in codeowners
    ]


def match_files_to_codeowners(
    filepaths: list[str],
    specs: list[tuple[str, list[str], pathspec.PathSpec]],
) -> dict[str, tuple[str, list[str]]]:
    """
    Map each filepath to its winning CODEOWNERS entry (last-match-wins).
    Uses pre-built pathspec objects for efficiency.
    """
    result: dict[str, tuple[str, list[str]]] = {}
    for filepath in filepaths:
        matched = None
        for pattern, owners, spec in specs:
            if spec.match_file(filepath):
                matched = (pattern, owners)
        if matched is not None:
            result[filepath] = matched
    return result


# ---------------------------------------------------------------------------
# Team membership resolution (cached, thread-safe)
# ---------------------------------------------------------------------------

_team_cache: dict[str, set[str]] = {}
_team_cache_lock = threading.Lock()


def resolve_team_members(team_slug: str, org: str = "tenstorrent") -> set[str]:
    """Return the set of GitHub logins that belong to the given team (cached)."""
    with _team_cache_lock:
        if team_slug in _team_cache:
            return _team_cache[team_slug]

    try:
        members = gh_api(f"/orgs/{org}/teams/{team_slug}/members", paginate=True)
        logins = {m["login"].lower() for m in members if isinstance(m, dict) and "login" in m}
    except RuntimeError:
        logins = set()

    with _team_cache_lock:
        _team_cache[team_slug] = logins
    return logins


def is_owner(username: str, owners: list[str], org: str = "tenstorrent") -> bool:
    """
    Return True if username is a direct owner or a member of an owner team.
    owners entries look like: @login  or  @org/team-slug
    """
    username_lower = username.lower()
    for owner in owners:
        owner = owner.lstrip("@")
        if "/" in owner:
            team_slug = owner.split("/", 1)[1]
            if username_lower in resolve_team_members(team_slug, org):
                return True
        else:
            if owner.lower() == username_lower:
                return True
    return False


# ---------------------------------------------------------------------------
# PR discovery and data fetching
# ---------------------------------------------------------------------------

PING_RE = re.compile(r"^\s*/codeowners?\s+ping", re.IGNORECASE | re.MULTILINE)


def parse_gh_timestamp(ts: str) -> datetime:
    """Parse ISO 8601 timestamp from GitHub API into an aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def business_seconds(start: datetime, end: datetime) -> float:
    """Return elapsed seconds between *start* and *end*, excluding full
    Saturday and Sunday calendar days (UTC).  Partial weekend‐boundary days
    are handled correctly."""
    if end <= start:
        return 0.0
    total = 0.0
    cur = start
    while cur < end:
        if cur.weekday() < 5:
            next_midnight = cur.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            total += (min(next_midnight, end) - cur).total_seconds()
        cur = cur.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return total


def get_prs_with_ping(repo: str, days: int = 30) -> list[dict]:
    """
    Return PRs (open or merged) updated within the last `days` days that have
    at least one '/codeowners ping' comment.

    Uses GitHub search API (paginates manually since the response is a dict,
    not a bare array). GitHub search API caps at 1000 total results.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f'repo:{repo} is:pr updated:>={since} "/codeowners ping" in:comments'
    print(f"Searching for PRs with '/codeowners ping' comments since {since}...")

    all_items: list[dict] = []
    page = 1
    per_page = 100
    while True:
        data = gh_api(f"/search/issues?q={_urlencode(query)}&per_page={per_page}&page={page}")
        if not isinstance(data, dict):
            break
        items = data.get("items", [])
        all_items.extend(items)
        total_count = data.get("total_count", 0)
        print(f"  Page {page}: got {len(items)} PR(s) (total reported by API: {total_count})")
        if len(items) < per_page or len(all_items) >= total_count:
            break
        page += 1

    print(f"Found {len(all_items)} candidate PR(s) total.")
    return all_items


def get_pr_comments(repo: str, pr_number: int) -> list[dict]:
    """Return all issue comments on a PR."""
    return gh_api(f"/repos/{repo}/issues/{pr_number}/comments?per_page=100", paginate=True)



def get_pr_reviews(repo: str, pr_number: int) -> list[dict]:
    """
    Return all submitted reviews on a PR.

    Each review has a state of APPROVED, CHANGES_REQUESTED, COMMENTED, or DISMISSED.
    Inline review comments are always attached to a submitted review, so a review
    with state COMMENTED already captures the earliest timestamp for any inline comment;
    there is no need to separately fetch the individual review comments endpoint.
    """
    return gh_api(f"/repos/{repo}/pulls/{pr_number}/reviews?per_page=100", paginate=True)


def get_pr_changed_files(repo: str, pr_number: int) -> list[str]:
    """Return the list of files changed in a PR."""
    files = gh_api(f"/repos/{repo}/pulls/{pr_number}/files?per_page=100", paginate=True)
    return [f["filename"] for f in files if isinstance(f, dict) and "filename" in f]


# ---------------------------------------------------------------------------
# Per-PR processing (runs in a thread pool worker)
# ---------------------------------------------------------------------------

def process_pr(
    pr: dict,
    repo: str,
    specs: list[tuple[str, list[str], pathspec.PathSpec]],
    total: int,
    index: int,
    min_age_days: int = 7,
    ignore_weekends: bool = False,
) -> dict[str, dict[str, float | None]] | None:
    """
    Fetch all data for a single PR and return per-pattern timing results.

    Returns a dict mapping pattern -> {"approval": seconds|None, "activity": seconds|None},
    or None if the PR should be skipped.
    """
    pr_number = pr["number"]
    pr_state = pr.get("state", "unknown")
    pr_title = pr.get("title", "")[:60]
    pr_author = pr.get("user", {}).get("login", "").lower()
    safe_print(f"  [{index}/{total}] PR #{pr_number} ({pr_state}): {pr_title}")

    # Fetch comments once — used for both ping detection and activity scanning.
    comments = get_pr_comments(repo, pr_number)

    # Find the first /codeowners ping comment.
    ping_time: datetime | None = None
    for comment in sorted(comments, key=lambda c: c["created_at"]):
        if PING_RE.search(comment.get("body", "")):
            ping_time = parse_gh_timestamp(comment["created_at"])
            break
    if ping_time is None:
        safe_print(f"    -> No /codeowners ping comment found, skipping.")
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)
    if ping_time > cutoff:
        safe_print(f"    -> Ping is less than {min_age_days} days old, skipping.")
        return None

    changed_files = get_pr_changed_files(repo, pr_number)
    if not changed_files:
        safe_print(f"    -> No changed files found, skipping.")
        return None

    file_to_entry = match_files_to_codeowners(changed_files, specs)
    if not file_to_entry:
        safe_print(f"    -> No CODEOWNERS matches for changed files, skipping.")
        return None

    # Unique CODEOWNERS patterns touched by this PR
    touched_entries: dict[str, list[str]] = {}
    for _filepath, (pattern, owners) in file_to_entry.items():
        touched_entries[pattern] = owners

    # Build sorted event timeline from both submitted reviews and issue comments.
    #
    # Rules applied to every event:
    #   - Skip events by the PR author (they can't review their own PR).
    #   - Pre-ping events are clamped to ping_time (recorded as 0s delta) because
    #     an owner who already responded before the ping was asked had already done
    #     their part.
    events: list[tuple[datetime, str, str]] = []

    # Submitted reviews (APPROVED / CHANGES_REQUESTED / COMMENTED / DISMISSED).
    # A COMMENTED review is created whenever a reviewer submits inline comments,
    # so this captures all review-style activity without a separate comments fetch.
    reviews = get_pr_reviews(repo, pr_number)
    for review in reviews:
        submitted = review.get("submitted_at")
        if not submitted:
            continue
        author = review["user"]["login"]
        if author.lower() == pr_author:
            continue
        ts = parse_gh_timestamp(submitted)
        state = review["state"]
        events.append((max(ts, ping_time), author, state))

    # Regular issue comments — any comment from an owner counts as activity.
    for comment in comments:
        author = comment.get("user", {}).get("login", "")
        if author.lower() == pr_author:
            continue
        ts = parse_gh_timestamp(comment["created_at"])
        events.append((max(ts, ping_time), author, "ISSUE_COMMENT"))

    events.sort(key=lambda e: e[0])

    result: dict[str, dict[str, float | None]] = {}
    for pattern, owners in touched_entries.items():
        first_approval_dt: datetime | None = None
        first_activity_dt: datetime | None = None

        for ts, author, event_type in events:
            if not is_owner(author, owners):
                continue
            if first_activity_dt is None:
                first_activity_dt = ts
            if event_type == "APPROVED" and first_approval_dt is None:
                first_approval_dt = ts

        if ignore_weekends:
            approval_delta = business_seconds(ping_time, first_approval_dt) if first_approval_dt else None
            activity_delta = business_seconds(ping_time, first_activity_dt) if first_activity_dt else None
        else:
            approval_delta = (first_approval_dt - ping_time).total_seconds() if first_approval_dt else None
            activity_delta = (first_activity_dt - ping_time).total_seconds() if first_activity_dt else None

        result[pattern] = {
            "state": pr_state,
            "approval": approval_delta,
            "activity": activity_delta,
        }

    return result


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xd Yh Zm' or 'Yh Zm' or 'Zm'."""
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def analyze(
    repo: str,
    codeowners_path: str,
    days: int = 30,
    workers: int = 8,
    min_age_days: int = 7,
    ignore_weekends: bool = False,
) -> None:
    codeowners = parse_codeowners(codeowners_path)
    print(f"Parsed {len(codeowners)} CODEOWNERS entries (excluding bypass).")
    print(f"Ignoring PRs whose first ping is less than {min_age_days} day(s) old.")
    if ignore_weekends:
        print("Weekend time (Sat/Sun) is excluded from latency calculations.")

    specs = build_specs(codeowners)

    prs = get_prs_with_ping(repo, days)
    if not prs:
        print("No PRs found. Nothing to analyze.")
        return

    total = len(prs)
    print(f"Processing {total} PRs with {workers} parallel workers...")

    approval_deltas: dict[str, list[float | None]] = defaultdict(list)
    activity_deltas: dict[str, list[float | None]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_pr, pr, repo, specs, total, i, min_age_days, ignore_weekends): pr
            for i, pr in enumerate(prs, 1)
        }
        for future in as_completed(futures):
            try:
                pr_result = future.result()
            except Exception as exc:
                pr = futures[future]
                safe_print(f"    -> PR #{pr['number']} raised an error: {exc}")
                continue
            if pr_result is None:
                continue
            for pattern, timing in pr_result.items():
                approval_deltas[pattern].append(timing["approval"])
                activity_deltas[pattern].append(timing["activity"])

    print()
    _print_table("Time to First APPROVAL (from /codeowners ping)", approval_deltas)
    print()
    _print_table(
        "Time to First REVIEW ACTIVITY (approval / comment / changes-requested)",
        activity_deltas,
    )


def _print_table(title: str, deltas: dict[str, list[float | None]]) -> None:
    if not deltas:
        print(f"=== {title} ===")
        print("  (no data)")
        return

    COL_PATTERN = 50
    COL_AVG = 12
    COL_MEDIAN = 12
    COL_COUNT = 7
    COL_PENDING = 13

    header = (
        f"{'CODEOWNERS Pattern':<{COL_PATTERN}} | "
        f"{'Avg Time':>{COL_AVG}} | "
        f"{'Median':>{COL_MEDIAN}} | "
        f"{'Count':>{COL_COUNT}} | "
        f"{'Still Pending':>{COL_PENDING}}"
    )
    sep = "-" * len(header)

    print(f"=== {title} ===")
    print(header)
    print(sep)

    def sort_key(item):
        _, vals = item
        resolved = [v for v in vals if v is not None]
        return mean(resolved) if resolved else float("inf")

    for pattern, vals in sorted(deltas.items(), key=sort_key):
        resolved = [v for v in vals if v is not None]
        pending = sum(1 for v in vals if v is None)
        count = len(vals)

        if resolved:
            avg_str = fmt_duration(mean(resolved))
            med_str = fmt_duration(median(resolved))
        else:
            avg_str = "N/A"
            med_str = "N/A"

        print(
            f"{pattern:<{COL_PATTERN}} | "
            f"{avg_str:>{COL_AVG}} | "
            f"{med_str:>{COL_MEDIAN}} | "
            f"{count:>{COL_COUNT}} | "
            f"{pending:>{COL_PENDING}}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure average CODEOWNERS review response time from /codeowners ping."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="How many days back to search for PRs (default: 30)",
    )
    parser.add_argument(
        "--repo",
        default="tenstorrent/tt-metal",
        help="GitHub repository in OWNER/REPO format (default: tenstorrent/tt-metal)",
    )
    parser.add_argument(
        "--codeowners",
        default=".github/CODEOWNERS",
        help="Path to CODEOWNERS file (default: .github/CODEOWNERS)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel API workers (default: 8)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "codeowners_metrics"),
        help="Directory for on-disk GitHub API response cache "
             "(default: ~/.cache/codeowners_metrics)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the disk cache; always fetch fresh data from GitHub",
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=7,
        help="Ignore PRs whose first ping is less than N days old to reduce recency bias (default: 7)",
    )
    parser.add_argument(
        "--ignore-weekends",
        action="store_true",
        help="Exclude Saturday/Sunday from latency calculations",
    )
    args = parser.parse_args()

    global _cache_dir, _no_cache
    _no_cache = args.no_cache
    if not _no_cache:
        _cache_dir = Path(args.cache_dir)
        _cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Disk cache: {_cache_dir}  (use --no-cache to bypass)")

    analyze(
        repo=args.repo,
        codeowners_path=args.codeowners,
        days=args.days,
        workers=args.workers,
        min_age_days=args.min_age_days,
        ignore_weekends=args.ignore_weekends,
    )


if __name__ == "__main__":
    main()
