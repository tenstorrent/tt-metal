#!/usr/bin/env python3
"""Targeted review requests and a DiT-approval status for models/tt_dit/ PRs.

Two modes:

  assign  Request reviews from the DiT team members who most recently committed to
          the directories the PR touches, so a PR gets 1-2 relevant reviewers rather
          than the whole team. At least one DiT team member is always requested.
          When every changed file is under models/tt_dit/, the team-wide requests
          that CODEOWNERS added are withdrawn. This does not weaken the code owner
          rule: GitHub still requires an approval from a code owner.

  check   Post a "DiT approval" commit status on the PR head commit: success when
          the PR does not touch models/tt_dit/ or a DiT team member has approved,
          pending otherwise. Branch protection can make this status required.

The script never checks out or runs PR code; it only reads PR metadata via the API.

Environment variables:
  GH_TOKEN       Token with read:org (to list team members) and pull request write
                 access (to change review requests).
  STATUS_TOKEN   Token used to post the commit status (default: GH_TOKEN).
  REPO           owner/name.
  PR_NUMBER      Pull request number.
  DIT_TEAM       Team whose members count as DiT reviewers (default: metalium-dit).
  REMOVE_TEAMS   Comma-separated teams whose review requests to withdraw in assign
                 mode (default: metalium-dit,cse-developer-ttnn).
  NUM_REVIEWERS  Number of reviewers to request (default: 2).
  HISTORY_DAYS   How far back to look at commit history (default: 180).
  DRY_RUN        "1" prints what would change without changing anything.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

API = "https://api.github.com"
DIT_PREFIX = "models/tt_dit/"
STATUS_CONTEXT = "DiT approval"
# Bounds the number of commit-history API calls for very large PRs.
MAX_DIRS = 25


def api(method: str, path: str, token: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        payload = resp.read()
    return json.loads(payload) if payload else None


def api_list(path: str, token: str) -> list:
    sep = "&" if "?" in path else "?"
    items: list = []
    page = 1
    while True:
        batch = api("GET", f"{path}{sep}per_page=100&page={page}", token)
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def team_members(org: str, team: str, token: str) -> set[str]:
    return {m["login"] for m in api_list(f"/orgs/{org}/teams/{team}/members", token)}


def dir_weights(files: list[str]) -> Counter:
    """Number of changed models/tt_dit/ files per parent directory."""
    return Counter(f.rsplit("/", 1)[0] for f in files if f.startswith(DIT_PREFIX))


def history_scores(repo: str, dirs: Counter, since: str, token: str) -> Counter:
    """Score commit authors by recent commits to each directory, weighted by how many
    of the PR's files are in that directory."""
    scores: Counter = Counter()
    for d, weight in dirs.most_common(MAX_DIRS):
        query = urllib.parse.urlencode({"path": d, "since": since, "per_page": 100})
        for commit in api("GET", f"/repos/{repo}/commits?{query}", token):
            author = commit.get("author")
            if author and author.get("type") != "Bot":
                scores[author["login"]] += weight
    return scores


def pick_reviewers(
    scores: Counter, members: set[str], author: str, already: set[str], n: int, seed: int
) -> list[str]:
    """Pick up to n DiT members to request, highest history score first.

    `already` holds logins who are already requested or have reviewed. If none of them
    is a DiT member and nobody has history in the touched directories, falls back to a
    random DiT member so at least one is always on the PR.
    """
    eligible = members - {author}
    present = already & eligible
    slots = n - len(present)
    if slots <= 0:
        return []
    ranked = sorted(
        (m for m in eligible - already if scores.get(m, 0) > 0),
        key=lambda m: (-scores[m], m),
    )
    picks = ranked[:slots]
    if not picks and not present:
        pool = sorted(eligible - already)
        if pool:
            picks = [random.Random(seed).choice(pool)]
    return picks


def dit_approvers(reviews: list[dict], members: set[str]) -> list[str]:
    """DiT members whose latest decisive review is an approval (mirrors how GitHub
    counts approvals: a later comment-only review does not cancel an approval)."""
    latest: dict[str, str] = {}
    for r in reviews:  # the API returns reviews oldest first
        user = r.get("user")
        if user and r["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest[user["login"]] = r["state"]
    return sorted(u for u, s in latest.items() if s == "APPROVED" and u in members)


def assign(repo: str, number: int, pr: dict, files: list[str], token: str, dry_run: bool) -> None:
    if pr["draft"]:
        print("Draft PR; reviewers are assigned when it is marked ready for review.")
        return
    dirs = dir_weights(files)
    if not dirs:
        print("No models/tt_dit/ changes; nothing to do.")
        return

    org = repo.split("/")[0]
    members = team_members(org, os.environ.get("DIT_TEAM", "metalium-dit"), token)
    author = pr["user"]["login"]
    reviewed = {r["user"]["login"] for r in api_list(f"/repos/{repo}/pulls/{number}/reviews", token) if r.get("user")}
    requested_users = {u["login"] for u in pr["requested_reviewers"]}
    requested_teams = {t["slug"] for t in pr["requested_teams"]}

    days = int(os.environ.get("HISTORY_DAYS", "180"))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    scores = history_scores(repo, dirs, since, token)
    picks = pick_reviewers(
        scores,
        members,
        author,
        requested_users | reviewed,
        int(os.environ.get("NUM_REVIEWERS", "2")),
        seed=number,
    )

    print("Top DiT candidates by recent commits:")
    for login, score in [(m, scores[m]) for m in sorted(members, key=lambda m: -scores[m])[:5]]:
        print(f"  {login}: {score}")
    print(f"Requesting: {picks or 'nobody (a DiT member is already on the PR)'}")

    remove = [t for t in os.environ.get("REMOVE_TEAMS", "metalium-dit,cse-developer-ttnn").split(",") if t]
    # Only withdraw team requests when the whole PR is DiT code; other changes may
    # legitimately need those teams.
    to_remove = sorted(requested_teams & set(remove)) if all(f.startswith(DIT_PREFIX) for f in files) else []
    print(f"Withdrawing team requests: {to_remove or 'none'}")

    if dry_run:
        print("DRY_RUN set; no changes made.")
        return
    path = f"/repos/{repo}/pulls/{number}/requested_reviewers"
    if picks:
        try:
            api("POST", path, token, {"reviewers": picks})
        except urllib.error.HTTPError:
            # One bad login (e.g. no repo access) fails the whole batch; retry singly.
            for login in picks:
                try:
                    api("POST", path, token, {"reviewers": [login]})
                except urllib.error.HTTPError as e:
                    print(f"::warning::Could not request review from {login}: HTTP {e.code}")
    # Withdraw team requests only after an individual DiT reviewer is in place.
    if to_remove and (picks or (requested_users | reviewed) & members):
        api("DELETE", path, token, {"reviewers": [], "team_reviewers": to_remove})


def check(repo: str, number: int, pr: dict, files: list[str], token: str, dry_run: bool) -> None:
    if not dir_weights(files):
        state, description = "success", "No models/tt_dit/ changes"
    else:
        org = repo.split("/")[0]
        members = team_members(org, os.environ.get("DIT_TEAM", "metalium-dit"), token)
        approvers = dit_approvers(api_list(f"/repos/{repo}/pulls/{number}/reviews", token), members)
        if approvers:
            state, description = "success", f"Approved by {', '.join(approvers)}"
        else:
            state, description = "pending", "Needs approval from a DiT team member"
    print(f"{STATUS_CONTEXT}: {state} ({description})")

    if dry_run:
        print("DRY_RUN set; status not posted.")
        return
    body = {"state": state, "context": STATUS_CONTEXT, "description": description[:140]}
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        body["target_url"] = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run_id}"
    status_token = os.environ.get("STATUS_TOKEN") or token
    api("POST", f"/repos/{repo}/statuses/{pr['head']['sha']}", status_token, body)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("assign", "check"):
        print(f"usage: {sys.argv[0]} assign|check", file=sys.stderr)
        return 2
    token = os.environ["GH_TOKEN"]
    repo = os.environ["REPO"]
    number = int(os.environ["PR_NUMBER"])
    dry_run = os.environ.get("DRY_RUN") == "1"

    pr = api("GET", f"/repos/{repo}/pulls/{number}", token)
    files = [f["filename"] for f in api_list(f"/repos/{repo}/pulls/{number}/files", token)]
    (assign if sys.argv[1] == "assign" else check)(repo, number, pr, files, token, dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
