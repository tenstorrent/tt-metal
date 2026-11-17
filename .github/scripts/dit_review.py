#!/usr/bin/env python3
"""Targeted review requests and opt-in auto-merge for models/tt_dit/ PRs.

Two modes:

  assign  Request reviews from the DiT team members who most recently committed to
          the directories the PR touches, so a PR gets 1-2 relevant reviewers rather
          than the whole team. At least one DiT team member is always requested.
          The DiT team-wide request that CODEOWNERS added is then withdrawn, so the
          PR leaves every other member's review queue. This does not weaken the code
          owner rule: GitHub still requires an approval from a code owner.

  automerge
          Approve and enable auto-merge for small DiT PRs whose author opted in with
          the dit-auto-merge label (see ineligibility_reasons for the criteria). The
          PR still goes through the merge queue. When a PR stops qualifying (new
          push, label removed, review comments), the approval is dismissed and
          auto-merge is turned off. A "DiT auto-merge" status explains the decision;
          it always passes, so it never blocks a PR.

The script never checks out or runs PR code; it only reads PR metadata via the API.
The token's account approves PRs, so it must be a code owner of models/tt_dit/ for
its approval to count.

Environment variables:
  GH_TOKEN       Token with read:org (to list team members) and pull request write
                 access (to change review requests).
  STATUS_TOKEN   Token used to post the commit status (default: GH_TOKEN).
  REPO           owner/name.
  PR_NUMBER      Pull request number.
  HEAD_SHA       automerge only, instead of PR_NUMBER: handle the open PRs whose head
                 is this commit (used when a test run finishes).
  TEST_YAMLS     automerge only: comma-separated test lists used to find the DiT model
                 names a test run must cover (default: the models unit and e2e lists).
  DIT_TEAM       Team whose members count as DiT reviewers (default: metalium-dit).
  NUM_REVIEWERS  Number of reviewers to request (default: 2).
  HISTORY_DAYS   How far back to look at commit history (default: 180).
  DRY_RUN        "1" prints what would change without changing anything.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

API = "https://api.github.com"
DIT_PREFIX = "models/tt_dit/"
# Bounds the number of commit-history API calls for very large PRs.
MAX_DIRS = 25

AUTOMERGE_CONTEXT = "DiT auto-merge"
OPT_IN_LABEL = "dit-auto-merge"
TEST_WORKFLOW = "all-model-tests.yaml"
DEFAULT_TEST_YAMLS = "tests/pipeline_reorg/models_unit_tests.yaml,tests/pipeline_reorg/models_e2e_tests.yaml"
# Auto-merge criteria. New test files are allowed and do not count toward the size
# limits; changes to existing tests are not (they often loosen quality gates).
MAX_SOURCE_FILES = 3
MAX_SOURCE_LINES = 40
# Shared by every pipeline, so changes here always get a human review.
CORE_DIRS = ("layers/", "parallel/", "utils/")


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
    result = json.loads(payload) if payload else None
    if path == "/graphql" and result.get("errors"):
        raise RuntimeError(f"GraphQL error: {result['errors']}")
    return result


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


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith(DIT_PREFIX + "tests/") or name.startswith("test_") or name == "conftest.py"


def ineligibility_reasons(
    files: list[dict],
    author: str,
    members: set[str],
    unresolved_threads: int,
    opted_in: bool,
    tests_passed: bool | None,
) -> list[str]:
    """Why a PR does not qualify for auto-merge; empty means it does.

    `files` are entries from the PR files API (filename, status, additions, deletions).
    `tests_passed` is None when the test runs were not looked up.
    """
    reasons = []
    if not opted_in:
        reasons.append(f"author has not added the {OPT_IN_LABEL} label")
    if author not in members:
        reasons.append("author not on DiT team")
    if any(not f["filename"].startswith(DIT_PREFIX) for f in files):
        reasons.append("changes files outside models/tt_dit/")
    if any(is_test_file(f["filename"]) and f["status"] != "added" for f in files):
        reasons.append("changes existing tests")
    if any(f["filename"].startswith(DIT_PREFIX + d) for f in files for d in CORE_DIRS):
        reasons.append("changes core code (" + ", ".join(d.rstrip("/") for d in CORE_DIRS) + ")")
    source = [f for f in files if not is_test_file(f["filename"])]
    if len(source) > MAX_SOURCE_FILES:
        reasons.append(f"{len(source)} non-test files (max {MAX_SOURCE_FILES})")
    lines = sum(f["additions"] + f["deletions"] for f in source)
    if lines > MAX_SOURCE_LINES:
        reasons.append(f"{lines} non-test lines (max {MAX_SOURCE_LINES})")
    if unresolved_threads:
        reasons.append(f"{unresolved_threads} unresolved review threads")
    if tests_passed is False:
        reasons.append("no passing All Model Tests run covering DiT on the latest commit")
    return reasons


def dit_test_models(paths: list[str]) -> set[str]:
    """Model names of test entries that run models/tt_dit/ tests."""
    models = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for entry in fh.read().split("\n- name:"):
                match = re.search(r"^  model: (\S+)", entry, re.MULTILINE)
                if match and DIT_PREFIX in entry:
                    models.add(match.group(1))
    return models


def run_covers_dit(title: str, dit_models: set[str]) -> bool:
    """Whether an All Model Tests run, by its title, ran at least one DiT model. The
    model input is a substring filter, shown in the title as Model[...]."""
    match = re.search(r"Model\[([^\]]*)\]", title)
    model_filter = match.group(1).strip() if match else ""
    return model_filter == "all" or (bool(model_filter) and any(model_filter in m for m in dit_models))


def dit_tests_passed(repo: str, sha: str, dit_models: set[str], token: str) -> bool:
    query = urllib.parse.urlencode({"head_sha": sha, "status": "success", "per_page": 100})
    runs = api("GET", f"/repos/{repo}/actions/workflows/{TEST_WORKFLOW}/runs?{query}", token)["workflow_runs"]
    return any(run_covers_dit(r["display_title"], dit_models) for r in runs)


def author_opted_in(repo: str, number: int, pr: dict, token: str) -> bool:
    """The opt-in label is present and the PR author is the one who last added it."""
    if not any(label["name"] == OPT_IN_LABEL for label in pr["labels"]):
        return False
    added = [
        e
        for e in api_list(f"/repos/{repo}/issues/{number}/events", token)
        if e["event"] == "labeled" and e["label"]["name"] == OPT_IN_LABEL
    ]
    return bool(added) and (added[-1].get("actor") or {}).get("login") == pr["user"]["login"]


def unresolved_thread_count(repo: str, number: int, token: str) -> int:
    owner, name = repo.split("/")
    query = """query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) { reviewThreads(first: 100) { nodes { isResolved } } }
      }
    }"""
    result = api("POST", "/graphql", token, {"query": query, "variables": {"owner": owner, "name": name, "number": number}})
    threads = result["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    return sum(1 for t in threads if not t["isResolved"])


def set_auto_merge(pr: dict, enable: bool, token: str) -> None:
    def run(mutation: str) -> None:
        api("POST", "/graphql", token, {"query": "mutation($id: ID!) { " + mutation + " }", "variables": {"id": pr["node_id"]}})

    if not enable:
        run("disablePullRequestAutoMerge(input: {pullRequestId: $id}) { clientMutationId }")
        return
    try:
        run("enablePullRequestAutoMerge(input: {pullRequestId: $id, mergeMethod: SQUASH}) { clientMutationId }")
    except RuntimeError as e:
        # GitHub refuses auto-merge when the PR can already merge ("clean status");
        # then add it to the merge queue directly.
        if "clean status" not in str(e):
            raise
        run("enqueuePullRequest(input: {pullRequestId: $id}) { clientMutationId }")


def automerge(repo: str, number: int, pr: dict, file_objs: list[dict], token: str, dry_run: bool) -> None:
    touches_dit = bool(dir_weights([f["filename"] for f in file_objs]))
    has_label = any(label["name"] == OPT_IN_LABEL for label in pr["labels"])
    if not touches_dit and not has_label:
        print("No models/tt_dit/ changes; nothing to do.")
        return

    org = repo.split("/")[0]
    members = team_members(org, os.environ.get("DIT_TEAM", "metalium-dit"), token)
    bot = api("GET", "/user", token)["login"]
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or pr["draft"]:
        reasons = ["PR is a draft or closed"]
    else:
        opted_in = author_opted_in(repo, number, pr, token)
        # Skip the slower checks until the author asks for auto-merge.
        tests_passed = None
        if opted_in:
            yamls = os.environ.get("TEST_YAMLS", DEFAULT_TEST_YAMLS).split(",")
            tests_passed = dit_tests_passed(repo, sha, dit_test_models(yamls), token)
        threads = unresolved_thread_count(repo, number, token) if opted_in else 0
        reasons = ineligibility_reasons(file_objs, pr["user"]["login"], members, threads, opted_in, tests_passed)
    eligible = not reasons

    # The bot's latest decisive review; a dismissed approval no longer counts.
    bot_reviews = [
        r for r in api_list(f"/repos/{repo}/pulls/{number}/reviews", token)
        if (r.get("user") or {}).get("login") == bot and r["state"] in ("APPROVED", "DISMISSED")
    ]
    bot_approval = bot_reviews[-1] if bot_reviews and bot_reviews[-1]["state"] == "APPROVED" else None
    auto_merge_by_bot = bool(pr["auto_merge"]) and (pr["auto_merge"].get("enabled_by") or {}).get("login") == bot

    description = "Approved; will merge once checks pass" if eligible else "Not auto-merging: " + "; ".join(reasons)
    print(f"{AUTOMERGE_CONTEXT}: {description}")
    if dry_run:
        print("DRY_RUN set; no changes made.")
        return

    if eligible:
        if not bot_approval or bot_approval["commit_id"] != sha:
            api(
                "POST",
                f"/repos/{repo}/pulls/{number}/reviews",
                token,
                {
                    "commit_id": sha,
                    "event": "APPROVE",
                    "body": f"Auto-approved: meets the DiT auto-merge criteria and the author added `{OPT_IN_LABEL}`. "
                    "See .github/scripts/dit_review.py.",
                },
            )
        if not pr["auto_merge"]:
            set_auto_merge(pr, True, token)
    else:
        if bot_approval:
            api(
                "PUT",
                f"/repos/{repo}/pulls/{number}/reviews/{bot_approval['id']}/dismissals",
                token,
                {"message": "No longer meets the DiT auto-merge criteria: " + "; ".join(reasons), "event": "DISMISS"},
            )
        if auto_merge_by_bot:
            set_auto_merge(pr, False, token)

    if touches_dit:
        body = {"state": "success", "context": AUTOMERGE_CONTEXT, "description": description[:140]}
        run_id = os.environ.get("GITHUB_RUN_ID")
        if run_id:
            body["target_url"] = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run_id}"
        status_token = os.environ.get("STATUS_TOKEN") or token
        api("POST", f"/repos/{repo}/statuses/{sha}", status_token, body)


def assign(repo: str, number: int, pr: dict, files: list[str], token: str, dry_run: bool) -> None:
    if pr["draft"]:
        print("Draft PR; reviewers are assigned when it is marked ready for review.")
        return
    dirs = dir_weights(files)
    if not dirs:
        print("No models/tt_dit/ changes; nothing to do.")
        return

    org = repo.split("/")[0]
    dit_team = os.environ.get("DIT_TEAM", "metalium-dit")
    # The auto-merge bot account may be on the team to make it a code owner; never
    # request a human review from it.
    members = team_members(org, dit_team, token) - {api("GET", "/user", token)["login"]}
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

    # The DiT team only owns models/tt_dit/, so its request is always safe to withdraw,
    # even when the PR also touches other paths.
    to_remove = sorted(requested_teams & {dit_team})
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


def main() -> int:
    modes = ("assign", "automerge")
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print(f"usage: {sys.argv[0]} {'|'.join(modes)}", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    token = os.environ["GH_TOKEN"]
    repo = os.environ["REPO"]
    dry_run = os.environ.get("DRY_RUN") == "1"
    if os.environ.get("PR_NUMBER"):
        numbers = [int(os.environ["PR_NUMBER"])]
    else:
        # A finished test run only knows its commit; find the open PRs at that commit.
        sha = os.environ["HEAD_SHA"]
        numbers = [
            p["number"]
            for p in api("GET", f"/repos/{repo}/commits/{sha}/pulls", token)
            if p["state"] == "open" and p["head"]["sha"] == sha
        ]
        print(f"Open PRs at {sha}: {numbers or 'none'}")

    for number in numbers:
        pr = api("GET", f"/repos/{repo}/pulls/{number}", token)
        file_objs = api_list(f"/repos/{repo}/pulls/{number}/files", token)
        if mode == "automerge":
            automerge(repo, number, pr, file_objs, token, dry_run)
        else:
            assign(repo, number, pr, [f["filename"] for f in file_objs], token, dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
