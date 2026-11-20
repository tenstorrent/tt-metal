#!/usr/bin/env python3
"""Targeted review requests and opt-in bot approval and auto-merge for models/tt_dit/ PRs.

Run by .github/workflows/dit-review-impl.yaml. For each PR:

  Reviewers
          Request reviews from the DiT team members who most recently committed to
          the directories the PR touches, so a PR gets 1-2 relevant reviewers rather
          than the whole team. At least one DiT team member is always requested.
          The DiT team-wide request that CODEOWNERS added is then withdrawn, so the
          PR leaves every other member's review queue. This does not weaken the code
          owner rule: GitHub still requires an approval from a code owner. Done once
          per PR, when it is first open and not a draft.

  Auto-merge
          Approve and enable auto-merge for small DiT PRs whose author opted in with
          the dit-auto-merge label (see ineligibility_reasons for the criteria). The
          PR still goes through the merge queue. When a PR stops qualifying (new
          push, label removed, review comments, DiT files removed), the approval is
          dismissed and auto-merge is turned off. A "DiT auto-merge" status explains
          the decision; it always passes, so it never blocks a PR.

The script never checks out or runs PR code; it only reads PR metadata via the API.

Modes:
  find    Write the open PRs this run has to handle to $GITHUB_OUTPUT as prs=<n> ...
          (a PR touching models/tt_dit/, carrying the label, or approved or set to
          auto-merge by the approval account). Needs no org token.
  run     Handle the PRs in PR_NUMBERS.

Environment variables:
  GH_TOKEN       The workflow token: PR reads and review requests, statuses, run lookups.
  ORG_TOKEN      run only: a token with org read access, to list team members.
  APPROVE_TOKEN  The account that approves and enables auto-merge. It must be a code
                 owner of models/tt_dit/ for its approval to count. Without it the
                 approval steps are skipped (the status still reports the decision).
  REPO           owner/name.
  HEAD_REPO, HEAD_BRANCH
                 find only: the open PRs from this branch.
  PR_NUMBER      find only, instead of HEAD_REPO/HEAD_BRANCH: this PR.
  PR_NUMBERS     run only: space-separated PR numbers.
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
DIT_TESTS = DIT_PREFIX + "tests/"
# Bounds the number of commit-history API calls for very large PRs.
MAX_DIRS = 25
# The workflow token's account, which requests the reviewers.
WORKFLOW_BOT = "github-actions[bot]"

AUTOMERGE_CONTEXT = "DiT auto-merge"
OPT_IN_LABEL = "dit-auto-merge"
TEST_WORKFLOW = "all-model-tests.yaml"
# The All Model Tests test types and their test lists (as in all-model-tests.yaml).
TEST_LISTS = {
    "unit": "tests/pipeline_reorg/models_unit_tests.yaml",
    "e2e": "tests/pipeline_reorg/models_e2e_tests.yaml",
}
# SKU groups of the All Model Tests skus input (as in all-model-tests.yaml; the
# exabox SKUs are in neither).
WH_SKUS = ("wh_n150", "wh_n150_civ2", "wh_n300", "wh_llmbox", "wh_llmbox_perf", "wh_galaxy", "wh_galaxy_perf")
BH_SKUS = ("bh_p150", "bh_p150b_civ2", "bh_p300", "bh_quietbox_2", "bh_galaxy", "bh_loudbox")
SKU_GROUPS = {"all": WH_SKUS + BH_SKUS, "wh_all": WH_SKUS, "bh_all": BH_SKUS}
# Auto-merge criteria. New test files under models/tt_dit/tests/ are allowed and do not
# count toward the size limits; changes to existing tests are not (they often loosen
# quality gates).
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


def file_paths(f: dict) -> list[str]:
    """The paths a PR files entry touches: a renamed file's old path too."""
    return [f["filename"]] + ([f["previous_filename"]] if f.get("previous_filename") else [])


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


def pick_reviewers(scores: Counter, members: set[str], author: str, already: set[str], n: int, seed: int) -> list[str]:
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
    return path.startswith(DIT_TESTS)


def ineligibility_reasons(
    files: list[dict],
    author: str,
    members: set[str],
    unresolved_threads: int,
    opted_in: bool,
    tests_passed: bool | None,
    base_ok: bool = True,
) -> list[str]:
    """Why a PR does not qualify for auto-merge; empty means it does.

    `files` are entries from the PR files API (filename, previous_filename, status,
    additions, deletions, patch). `tests_passed` is None when the test runs were not
    looked up.
    """
    reasons = []
    if not opted_in:
        reasons.append(f"author has not added the {OPT_IN_LABEL} label")
    if author not in members:
        reasons.append("author not on DiT team")
    if not base_ok:
        reasons.append("base branch is not the default branch")
    paths = [p for f in files for p in file_paths(f)]
    if any(not p.startswith(DIT_PREFIX) for p in paths):
        reasons.append("changes files outside models/tt_dit/")
    if any(f["status"] != "added" and any(is_test_file(p) for p in file_paths(f)) for f in files):
        reasons.append("changes existing tests")
    if any(p.startswith(DIT_PREFIX + d) for p in paths for d in CORE_DIRS):
        reasons.append("changes core code (" + ", ".join(d.rstrip("/") for d in CORE_DIRS) + ")")
    # The files API leaves out the patch of binary files and of very large diffs.
    if any("patch" not in f and f["status"] != "removed" for f in files):
        reasons.append("changes binary or very large files")
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


def dit_test_legs(test_lists: dict[str, str]) -> list[tuple[str, str, str, int]]:
    """(test type, model, SKU, tier) of every All Model Tests leg that runs models/tt_dit/ tests."""
    import yaml

    legs = []
    for test_type, path in test_lists.items():
        with open(path, encoding="utf-8") as fh:
            for test in yaml.safe_load(fh) or []:
                skus = test.get("skus")
                if DIT_PREFIX not in test.get("cmd", "") or not isinstance(skus, dict):
                    continue
                for sku, cfg in skus.items():
                    legs.append((test_type, test.get("model", ""), sku, (cfg or {}).get("tier")))
    return legs


def model_matches(model: str, model_filter: str) -> bool:
    """All Model Tests' model filter: 'all', or a comma-separated list of substrings."""
    if model_filter == "all":
        return True
    return any(tok.strip() and tok.strip().lower() in model.lower() for tok in model_filter.split(","))


def sku_selected(sku: str, sku_filter: str) -> bool:
    tokens = {t.strip() for t in sku_filter.split(",") if t.strip()} or {"all"}
    return any(sku == t or sku in SKU_GROUPS.get(t, ()) for t in tokens)


def run_covers_dit(title: str, legs: list[tuple[str, str, str, int]]) -> bool:
    """Whether an All Model Tests run, by its title (Tiers[..] Tests[..] Model[..] SKUs[..]),
    selected at least one DiT leg."""
    fields = dict(re.findall(r"(\w+)\[([^\]]*)\]", title))
    tiers = {int(t) for t in fields.get("Tiers", "").split(",") if t.strip().isdigit()}
    types = {t.strip() for t in fields.get("Tests", "").split(",")}
    model_filter = fields.get("Model", "").strip() or "all"
    sku_filter = fields.get("SKUs", "")
    return any(
        test_type in types and tier in tiers and model_matches(model, model_filter) and sku_selected(sku, sku_filter)
        for test_type, model, sku, tier in legs
    )


def dit_tests_passed(repo: str, sha: str, legs: list, token: str) -> bool:
    query = urllib.parse.urlencode({"head_sha": sha, "status": "success", "per_page": 100})
    runs = api("GET", f"/repos/{repo}/actions/workflows/{TEST_WORKFLOW}/runs?{query}", token)["workflow_runs"]
    return any(run_covers_dit(r["display_title"], legs) for r in runs)


def label_adders(events: list) -> list[str]:
    return [
        (e.get("actor") or {}).get("login")
        for e in events
        if e["event"] == "labeled" and e["label"]["name"] == OPT_IN_LABEL
    ]


def has_label(pr: dict) -> bool:
    return any(label["name"] == OPT_IN_LABEL for label in pr["labels"])


def unresolved_thread_count(repo: str, number: int, token: str) -> int:
    owner, name = repo.split("/")
    query = """query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) { reviewThreads(first: 100) { nodes { isResolved } } }
      }
    }"""
    variables = {"owner": owner, "name": name, "number": number}
    result = api("POST", "/graphql", token, {"query": query, "variables": variables})
    threads = result["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    return sum(1 for t in threads if not t["isResolved"])


def set_auto_merge(pr: dict, enable: bool, token: str) -> None:
    def run(mutation: str) -> None:
        query = "mutation($id: ID!) { " + mutation + " }"
        api("POST", "/graphql", token, {"query": query, "variables": {"id": pr["node_id"]}})

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


class BotState:
    """What the approval account has done on a PR."""

    def __init__(self, repo: str, number: int, pr: dict, bot: str | None, token: str):
        # Its latest decisive review; a dismissed approval no longer counts.
        reviews = [
            r
            for r in api_list(f"/repos/{repo}/pulls/{number}/reviews", token)
            if bot and (r.get("user") or {}).get("login") == bot and r["state"] in ("APPROVED", "DISMISSED")
        ]
        self.approval = reviews[-1] if reviews and reviews[-1]["state"] == "APPROVED" else None
        auto_merge = pr["auto_merge"] or {}
        self.auto_merge = bool(bot) and (auto_merge.get("enabled_by") or {}).get("login") == bot

    def __bool__(self) -> bool:
        return bool(self.approval) or self.auto_merge


def automerge(ctx: dict, number: int, pr: dict, files: list[dict], members: set[str], events: list) -> None:
    repo, token, bot, approve_token = ctx["repo"], ctx["token"], ctx["bot"], ctx["approve_token"]
    touches_dit = bool(dir_weights([p for f in files for p in file_paths(f)]))
    state = BotState(repo, number, pr, bot, token)
    if not touches_dit and not has_label(pr) and not state:
        print(f"#{number}: no models/tt_dit/ changes; nothing to do.")
        return

    sha = pr["head"]["sha"]
    if pr["state"] != "open" or pr["draft"]:
        reasons = ["PR is a draft or closed"]
    else:
        adders = label_adders(events)
        # The label is present and the PR author is the one who last added it.
        opted_in = has_label(pr) and bool(adders) and adders[-1] == pr["user"]["login"]
        # Skip the slower checks until the author asks for auto-merge.
        tests_passed = dit_tests_passed(repo, sha, dit_test_legs(TEST_LISTS), token) if opted_in else None
        threads = unresolved_thread_count(repo, number, token) if opted_in else 0
        base_ok = pr["base"]["ref"] == pr["base"]["repo"]["default_branch"]
        reasons = ineligibility_reasons(files, pr["user"]["login"], members, threads, opted_in, tests_passed, base_ok)
    eligible = not reasons

    description = "Approved; will merge once checks pass" if eligible else "Not auto-merging: " + "; ".join(reasons)
    print(f"#{number} {AUTOMERGE_CONTEXT}: {description}")
    if ctx["dry_run"]:
        print("DRY_RUN set; no changes made.")
        return

    if not approve_token:
        print("::warning::No approval token; approval and auto-merge skipped.")
    elif eligible:
        if not state.approval or state.approval["commit_id"] != sha:
            body = (
                f"Auto-approved: meets the DiT auto-merge criteria and the author added `{OPT_IN_LABEL}`. "
                "See .github/scripts/dit_review.py."
            )
            api(
                "POST",
                f"/repos/{repo}/pulls/{number}/reviews",
                approve_token,
                {"commit_id": sha, "event": "APPROVE", "body": body},
            )
        if not pr["auto_merge"]:
            set_auto_merge(pr, True, approve_token)
    else:
        if state.approval:
            message = "No longer meets the DiT auto-merge criteria: " + "; ".join(reasons)
            api(
                "PUT",
                f"/repos/{repo}/pulls/{number}/reviews/{state.approval['id']}/dismissals",
                approve_token,
                {"message": message, "event": "DISMISS"},
            )
        if state.auto_merge:
            set_auto_merge(pr, False, approve_token)

    if touches_dit:
        body = {"state": "success", "context": AUTOMERGE_CONTEXT, "description": description[:140]}
        run_id = os.environ.get("GITHUB_RUN_ID")
        if run_id:
            server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            body["target_url"] = f"{server}/{repo}/actions/runs/{run_id}"
        api("POST", f"/repos/{repo}/statuses/{sha}", token, body)


def assign(ctx: dict, number: int, pr: dict, files: list[str], members: set[str], events: list) -> None:
    repo, token = ctx["repo"], ctx["token"]
    if pr["state"] != "open" or pr["draft"] or pr["user"]["type"] == "Bot":
        return
    dirs = dir_weights(files)
    if not dirs:
        return
    # Once per PR: later events leave the reviewers to the author.
    if any(e["event"] == "review_requested" and (e.get("actor") or {}).get("login") == WORKFLOW_BOT for e in events):
        print(f"#{number}: DiT reviewers already requested.")
        return

    dit_team = ctx["dit_team"]
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

    print(f"#{number}: top DiT candidates by recent commits:")
    for login, score in [(m, scores[m]) for m in sorted(members, key=lambda m: -scores[m])[:5]]:
        print(f"  {login}: {score}")
    print(f"Requesting: {picks or 'nobody (a DiT member is already on the PR)'}")

    # The DiT team only owns models/tt_dit/, so its request is always safe to withdraw,
    # even when the PR also touches other paths.
    to_remove = sorted(requested_teams & {dit_team})
    print(f"Withdrawing team requests: {to_remove or 'none'}")

    if ctx["dry_run"]:
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


def find(ctx: dict) -> list[int]:
    repo, token = ctx["repo"], ctx["token"]
    if os.environ.get("PR_NUMBER"):
        prs = [api("GET", f"/repos/{repo}/pulls/{int(os.environ['PR_NUMBER'])}", token)]
    else:
        head_repo, branch = os.environ["HEAD_REPO"], os.environ["HEAD_BRANCH"]
        query = urllib.parse.urlencode({"head": f"{head_repo.split('/')[0]}:{branch}", "state": "open"})
        prs = [
            p
            for p in api_list(f"/repos/{repo}/pulls?{query}", token)
            if (p["head"]["repo"] or {}).get("full_name") == head_repo
        ]
    numbers = []
    for pr in prs:
        files = api_list(f"/repos/{repo}/pulls/{pr['number']}/files", token)
        touches_dit = bool(dir_weights([p for f in files for p in file_paths(f)]))
        if touches_dit or has_label(pr) or BotState(repo, pr["number"], pr, ctx["bot"], token):
            numbers.append(pr["number"])
    print(f"PRs to handle: {numbers or 'none'}")
    return numbers


def main() -> int:
    modes = ("find", "run")
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print(f"usage: {sys.argv[0]} {'|'.join(modes)}", file=sys.stderr)
        return 2
    repo = os.environ["REPO"]
    token = os.environ["GH_TOKEN"]
    approve_token = os.environ.get("APPROVE_TOKEN", "")
    ctx = {
        "repo": repo,
        "token": token,
        "approve_token": approve_token,
        "bot": api("GET", "/user", approve_token)["login"] if approve_token else None,
        "dit_team": os.environ.get("DIT_TEAM", "metalium-dit"),
        "dry_run": os.environ.get("DRY_RUN") == "1",
    }

    if sys.argv[1] == "find":
        numbers = find(ctx)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
                out.write(f"prs={' '.join(map(str, numbers))}\n")
        return 0

    # The approval account may be on the team to make it a code owner; never request
    # a human review from it.
    members = team_members(repo.split("/")[0], ctx["dit_team"], os.environ["ORG_TOKEN"]) - {ctx["bot"]}
    for number in map(int, os.environ.get("PR_NUMBERS", "").split()):
        pr = api("GET", f"/repos/{repo}/pulls/{number}", token)
        files = api_list(f"/repos/{repo}/pulls/{number}/files", token)
        events = api_list(f"/repos/{repo}/issues/{number}/events", token)
        assign(ctx, number, pr, [f["filename"] for f in files], members, events)
        automerge(ctx, number, pr, files, members, events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
