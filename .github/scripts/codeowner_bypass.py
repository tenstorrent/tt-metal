#!/usr/bin/env python3
"""Decision logic for `.github/workflows/codeowner-bypass.yaml`.

Two subcommands, both configured through environment variables:

  bless    A bypass-team member commented `/codeowners bypass`: label the PR.
  enqueue  Put a blessed PR into the merge queue once everything is satisfied.

All GitHub access goes through the `Gh` interface so the decisions can be tested with an
in-memory fake (see test_codeowner_bypass.py). `GhCli` is the real implementation and shells
out to `gh`, which reads GH_TOKEN and GH_REPO from the environment.

Environment (bless):  COMMENT_BODY, COMMENT_ID, COMMENT_AUTHOR, PR_NUMBER
Environment (enqueue): EVENT_PR_NUMBER, RUN_PR_NUMBER, RUN_HEAD_SHA (all optional; at least
                       one identifies the PR)
Environment (both):   GH_REPO, BYPASS_LABEL, BYPASS_TEAM, ORG
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

BLESS_RE = re.compile(r"^/codeowners?[ \t]+bypass[ \t]*$", re.MULTILINE)
PASSING_CONCLUSIONS = {"success", "neutral", "skipped"}

PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      id state isDraft headRefOid baseRefName isInMergeQueue
      autoMergeRequest { enabledAt }
      labels(first: 100) { nodes { name } }
    }
  }
}
"""
ENQUEUE_MUTATION = (
    "mutation($id: ID!) { enqueuePullRequest(input: {pullRequestId: $id}) { mergeQueueEntry { position } } }"
)


class GhError(Exception):
    pass


class Gh:
    """What the decisions need from GitHub. Overridden by GhCli and by test fakes."""

    def viewer_login(self) -> str:
        raise NotImplementedError

    def team_state(self, org: str, team: str, user: str) -> str:
        """'active' for current members, anything else otherwise."""
        raise NotImplementedError

    def pull_request(self, number: int) -> dict:
        """Keys: id, state, isDraft, headRefOid, baseRefName, isInMergeQueue, autoMergeRequest, labels."""
        raise NotImplementedError

    def open_pr_for_sha(self, sha: str) -> Optional[int]:
        raise NotImplementedError

    def label_events(self, number: int, label: str) -> list:
        """Actor logins of `labeled` events for `label`, oldest first."""
        raise NotImplementedError

    def required_contexts(self, base_ref: str) -> list:
        raise NotImplementedError

    def check_runs(self, sha: str) -> list:
        """Dicts with name, id, status, conclusion."""
        raise NotImplementedError

    def commit_statuses(self, sha: str) -> list:
        """Dicts with context, state."""
        raise NotImplementedError

    def enqueue(self, node_id: str) -> Optional[str]:
        """Enqueue; returns the queue position. Raises GhError with GitHub's message on failure."""
        raise NotImplementedError

    def comment(self, number: int, body: str) -> None:
        raise NotImplementedError

    def react(self, comment_id: str, content: str) -> None:
        raise NotImplementedError

    def ensure_label(self, label: str, description: str) -> None:
        raise NotImplementedError

    def remove_label(self, number: int, label: str) -> None:
        raise NotImplementedError

    def add_label(self, number: int, label: str) -> None:
        raise NotImplementedError


class GhCli(Gh):
    def __init__(self, repo: str):
        self.repo = repo

    def _gh(self, *args: str) -> str:
        proc = subprocess.run(["gh", *args], capture_output=True, text=True)
        if proc.returncode != 0:
            raise GhError((proc.stdout + proc.stderr).strip())
        return proc.stdout

    def _api(self, path: str, *fields: str, method: str = "GET"):
        out = self._gh("api", "-X", method, path, *fields)
        return json.loads(out) if out.strip() else None

    def _pages(self, path: str) -> list:
        return json.loads(self._gh("api", "--paginate", "--slurp", path))

    def viewer_login(self) -> str:
        return self._api("user")["login"]

    def team_state(self, org: str, team: str, user: str) -> str:
        try:
            return self._api(f"orgs/{org}/teams/{team}/memberships/{user}")["state"]
        except GhError:
            return "none"

    def _graphql(self, query: str, **variables) -> dict:
        args = ["api", "graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        data = json.loads(self._gh(*args))
        if data.get("errors"):
            raise GhError(data["errors"][0].get("message", "GraphQL error"))
        return data["data"]

    def pull_request(self, number: int) -> dict:
        owner, name = self.repo.split("/", 1)
        pr = self._graphql(PR_QUERY, owner=owner, name=name, number=number)["repository"]["pullRequest"]
        pr["labels"] = [node["name"] for node in pr["labels"]["nodes"]]
        return pr

    def open_pr_for_sha(self, sha: str) -> Optional[int]:
        prs = self._api(f"repos/{self.repo}/commits/{sha}/pulls") or []
        return next((pr["number"] for pr in prs if pr["state"] == "open"), None)

    def label_events(self, number: int, label: str) -> list:
        events = [e for page in self._pages(f"repos/{self.repo}/issues/{number}/timeline?per_page=100") for e in page]
        return [
            (e.get("actor") or {}).get("login", "")
            for e in events
            if e.get("event") == "labeled" and e["label"]["name"] == label
        ]

    def required_contexts(self, base_ref: str) -> list:
        rules = self._api(f"repos/{self.repo}/rules/branches/{base_ref}") or []
        contexts = {
            check["context"]
            for rule in rules
            if rule["type"] == "required_status_checks"
            for check in rule["parameters"]["required_status_checks"]
        }
        return sorted(contexts)

    def check_runs(self, sha: str) -> list:
        pages = self._pages(f"repos/{self.repo}/commits/{sha}/check-runs?per_page=100")
        return [run for page in pages for run in page["check_runs"]]

    def commit_statuses(self, sha: str) -> list:
        return self._api(f"repos/{self.repo}/commits/{sha}/status?per_page=100")["statuses"]

    def enqueue(self, node_id: str) -> Optional[str]:
        data = self._graphql(ENQUEUE_MUTATION, id=node_id)
        entry = (data.get("enqueuePullRequest") or {}).get("mergeQueueEntry") or {}
        return entry.get("position")

    def comment(self, number: int, body: str) -> None:
        self._api(f"repos/{self.repo}/issues/{number}/comments", "-f", f"body={body}", method="POST")

    def react(self, comment_id: str, content: str) -> None:
        try:
            self._api(
                f"repos/{self.repo}/issues/comments/{comment_id}/reactions", "-f", f"content={content}", method="POST"
            )
        except GhError:
            pass

    def ensure_label(self, label: str, description: str) -> None:
        try:
            self._gh("label", "create", label, "--color", "B60205", "--description", description)
        except GhError:
            pass  # already exists

    def remove_label(self, number: int, label: str) -> None:
        try:
            self._api(f"repos/{self.repo}/issues/{number}/labels/{label}", method="DELETE")
        except GhError:
            pass  # not present

    def add_label(self, number: int, label: str) -> None:
        self._api(f"repos/{self.repo}/issues/{number}/labels", "-f", f"labels[]={label}", method="POST")


@dataclass
class Config:
    label: str = "codeowner-bypass"
    team: str = "codeowner-bypass"
    org: str = "tenstorrent"


def bless(gh: Gh, cfg: Config, pr_number: int, comment_id: str, author: str, body: str) -> None:
    # Only the exact command on its own line; other /codeowners commands are handled by
    # codeowners-group-analysis.yaml.
    if not BLESS_RE.search(body):
        print("Not a bypass command; nothing to do.")
        return

    gh.react(comment_id, "eyes")

    if gh.team_state(cfg.org, cfg.team, author) != "active":
        print(f"{author} is not an active member of {cfg.org}/{cfg.team}")
        gh.comment(
            pr_number,
            f"❌ @{author}: only members of @{cfg.org}/{cfg.team} can bless a PR for code-owner bypass.",
        )
        return

    if gh.pull_request(pr_number)["state"] != "OPEN":
        gh.comment(pr_number, "❌ PR is not open; not blessing.")
        return

    gh.ensure_label(cfg.label, "Blessed by the codeowner-bypass team; Merge when ready will enqueue it")
    # Re-adding a label that is already present records no new `labeled` event, so drop it
    # first: the enqueue check requires that the latest one is the bot's.
    gh.remove_label(pr_number, cfg.label)
    gh.add_label(pr_number, cfg.label)
    gh.comment(
        pr_number,
        f"✅ Blessed for code-owner bypass by @{author}. Click **Merge when ready**: once auto-merge is on and "
        "all required checks are green, the bot enqueues the PR (click it again if the PR is evicted). "
        "Required checks and the merge queue still apply.",
    )


def not_green(required: list, check_runs: list, statuses: list) -> dict:
    """Required contexts that are not passing, mapped to why. Empty means all green.

    The newest run per check name wins (a rerun has a higher id), and a context may also be
    satisfied by a legacy commit status. Skipped and neutral count as passing, as in GitHub.
    """
    latest = {}
    for run in check_runs:
        if run["name"] not in latest or run["id"] > latest[run["name"]]["id"]:
            latest[run["name"]] = run
    results = {name: (run["conclusion"] if run["status"] == "completed" else "pending") for name, run in latest.items()}
    status_results = {s["context"]: s["state"] for s in statuses}

    bad = {}
    for context in required:
        result = results.get(context, status_results.get(context, "missing"))
        if result not in PASSING_CONCLUSIONS:
            bad[context] = result
    return bad


def evaluate_enqueue(
    gh: Gh,
    cfg: Config,
    event_pr_number: Optional[int] = None,
    run_pr_number: Optional[int] = None,
    run_head_sha: Optional[str] = None,
):
    """Returns (node_id, None) when the PR should be enqueued now, else (None, reason).

    Deliberately independent of which event fired and of the triggering workflow's overall
    conclusion: every trigger re-evaluates the same condition, and the required-check
    verification below is the only judge of whether the checks are green.
    """
    pr_number = event_pr_number or run_pr_number
    if not pr_number and run_head_sha:
        # workflow_run leaves pull_requests empty for fork PRs; look the PR up by SHA.
        pr_number = gh.open_pr_for_sha(run_head_sha)
    if not pr_number:
        return None, "no open PR for this event"

    pr = gh.pull_request(pr_number)
    if pr["state"] != "OPEN":
        return None, f"PR #{pr_number} is not open"
    if pr["isDraft"]:
        return None, f"PR #{pr_number} is a draft"
    if pr["isInMergeQueue"]:
        return None, f"PR #{pr_number} is already in the merge queue"
    if not pr["autoMergeRequest"]:
        return None, f"auto-merge is not enabled on PR #{pr_number}"
    if cfg.label not in pr["labels"]:
        return None, f"PR #{pr_number} has no {cfg.label} label"

    head_sha = pr["headRefOid"]
    # A finished run for an older commit says nothing about the current head.
    if run_head_sha and run_head_sha != head_sha:
        return None, f"workflow run is for {run_head_sha}, PR head is {head_sha}"

    # The latest application of the label must be the bot's, not a hand-added one.
    labelers = gh.label_events(pr_number, cfg.label)
    labeler = labelers[-1] if labelers else ""
    if labeler != gh.viewer_login():
        return None, f"{cfg.label} was last applied by '{labeler}', not the bot"

    # PR Gate Status is not an aggregate of everything required, so verify every context
    # the ruleset requires.
    required = gh.required_contexts(pr["baseRefName"])
    if not required:
        raise GhError(f"could not read required status checks for {pr['baseRefName']}")
    bad = not_green(required, gh.check_runs(head_sha), gh.commit_statuses(head_sha))
    if bad:
        detail = "".join(f"\n  - {name}: {why}" for name, why in sorted(bad.items()))
        return None, f"required checks not green on {head_sha}:{detail}"

    return (pr_number, pr["id"]), None


def enqueue(gh: Gh, cfg: Config, **ids) -> int:
    """Returns the process exit code."""
    target, reason = evaluate_enqueue(gh, cfg, **ids)
    if target is None:
        print(f"Skipping: {reason}")
        return 0
    pr_number, node_id = target
    try:
        position = gh.enqueue(node_id)
    except GhError as err:
        print(err)
        gh.comment(pr_number, f"❌ Code-owner bypass: enqueue failed: {err or 'see the workflow run for details'}.")
        return 1
    gh.comment(
        pr_number,
        f"✅ Code-owner bypass: auto-merge is on and all required checks are green, so the bot enqueued this PR "
        f"(queue position {position or '?'}). If it is evicted, click **Merge when ready** again.",
    )
    return 0


def _int(name: str) -> Optional[int]:
    value = os.environ.get(name, "")
    return int(value) if value else None


def main(argv: list) -> int:
    if len(argv) != 2 or argv[1] not in ("bless", "enqueue"):
        print(__doc__)
        return 2
    gh = GhCli(os.environ["GH_REPO"])
    cfg = Config(
        label=os.environ.get("BYPASS_LABEL", Config.label),
        team=os.environ.get("BYPASS_TEAM", Config.team),
        org=os.environ.get("ORG", Config.org),
    )
    if argv[1] == "bless":
        bless(
            gh,
            cfg,
            pr_number=int(os.environ["PR_NUMBER"]),
            comment_id=os.environ["COMMENT_ID"],
            author=os.environ["COMMENT_AUTHOR"],
            body=os.environ["COMMENT_BODY"],
        )
        return 0
    return enqueue(
        gh,
        cfg,
        event_pr_number=_int("EVENT_PR_NUMBER"),
        run_pr_number=_int("RUN_PR_NUMBER"),
        run_head_sha=os.environ.get("RUN_HEAD_SHA") or None,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
