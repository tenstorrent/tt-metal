#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the agent's build and verify on hardware it cannot reach, and hand back only the answer.

The agent runs on a GitHub-hosted runner, because that is the only place the Copilot API is
reachable -- CIv2 egresses through a proxy that answers 403 to `api.githubcopilot.com`. A hosted
runner has four cores and cannot build tt-metal at any useful speed, and it has no card at all. So
the two things the agent's loop is made of, compiling and measuring, both happen somewhere else:
this pushes the working tree to a scratch ref, which starts `port-measure.yaml` on CIv2, waits, and
prints the verdict.

It runs host-side, in the agent job but outside the agent's sandbox. That distinction is the whole
security design and it is narrower than it looks. The sandbox is handed
`${RUNNER_TEMP}/gh-aw` read-only and the workspace read-write, and awf is invoked with `--env-all`,
so a credential in either place -- including one interpolated into an mcp-scripts handler, which is
what `${{ secrets.X }}` in a tool body compiles to -- is a credential the agent can read. The token
therefore arrives by neither route: a pre-step writes it to a file under `$HOME`, which is in none
of those mounts, and this reads that file. See HANDOFF.md for how that was established.

The credential must be a PAT rather than the job's own `GITHUB_TOKEN`, and not for the reason one
would guess. It is not about scope: it is that GitHub deliberately does not start workflow runs from
pushes made with `GITHUB_TOKEN`, to stop runs from triggering runs. Since the push *is* the trigger
here, that suppression would leave this waiting five minutes for a run that was never going to
happen. `PORT_PUSH_TOKEN` needs `contents: write` to push the scratch ref and `actions: read` to
read back the run and its artifacts.

Four things here are less obvious than they look:

  Nothing is dispatched, despite the name. `port-measure.yaml` triggers on pushes to
  `port-op-scratch/**`, and a `push` event runs the workflow as it exists in the pushed commit --
  unlike `workflow_dispatch`, which only ever sees workflow files on the default branch. Since the
  code under test has to be pushed anyway, using the push as the trigger is both one fewer API call
  and the thing that lets the entire pipeline be developed and exercised on a branch.

  The parameters travel in the commit message, as JSON. A push event has no inputs, and the message
  is the only free-form channel that arrives in the event payload -- so the workflow can read it
  without a checkout -- and leaves nothing behind in the tree. A parameters *file* would be visible
  to gate.py's write-path guard, which diffs the checkout against the base commit, and would read
  as the agent having written somewhere it was not allowed to.

  The commit never touches the agent's tree. A temporary index is populated from the worktree and
  turned into a commit with `commit-tree`, parented on the base. HEAD does not move, the agent's
  index is untouched, and every dispatch is exactly one commit on top of the base -- which is what
  lets the measure job find the base with a depth-1 fetch instead of a full clone.

  Waiting is loud. A verify is 35-45 minutes of nothing, and a silent process that long is
  indistinguishable from a hung one to anything watching -- including whoever is reading the job
  log at the time. The heartbeat goes to stderr so it does not pad the tool result the agent reads.

  Waiting is also resumable, and for the agent it has to be. gh-aw caps an MCP tool call at ten
  minutes and will not compile a higher ceiling, which is shorter than a build and far shorter than
  a verify. So `--start` pushes and hands back a handle, and `--wait HANDLE` blocks for a bounded
  slice before saying "still going, ask again". The workflow's own baseline step is a plain shell
  step under no such cap and still runs this end to end, which is why both paths exist.

  Nothing outlives the process that started it, as far as it can be helped. A tool call cancelled
  mid-flight used to leave a CIv2 run holding a card with nobody reading the result -- the first
  real agentic run ended with four of them at once. `--start` now retires whatever it supersedes.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

WORKFLOW_FILE = "port-measure.yaml"
# The workflow a chained attempt starts: this one's own, with an agent, not the measurement job. It is
# the compiled lock rather than the markdown, because gh-aw's lock file is what Actions runs.
PORT_OP_WORKFLOW = "port-op.lock.yml"
SCRATCH_PREFIX = "port-op-scratch"

# The workflow's own name for what it produced, minus the run id it appends.
RESULTS_ARTIFACT_PREFIX = "port-results-"

POLL_SECONDS = 20
HEARTBEAT_SECONDS = 120

# How long a single `--wait` may block. The MCP gateway cancels a tool call at 600s and gh-aw will
# not allow a higher ceiling, so this has to leave room for what happens after the run completes --
# downloading the artifact, laying it over the tree -- and for the gateway's own overhead. A cancelled
# call is not merely a lost answer: the run keeps burning a card with nobody reading it.
WAIT_BUDGET = 420
# How long to wait for a dispatched run to become visible in the run list. Dispatch is not
# instantaneous and neither is indexing; a minute has been enough, five is not tight.
RUN_APPEARS_TIMEOUT = 300
# Everything after that. `port-measure.yaml` caps its own device job at 120 minutes and a build is
# 15-25, so a wait past this means something is queued behind hardware that is not coming free.
RUN_COMPLETE_TIMEOUT = 3 * 60 * 60

VERDICT_WIN = "win"


def log(message: str) -> None:
    """Progress, not results. stderr keeps it out of the tool result the agent pays for."""
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# GitHub REST, without a dependency


class DispatchError(RuntimeError):
    """The launcher could not do its job. Distinct from a failing verdict, which is a result."""


class Refusal(DispatchError):
    """The launcher declined on purpose, and the agent is the one who can resolve it.

    Separate from `DispatchError` only because of how the two are surfaced. A refusal is an answer
    -- it says what to change -- and must not reach the agent dressed as a broken tool, because the
    reasonable response to a broken tool is to call it again. A genuine harness failure is a broken
    tool and should look like one.
    """


class ApiError(DispatchError):
    """A GitHub call that failed. Recoverable at some call sites -- a heartbeat that cannot list
    jobs should say so and keep waiting, not abort a measurement that is still running."""


class Api:
    def __init__(self, token: str, repo: str, base_url: str) -> None:
        self._token = token
        self.repo = repo
        self.base = base_url.rstrip("/")

    def _request(self, method: str, url: str, redirect: bool = True, body: dict | None = None):
        if url.startswith("/"):
            url = f"{self.base}{url}"
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, method=method, data=payload)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")

        opener = urllib.request.build_opener() if redirect else urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=120) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            if not redirect and exc.code in (301, 302, 303, 307, 308):
                return exc.code, exc.headers, b""
            # The body carries GitHub's actual complaint, which is the only useful part.
            detail = exc.read().decode("utf-8", "replace")[:600]
            raise ApiError(f"{method} {url} failed: HTTP {exc.code} {detail}") from None

    def get_json(self, url: str) -> dict:
        _, _, body = self._request("GET", url)
        return json.loads(body)

    def post(self, url: str) -> None:
        """Fire and forget. Only used to cancel a run, whose reply carries nothing worth reading."""
        self._request("POST", url)

    def post_json(self, url: str, body: dict) -> None:
        """POST with a body, for `workflow_dispatch`, whose 204 carries nothing either."""
        self._request("POST", url, body=body)

    def token_scopes(self) -> str | None:
        """What the credential is allowed to do, straight from GitHub, or None if it will not say.

        Only ever called to explain a failure. A classic PAT's scopes come back in a header. A
        fine-grained or App token sends no such header, and that silence means *unknown*, not
        *none* -- reading it as none would condemn exactly the tokens worth preferring here.
        """
        try:
            _, headers, _ = self._request("GET", f"{self.base}/")
        except ApiError:
            return None
        scopes = headers.get("x-oauth-scopes")
        if scopes is None:
            return None
        return scopes or "(none)"

    def download(self, url: str) -> bytes:
        """Follow the signed-URL redirect GitHub uses for logs and artifacts.

        The redirect target is pre-authenticated and rejects an `Authorization` header, so it has to
        be fetched as a separate unauthenticated request rather than by letting urllib follow it.
        """
        status, headers, body = self._request("GET", url, redirect=False)
        if status in (301, 302, 303, 307, 308):
            with urllib.request.urlopen(headers["Location"], timeout=300) as response:
                return response.read()
        return body


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


# ---------------------------------------------------------------------------------------------
# Getting the working tree to CIv2


def git(*args: str, cwd: Path, env: dict | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise DispatchError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def refuse_pipeline_edits(repo: Path, base: str, commit: str) -> None:
    """Refuse to send a snapshot that modifies the pipeline grading it.

    This is the one place the push trigger costs something. The workflow that runs is the copy in
    the commit that was pushed, so an edit to `.github/workflows/port-measure.yaml` would not be a
    proposal -- it would be the next thing to execute, on CIv2, with the generator credential in
    scope. The same goes for the harness scripts: `gate.py` is the thing deciding whether the port
    is any good, and it also comes from the scratch commit.

    A narrow push token makes most of this GitHub's problem, since it rejects pushes touching
    `.github/workflows/**` without an explicit workflow scope. A classic `repo` PAT may carry that
    scope, so the guard is here rather than assumed. It is cheap and it does not depend on which
    credential happens to be configured.

    `gate.py`'s write-path check covers the same ground from the other side, but only at verify
    time -- which is after the modified workflow would already have run.
    """
    changed = git("diff", "--name-only", base, commit, cwd=repo).splitlines()
    pipeline = sorted(p for p in changed if p.startswith(".github/"))
    if pipeline:
        raise Refusal(
            "refusing to dispatch: this would change the pipeline that measures and grades the "
            "port, and the pushed copy is the one that runs. Revert these and try again:\n  "
            + "\n  ".join(pipeline)
        )


def commit_worktree(repo: Path, work: Path, message: str) -> tuple[str, str]:
    """Turn the current worktree into one commit on top of HEAD, without disturbing either.

    A scratch index means `git add -A` here cannot collide with whatever the agent has staged, and
    `commit-tree` writes an object without moving a ref. The scaffolded `codegen/` sources are
    untracked, so `-A` rather than a diff; `.git/info/exclude` is what keeps the generator checkout
    from being swept in with them.

    The agent job checks out with `submodules: false`, so every submodule here is an empty directory.
    `git add -A` leaves those gitlinks alone rather than staging a deletion, which is what lets the
    measure job check the same commit out with submodules and build it.
    """
    base = git("rev-parse", "HEAD", cwd=repo)
    index = work / "scratch-index"
    index.unlink(missing_ok=True)
    env = {"GIT_INDEX_FILE": str(index)}

    git("read-tree", "HEAD", cwd=repo, env=env)
    git("add", "-A", cwd=repo, env=env)
    tree = git("write-tree", cwd=repo, env=env)
    if tree == git("rev-parse", "HEAD^{tree}", cwd=repo):
        log("note: the working tree is identical to the base commit")

    commit = git(
        "-c",
        "user.name=port-dispatch",
        "-c",
        "user.email=port-dispatch@tenstorrent.com",
        "commit-tree",
        tree,
        "-p",
        base,
        "-m",
        message,
        cwd=repo,
    )
    return commit, base


# GitHub's git server decides, on every push, whether the push creates or updates a workflow file,
# because a token without `workflow` scope may not. On this repo that decision routinely times out
# and the push is rejected with a message blaming the scope -- see tenstorrent/tt-metal#32354, which
# hit the fork-mirroring workflow the same way and is labelled `infra-ci`. It is a false negative:
# cli/cli#13635 reports it against a token that *did* hold the permission, and the same push succeeds
# on a retry. Nothing about the scratch commit provokes it; `refuse_pipeline_edits` has already
# guaranteed the push changes no workflow at all.
PUSH_ATTEMPTS = 5
PUSH_RETRY_MESSAGES = (
    "Unable to determine if workflow can be created or updated",
    "the remote end hung up unexpectedly",
)


def push_ref(
    repo: Path,
    remote: str,
    refspec: str,
    git_env: dict,
    *,
    required: bool = True,
    scopes: callable = None,
) -> bool:
    """Push, retrying the rejection GitHub hands out when its own workflow check times out.

    Backs off rather than hammering, since the failure is a server-side deadline and an immediate
    retry is the one most likely to hit the same cold cache.

    Retrying only helps when the check timed out despite the credential being allowed to push
    workflows. A classic PAT without `workflow` scope will be rejected every time, so the first
    rejection asks GitHub what the token actually holds and gives up immediately if that is the
    answer -- four more attempts would cost 70 seconds per dispatch to relearn the same fact.
    """
    delay = 5
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        try:
            git("push", remote, refspec, cwd=repo, env=git_env)
            return True
        except DispatchError as exc:
            retryable = any(m in str(exc) for m in PUSH_RETRY_MESSAGES)
            if retryable and attempt == 1 and scopes is not None:
                held = scopes()
                # None means GitHub did not report scopes, which a fine-grained or App token never
                # does. Unknown is not missing, so those keep their retries.
                if held is not None and "workflow" not in held:
                    retryable = False
                    exc = DispatchError(
                        f"{exc}\n\nthe push credential holds only: {held}\n"
                        "Without `workflow` scope GitHub must decide whether this push touches a "
                        "workflow file, and on this repo that decision times out -- every time, not "
                        "intermittently. Give the push token `workflow` scope alongside `repo`. "
                        "That is safe here: refuse_pipeline_edits already rejects any snapshot that "
                        "changes anything under .github/, so the scope is never exercised."
                    )
            if not retryable or attempt == PUSH_ATTEMPTS:
                if not required:
                    return False
                raise exc from None
            log(f"  push attempt {attempt}/{PUSH_ATTEMPTS} rejected by GitHub's workflow check; retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    return False


@contextmanager
def credentialed(work: Path, token_file: Path):
    """A git environment that can authenticate, without the token reaching a command line.

    Via `GIT_ASKPASS` rather than a URL with credentials in it, because git echoes remote URLs back
    in its error messages, and those messages end up in a log this agent can read.
    """
    askpass = work / "askpass.sh"
    askpass.write_text(f'#!/bin/sh\nexec cat "{token_file}"\n')
    askpass.chmod(0o700)
    try:
        yield {"GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0"}
    finally:
        askpass.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# Dispatch, wait, collect


def find_run(api: Api, head_sha: str, since: float) -> dict | None:
    """The run the push started, identified by the commit that started it.

    Exact rather than heuristic: the commit is unique to this call, so nothing else can match it,
    and there is no race with another port in flight. The query is by sha across all workflows and
    the filter is on `path`, rather than asking the by-filename endpoint, because that endpoint
    needs the workflow to be registered already and this one may never have run before.
    """
    listing = api.get_json(
        f"/repos/{api.repo}/actions/runs?head_sha={head_sha}&per_page=40&exclude_pull_requests=true"
    )
    for run in listing.get("workflow_runs", []):
        if (run.get("path") or "").endswith(f"/{WORKFLOW_FILE}"):
            return run
    if time.monotonic() - since > RUN_APPEARS_TIMEOUT:
        raise ApiError(
            f"pushed {head_sha[:12]} but no {WORKFLOW_FILE} run appeared for it within "
            f"{RUN_APPEARS_TIMEOUT // 60} minutes. Either the branch does not match the workflow's "
            f"`port-op-scratch/**` filter, or the push was made with a token whose pushes GitHub "
            f"does not let start workflows -- PORT_PUSH_TOKEN must be a PAT, not GITHUB_TOKEN."
        )
    return None


def wait_for_completion(api: Api, run: dict, label: str, budget: float | None = None, age: float = 0.0) -> dict | None:
    """Poll until the run finishes, or until `budget` seconds of *this call* have gone by.

    Returns None on running out of budget, which is not a failure: the caller hands the handle back
    and is called again. `age` is how long the run has already been going across earlier calls, and
    exists only so the heartbeat and the give-up deadline talk about the run's life rather than this
    slice of it.
    """
    started = time.monotonic()
    last_beat = 0.0
    while True:
        run = api.get_json(f"/repos/{api.repo}/actions/runs/{run['id']}")
        if run.get("status") == "completed":
            return run

        elapsed = time.monotonic() - started
        if age + elapsed > RUN_COMPLETE_TIMEOUT:
            raise ApiError(f"{label} run {run['html_url']} has not finished after {(age + elapsed) / 60:.0f} minutes")
        if budget is not None and elapsed + POLL_SECONDS > budget:
            return None
        if elapsed - last_beat >= HEARTBEAT_SECONDS:
            last_beat = elapsed
            log(f"  [{(age + elapsed) / 60:5.1f}m] {label}: {run.get('status')} -- {_job_summary(api, run['id'])}")
        time.sleep(POLL_SECONDS)


def _job_summary(api: Api, run_id: int) -> str:
    try:
        jobs = api.get_json(f"/repos/{api.repo}/actions/runs/{run_id}/jobs?per_page=50").get("jobs", [])
    except ApiError:
        return "job list unavailable"
    active = [j for j in jobs if j.get("status") != "completed"]
    if active:
        return ", ".join(f"{j['name']} {j['status']}" for j in active[:3])
    return ", ".join(f"{j['name']} {j.get('conclusion')}" for j in jobs[-3:]) or "no jobs yet"


def fetch_results(api: Api, run_id: int, into: Path) -> Path | None:
    """Unpack the results artifact, if the device job got far enough to upload one."""
    listing = api.get_json(f"/repos/{api.repo}/actions/runs/{run_id}/artifacts?per_page=50")
    artifact = next(
        (a for a in listing.get("artifacts", []) if a["name"].startswith(RESULTS_ARTIFACT_PREFIX)),
        None,
    )
    if artifact is None:
        return None
    blob = api.download(f"/repos/{api.repo}/actions/artifacts/{artifact['id']}/zip")
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(into)
    return into


def failure_output(api: Api, run_id: int, tail: int = 120) -> str:
    """What the failing job actually said, which for a `build` is the compiler.

    Error lines first and the tail after: ninja prints the diagnostic and then keeps going through
    the remaining targets, so the last hundred lines of a failed build are often the least
    informative part of it.
    """
    jobs = api.get_json(f"/repos/{api.repo}/actions/runs/{run_id}/jobs?per_page=50").get("jobs", [])
    failed = [j for j in jobs if j.get("conclusion") not in (None, "success", "skipped")]
    if not failed:
        return "the run failed but no job reported a failure; see the run page"

    chunks = []
    for job in failed[:2]:
        try:
            text = api.download(f"/repos/{api.repo}/actions/jobs/{job['id']}/logs").decode("utf-8", "replace")
        except ApiError as exc:
            chunks.append(f"### {job['name']}: could not read the log ({exc})")
            continue
        lines = text.splitlines()
        errors = [ln for ln in lines if "error:" in ln.lower() or ln.startswith("FAILED:")]
        body = "\n".join(errors[:60]) if errors else ""
        chunks.append(
            f"### {job['name']} ({job.get('conclusion')})\n"
            + (f"{body}\n\n--- last {tail} lines ---\n" if body else "")
            + "\n".join(lines[-tail:])
        )
    return "\n\n".join(chunks)


def adopt_workspace(results: Path, repo: Path) -> list[str]:
    """Lay the `workspace/` half of the artifact over the checkout.

    The baseline dispatch generates the routing test, because rendering it needs ttnn and a card.
    It is part of the deliverable and has to exist in the tree the agent commits, so it comes home
    at its repo-relative path and is copied into place rather than described.
    """
    source = results / "workspace"
    if not source.is_dir():
        return []
    adopted = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            adopted.append(str(relative))
    return adopted


# ---------------------------------------------------------------------------------------------


def report_build(run: dict, api: Api) -> int:
    if run.get("conclusion") == "success":
        print("BUILD PASSED -- the tree compiles and the wheel was produced.")
        return 0
    # Said this plainly because the agent has twice now read a delivered answer as a broken tool and
    # retried it unchanged. A compile is deterministic: the same tree gives the same errors.
    print(
        f"BUILD FAILED -- the compiler rejected your code. The tool worked; the diagnostics below "
        f"are the answer, and rebuilding without editing will reproduce them exactly.\n\n"
        f"Edit the files named below, then `build` again. Do not call `verify` yet: a verify builds "
        f"before it measures, so on this tree it would fail in the same place after queueing for a "
        f"card.\n\nRun: {run['html_url']}\n"
    )
    print(failure_output(api, run["id"]))
    return 1


def report_verify(run: dict, api: Api, results: Path | None) -> int:
    gate = (results / "gate.json") if results else None
    if gate is None or not gate.is_file() or not gate.read_text().strip():
        print(
            f"NO VERDICT -- the measurement ran but produced no gate.json, which usually means it "
            f"could not get as far as measuring. This is not a verdict on your port and must not be "
            f"reported as one.\n\nRun: {run['html_url']}\n"
        )
        print(failure_output(api, run["id"]))
        return 2

    body = gate.read_text()
    # Named for what it is, because the agent's next move is decided by reading the verdict inside
    # rather than by anything about how the call ended.
    print("VERDICT DELIVERED -- the measurement ran on hardware and this is its result.\n")
    print(body)
    try:
        verdict = json.loads(body).get("verdict")
    except json.JSONDecodeError:
        print(f"\nthe verdict is not parseable JSON. {run['html_url']}")
        return 2
    # Mirrors gate.py's own exit codes, which the in-cluster pipeline propagated directly: 0 only
    # for a win, 2 for a tree the gate refused to measure at all.
    if verdict == VERDICT_WIN:
        return 0
    return 2 if verdict == "blocked" else 1


def report_baseline(run: dict, api: Api, results: Path | None, repo: Path) -> int:
    if run.get("conclusion") != "success" or results is None:
        print(f"the baseline dispatch failed ({run.get('conclusion')}). {run['html_url']}\n")
        print(failure_output(api, run["id"]))
        return 1
    adopted = adopt_workspace(results, repo)
    print(f"baseline OK. {run['html_url']}")
    for name in adopted:
        print(f"  brought back: {name}")
    summary = results / "baseline.json"
    if summary.is_file():
        print(f"\n{summary.read_text()}")

    # On a resumed branch, what the port already there currently fails. This is the work list, and it
    # is printed into the baseline step's log rather than left in an artifact because the agent reads
    # the log: a run that has to be told where to start should be told before it starts guessing.
    incoming = results / "incoming.json"
    if incoming.is_file():
        print(
            "\nThis branch already carries a port. Measured on the card just now, before you were "
            "started, here is what it currently fails -- this is your work list, and the cases under "
            "`prototype_gaps` are ones the generator itself cannot serve, so they are excused and not "
            f"yours to fix:\n\n{incoming.read_text()}"
        )
    return 0


# ---------------------------------------------------------------------------------------------
# Job records: the whole of what `--start` and `--wait` share
#
# The two halves are separate processes, minutes apart, so what one learned has to be written down.
# `$HOME/.port-dispatch/jobs` is the right place for the same reason the token lives next to it:
# it is in none of the paths the agent's sandbox mounts, so a handle cannot be forged into pointing
# at someone else's run.


def jobs_dir(work: Path) -> Path:
    directory = work / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_job(work: Path, handle: str, record: dict) -> None:
    (jobs_dir(work) / f"{handle}.json").write_text(json.dumps(record, indent=2))


def load_job(work: Path, handle: str) -> dict:
    path = jobs_dir(work) / f"{handle}.json"
    if not path.is_file():
        known = sorted(p.stem for p in jobs_dir(work).glob("*.json"))
        # A Refusal, not a failure: the usual cause is waiting on a handle that was already
        # collected, or one that a later dispatch superseded, and in both cases the agent has
        # something sensible to do next.
        raise Refusal(
            f"there is no dispatch called {handle!r}, so there is nothing to wait for. "
            + (
                f"Still in flight: {', '.join(known)} -- wait on one of those."
                if known
                else "Nothing is in flight. Either you already collected this one, or starting a "
                "newer dispatch superseded it; start a fresh build or verify."
            )
        )
    return json.loads(path.read_text())


def forget_job(work: Path, handle: str) -> None:
    (jobs_dir(work) / f"{handle}.json").unlink(missing_ok=True)


def retire_inflight(api: Api, work: Path, repo: Path, remote: str, token_file: Path) -> None:
    """Cancel whatever an earlier call left running before starting another.

    One card, one dispatch. The first real agentic run ended with four builds on CIv2 at once: the
    MCP gateway cancelled `build` at sixty seconds, the agent reasonably retried, and each retry had
    already pushed. Nothing on the far side knew the near side had stopped reading. A dispatch that
    supersedes an older one should end it, rather than race it for a card.
    """
    for record_path in sorted(jobs_dir(work).glob("*.json")):
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            record_path.unlink(missing_ok=True)
            continue
        run_id, branch = record.get("run_id"), record.get("branch")
        try:
            if run_id and api.get_json(f"/repos/{api.repo}/actions/runs/{run_id}").get("status") != "completed":
                log(f"  superseding {record.get('handle')}: cancelling run {run_id}")
                # stdout as well, because the agent is holding that handle and would otherwise wait
                # on a run this call just killed.
                print(
                    f"NOTE: cancelled the {record.get('mode')} still running under handle "
                    f"{record.get('handle')}, because starting this one supersedes it. Do not wait "
                    f"on that handle; it is gone."
                )
                api.post(f"/repos/{api.repo}/actions/runs/{run_id}/cancel")
        except ApiError as exc:
            log(f"  could not cancel the earlier run {run_id}: {exc}")
        if branch:
            with credentialed(work, token_file) as git_env:
                push_ref(repo, remote, f":refs/heads/{branch}", git_env, required=False)
        record_path.unlink(missing_ok=True)


BUILD_STATE = "last-build.json"


def _build_state(work: Path) -> dict:
    try:
        return json.loads((work / BUILD_STATE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def record_build_outcome(work: Path, tree: str, ok: bool) -> None:
    """Remember how the last build of this exact tree turned out, for the guard below."""
    if tree:
        (work / BUILD_STATE).write_text(json.dumps({"tree": tree, "outcome": "passed" if ok else "failed"}))


def refuse_pointless_dispatch(work: Path, mode: str, tree: str) -> None:
    """Refuse dispatches whose answer is already known, before they cost a card.

    Scaffolding rather than a plea in the prompt, because the prompt already asks for both of these
    and both happened anyway. Compilation is deterministic: a given tree compiles the same way every
    time, so there are exactly two dispatches that cannot teach anything.

      - Building a tree that has already been built. Seen 2026-08-12: the agent re-dispatched a
        byte-identical tree twenty seconds after a build failed.
      - Verifying a tree whose build failed. Seen the same afternoon, and the more expensive of the
        two: `verify` builds before it measures, so it fails in the same place having queued for a
        card first.

    Re-verifying a tree that *built* is deliberately still allowed. Measurement is noisy in a way
    compilation is not, and wanting a second opinion on a number is legitimate.
    """
    state = _build_state(work)
    if not tree or state.get("tree") != tree:
        if mode == "build":
            (work / BUILD_STATE).write_text(json.dumps({"tree": tree, "outcome": "dispatched"}))
        return

    if mode == "build":
        raise Refusal(
            "this is the same tree as the last build, byte for byte, so it would fail in exactly "
            "the same way -- a compile is deterministic. Nothing was dispatched and no time was "
            "spent. Read the diagnostics from the previous build, edit the files they name, and "
            "call `build` again once the tree differs."
        )
    if mode == "verify" and state.get("outcome") == "failed":
        raise Refusal(
            "this exact tree failed to build, and `verify` builds before it measures -- so it would "
            "fail in the same place, having queued for a card first. Nothing was dispatched. Fix "
            "the compile errors the last build reported, get a `build` to pass, and verify then. "
            "A verify cannot measure code that does not compile."
        )


def cleanup_ref(work: Path, repo: Path, remote: str, token_file: Path, branch: str) -> None:
    # Deleting can meet the same check as creating, and a ref left behind is litter rather than a
    # failure -- the agent job sweeps the namespace at the end. So retry, but never raise.
    with credentialed(work, token_file) as git_env:
        if not push_ref(repo, remote, f":refs/heads/{branch}", git_env, required=False):
            log(f"  could not delete {branch}; the post-run sweep will get it")


TRAILER_PREFIX = "Port-"


def read_verdict(work: Path) -> dict:
    """The last verdict this run reached, from the report the launcher already brought home.

    Read off disk rather than passed in, because publish is a post-step: the agent has ended, and
    whatever it claimed in its own summary is not evidence. Every `--wait` unpacks the results
    artifact into `results-<handle>/` beside the credential, so the newest `gate.json` there is the
    last thing `gate.py` actually decided on a card. Missing is a real answer too -- a run that never
    got a verdict publishes `none`, which is exactly what the next attempt should see.
    """
    candidates = [p for p in work.glob("results-*/gate.json") if p.is_file() and p.stat().st_size]
    if not candidates:
        return {}
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(newest.read_text())
    except json.JSONDecodeError:
        return {}


def publish(api: Api, repo: Path, work: Path, remote: str, token_file: Path, args) -> int:
    """Commit whatever the agent left and fast-forward the branch it was working on.

    A post-step rather than a tool, which is the whole design. Run 31406186048 produced a correct
    port -- 184 of 184 cases bit-exact -- and threw it away, because the verdict never reached `win`,
    the agent correctly declined to open a PR, and the C++ existed nowhere but that workspace. So the
    push does not depend on the verdict, or on the agent choosing to do it, or on the agent having a
    token: it happens because the job is ending.

    The commit message is the entire chain state. A resumed run reads these trailers off the branch
    tip and needs nothing else -- no run id, no report artifact, no attempt count from a human. That
    is what makes a branch self-describing, and what makes a branch someone wrote by hand resumable on
    exactly the same terms as one this pipeline wrote.

    `Port-generator` is load-bearing rather than informational: the workflow checks the generator out
    at whatever commit it names, so the next attempt transliterates the same generator this one did.
    It holds a full sha because `--codegen-ref` is resolved to one before any dispatch, and the pin
    step ignores anything that is not, which is how a branch published by an older harness -- whose
    trailer holds the branch *name* -- correctly reads as unpinned.
    """
    if not args.branch:
        log("publish: no branch to publish to; nothing to do")
        return 0

    report_data = read_verdict(work)
    verdict = report_data.get("verdict") or "none"
    correctness = report_data.get("correctness") or {}
    failing = correctness.get("failure_count")

    subject = f"{args.op}: port to codegen, attempt {args.attempt} ({verdict})"
    trailers = [
        f"{TRAILER_PREFIX}verdict: {verdict}",
        f"{TRAILER_PREFIX}attempt: {args.attempt}",
        f"{TRAILER_PREFIX}failing: {failing if failing is not None else -1}",
        f"{TRAILER_PREFIX}generator: {args.codegen_ref}",
    ]
    body = (
        "Written by the port-op workflow, not by hand. The trailers below are the chain state a "
        "resumed run reads off this branch; `Port-failing: -1` means no correctness band was graded.\n\n"
        + "\n".join(trailers)
    )
    commit, base = commit_worktree(repo, work, f"{subject}\n\n{body}")

    if commit == base or git("rev-parse", f"{commit}^{{tree}}", cwd=repo) == git(
        "rev-parse", f"{base}^{{tree}}", cwd=repo
    ):
        # Nothing was written. Publishing an empty commit would defeat the no-progress stop, which
        # asks precisely whether the tree moved.
        log("publish: the worktree is identical to the base commit; nothing to publish")
        print(json.dumps({"published": False, "reason": "no changes", "verdict": verdict}))
        return 0

    # The same guard the dispatch path uses, for the same reason and then one more: this commit lands
    # on a long-lived branch that a person will open a pull request from, so an edit to the pipeline
    # here would be a proposal to change the pipeline, wearing the port's clothes.
    refuse_pipeline_edits(repo, base, commit)

    with credentialed(work, token_file) as git_env:
        # Not forced. A fast-forward failure means something else moved the branch, and overwriting
        # it would be destroying work this run never saw.
        push_ref(repo, remote, f"{commit}:refs/heads/{args.branch}", git_env)

    log(f"publish: {commit[:12]} -> {args.branch} ({verdict}, failing={failing})")
    again, why = should_chain(verdict, failing, args)
    log(f"publish: {'chaining' if again else 'stopping'} -- {why}")
    if again:
        # Re-dispatched by workflow_dispatch rather than by pushing a trigger branch, because the run
        # this starts is a full port-op run with its own agent, and it must read the branch that was
        # just published rather than a scratch ref.
        try:
            api.post_json(
                f"/repos/{api.repo}/actions/workflows/{PORT_OP_WORKFLOW}/dispatches",
                {"ref": args.branch, "inputs": {"op": args.op, "branch": args.branch}},
            )
            log(f"publish: dispatched attempt {args.attempt + 1} on {args.branch}")
        except DispatchError as exc:
            # Never fatal. The work is pushed, which was the point; a chain that fails to start is a
            # run someone can start by hand, and losing the commit would not be.
            log(f"publish: could not start the next attempt ({exc}); the branch is pushed regardless")
            again, why = False, f"re-dispatch failed: {exc}"
    print(
        json.dumps(
            {
                "published": True,
                "commit": commit,
                "branch": args.branch,
                "verdict": verdict,
                "failing": failing,
                "attempt": args.attempt,
                "chain": again,
                "chain_reason": why,
            }
        )
    )
    return 0


def should_chain(verdict: str, failing: int | None, args) -> tuple[bool, str]:
    """Whether to spend another run on this branch.

    Run 3 of this workflow looped: eleven verifies, every one `blocked` by a harness defect the agent
    could not see or fix, until the job hit its ceiling. Both guards below exist because of it, and
    they answer different questions -- the cap bounds the cost of any loop at all, and the
    no-progress stop catches the loop that is cheap per attempt and still getting nowhere.

    Progress is the failing count falling. Not the verdict improving, which can happen for reasons
    that have nothing to do with the port, and not the tree changing, which an agent can do all day.
    A count that is equal or higher than the previous attempt's means the last run of the machinery
    bought nothing, and the next one has no more information than it did.
    """
    if verdict == VERDICT_WIN:
        return False, "the port won; there is nothing left to chain for"
    if args.attempt >= args.max_attempts:
        return False, f"attempt {args.attempt} of a {args.max_attempts}-attempt cap"
    if verdict == "blocked":
        # Blocked is never the port's fault -- it means the gate refused to measure, which is a
        # harness or tree problem no further attempt will resolve by trying harder.
        return False, "the gate refused to measure this tree; another attempt would refuse identically"
    if failing is None or failing < 0:
        # No correctness band was graded, so there is no progress signal at all. One more attempt is
        # allowed rather than none, because this is what the first attempt on a branch looks like.
        return args.prev_failing < 0, "no graded correctness band to compare against"
    if args.prev_failing >= 0 and failing >= args.prev_failing:
        return False, f"no progress: {args.prev_failing} failing before, {failing} now"
    return True, f"failing fell from {args.prev_failing if args.prev_failing >= 0 else 'unmeasured'} to {failing}"


def report(mode: str, run: dict, api: Api, results: Path | None, repo: Path) -> int:
    if mode == "build":
        return report_build(run, api)
    if mode == "verify":
        return report_verify(run, api, results)
    return report_baseline(run, api, results, repo)


def delivered(code: int, as_tool: bool) -> int:
    """A tool that answered did its job, whatever the answer was.

    gh-aw's mcp-scripts runner hands the handler to `execFile` and rejects the promise on any
    non-zero exit, so what reaches the agent is `Command failed: build.sh (exit code: 1)` with the
    real output demoted beneath it. The agent then does the only sensible thing with a broken tool,
    which is to call it again -- and that is not what you want it to do with a compiler diagnostic.
    Both of the first two agentic runs died this way: one retrying a 60s gateway timeout, one
    retrying a perfectly good build failure and a `wait` that had merely said "not finished yet".

    So under `--as-tool` the exit code carries nothing and the text carries everything. The workflow
    steps, which genuinely do consume exit codes, do not pass the flag and are unaffected.
    """
    return 0 if as_tool else code


def main() -> int:
    ap = argparse.ArgumentParser(description="Dispatch a build or a measurement onto CIv2 and wait for it.")
    ap.add_argument("--mode", choices=["baseline", "build", "verify", "publish"])
    ap.add_argument(
        "--start",
        action="store_true",
        help="push and return a handle instead of waiting, for callers under a tool-call deadline",
    )
    ap.add_argument("--wait", metavar="HANDLE", help="resume waiting on a dispatch started earlier")
    ap.add_argument("--budget", type=int, default=WAIT_BUDGET, help="seconds a single --wait may block")
    ap.add_argument(
        "--as-tool",
        action="store_true",
        help="exit 0 whenever an answer was delivered; see delivered() for why this is not optional",
    )
    ap.add_argument("--band", default="both", choices=["correctness", "performance", "both"])
    ap.add_argument("--op", default=os.environ.get("PORT_OP", "untilize"))
    ap.add_argument("--category", default=os.environ.get("PORT_CATEGORY", "data_movement"))
    ap.add_argument("--codegen-ref", default=os.environ.get("PORT_CODEGEN_REF", "codegen_agentic_port"))
    ap.add_argument("--perf-limit", default=os.environ.get("PORT_LIMIT", "24"))
    ap.add_argument(
        "--branch",
        default=os.environ.get("PORT_BRANCH", ""),
        help="the long-lived branch a run continues and publishes back to. Empty means a one-shot run "
        "whose work is not chained, and `--mode publish` then does nothing.",
    )
    ap.add_argument(
        "--attempt",
        type=int,
        default=int(os.environ.get("PORT_ATTEMPT", "1")),
        help="which attempt on this branch, resolved from the tip's commit trailer by the workflow",
    )
    ap.add_argument(
        "--prev-failing",
        type=int,
        default=int(os.environ.get("PORT_PREV_FAILING", "-1")),
        help="failing count the previous attempt recorded, -1 when there was none",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("PORT_MAX_ATTEMPTS", "6")),
        help="hard cap on chained attempts, counting this one",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("PORT_RESUME") == "1",
        help="the tree already carries a port, so the baseline measures what it currently fails. The "
        "workflow's resolve step sets PORT_RESUME by looking for a tracked codegen directory.",
    )
    ap.add_argument(
        "--runner-label",
        default=os.environ.get("PORT_RUNNER_LABEL", '["tt-ubuntu-2204-N150-viommu-stable"]'),
    )
    ap.add_argument("--repo-path", default=os.environ.get("GITHUB_WORKSPACE", "."))
    ap.add_argument(
        "--token-file",
        default=os.environ.get("PORT_DISPATCH_TOKEN_FILE", str(Path.home() / ".port-dispatch" / "token")),
        help="written by a pre-step, outside every path the agent sandbox mounts",
    )
    args = ap.parse_args()

    token_file = Path(args.token_file)
    if not token_file.is_file():
        raise DispatchError(f"no dispatch credential at {token_file}; the pre-step that writes it did not run")
    token = token_file.read_text().strip()
    if not token:
        raise DispatchError(f"the dispatch credential at {token_file} is empty")

    repo_slug = os.environ.get("GITHUB_REPOSITORY")
    if not repo_slug:
        raise DispatchError("GITHUB_REPOSITORY is unset; this only runs inside a workflow job")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    api = Api(token, repo_slug, os.environ.get("GITHUB_API_URL", "https://api.github.com"))

    repo = Path(args.repo_path).resolve()
    work = token_file.parent
    work.mkdir(parents=True, exist_ok=True)
    remote_url = f"{server}/{repo_slug}.git"

    if args.wait:
        return resume(api, args, work, repo, remote_url, token_file)
    if not args.mode:
        raise DispatchError("say what to run with --mode, or which dispatch to resume with --wait")
    if args.mode == "publish":
        # Nothing is dispatched and nothing is waited on: this pushes what is already here.
        return publish(api, repo, work, remote_url, token_file, args)

    nonce = f"{os.environ.get('GITHUB_RUN_ID', 'local')}-{uuid.uuid4().hex[:8]}"
    branch = f"{SCRATCH_PREFIX}/{args.op}-{args.mode}-{nonce}"
    remote = remote_url

    # The workflow reads these back out of the commit message, so the message is the payload and
    # nothing else. The nonce is in it to guarantee two identical dispatches cannot hash to the
    # same commit, which is what makes the sha a usable handle on the run.
    base = git("rev-parse", "HEAD", cwd=repo)
    params = {
        "mode": args.mode,
        "band": args.band,
        "op": args.op,
        "category": args.category,
        "codegen-ref": args.codegen_ref,
        "perf-limit": str(args.perf_limit),
        "base-sha": base,
        "runner-label": args.runner_label,
        # Only the baseline reads this, and only to decide whether there is a port here worth
        # measuring. A fresh run's tree holds nothing but stubs, and a correctness band over stubs is
        # 112 failures that say nothing.
        "resume": "1" if args.resume else "0",
        "nonce": nonce,
    }

    commit, parent = commit_worktree(repo, work, json.dumps(params, separators=(",", ":")))
    if parent != base:
        # The write-path guard diffs the checkout against the base-sha in the message. If that is
        # not the commit's own parent, it grades against the wrong tree and quietly reports the
        # difference as the agent writing where it should not have.
        raise DispatchError(f"HEAD moved from {base[:12]} to {parent[:12]} while snapshotting")
    refuse_pipeline_edits(repo, base, commit)
    tree = git("rev-parse", f"{commit}^{{tree}}", cwd=repo)
    refuse_pointless_dispatch(work, args.mode, tree)
    retire_inflight(api, work, repo, remote, token_file)
    log(f"{args.mode} for {args.op}: pushing {commit[:12]} (base {base[:12]}) to {branch}")

    with credentialed(work, token_file) as git_env:
        push_ref(repo, remote, f"{commit}:refs/heads/{branch}", git_env, scopes=api.token_scopes)

    # From here the run exists on CIv2 whether or not this process survives, so a failure to find it
    # has to take the ref down with it rather than leave a run nobody is holding a handle to.
    try:
        started = time.monotonic()
        run = None
        while run is None:
            time.sleep(POLL_SECONDS)
            run = find_run(api, commit, started)
        log(f"  run {run['id']}: {run['html_url']}")
    except BaseException:
        cleanup_ref(work, repo, remote, token_file, branch)
        raise

    if args.start:
        save_job(
            work,
            nonce,
            {
                "handle": nonce,
                "mode": args.mode,
                "op": args.op,
                "branch": branch,
                "commit": commit,
                "tree": tree,
                "run_id": run["id"],
                "run_url": run["html_url"],
                "started": time.time(),
            },
        )
        print(
            f"{args.mode} started on CIv2 as {run['html_url']}\n"
            f"handle: {nonce}\n"
            f"It is running now and nothing further happens until you collect it. Call `wait` with "
            f"that handle; if `wait` reports it is still going, call `wait` again with the same handle."
        )
        return 0

    try:
        run = wait_for_completion(api, run, args.mode, age=0.0)
        log(f"  {args.mode} finished as {run.get('conclusion')} after {(time.monotonic() - started) / 60:.1f} minutes")
        results = fetch_results(api, run["id"], work / f"results-{nonce}")
    finally:
        cleanup_ref(work, repo, remote, token_file, branch)

    return delivered(report(args.mode, run, api, results, repo), args.as_tool)


def resume(api: Api, args, work: Path, repo: Path, remote: str, token_file: Path) -> int:
    """Wait a while longer on a dispatch someone else started, and collect it if it has landed.

    Returns 4 -- distinct from every verdict -- when the run is simply still going. That is not a
    result and must not read like one: the caller's next move is to call again, not to conclude
    anything about the port.
    """
    record = load_job(work, args.wait)
    age = max(0.0, time.time() - record.get("started", time.time()))
    run = api.get_json(f"/repos/{api.repo}/actions/runs/{record['run_id']}")

    if run.get("status") != "completed":
        run = wait_for_completion(api, run, record["mode"], budget=args.budget, age=age)
        if run is None:
            elapsed = (time.time() - record.get("started", time.time())) / 60
            print(
                f"STILL RUNNING -- this is not a result and not a failure.\n\n"
                f"The {record['mode']} has been going for {elapsed:.0f} minutes and has not finished. "
                f"{record['run_url']}\n\n"
                f"Do exactly one thing: call `wait` again with handle {args.wait}. Do not edit code, "
                f"do not start another {record['mode']}, and do not draw any conclusion about your "
                f"port -- nothing has been measured yet. Starting another one cancels this one."
            )
            return delivered(4, args.as_tool)

    log(f"  {record['mode']} finished as {run.get('conclusion')} after {(time.time() - record.get('started', 0)) / 60:.1f} minutes")
    try:
        results = fetch_results(api, run["id"], work / f"results-{args.wait}")
    finally:
        cleanup_ref(work, repo, remote, token_file, record["branch"])
        forget_job(work, args.wait)

    if record["mode"] == "build":
        record_build_outcome(work, record.get("tree", ""), run.get("conclusion") == "success")

    return delivered(report(record["mode"], run, api, results, repo), args.as_tool)


if __name__ == "__main__":
    # Read before argparse, because how the two failure classes below are surfaced depends on it.
    AS_TOOL = "--as-tool" in sys.argv
    try:
        sys.exit(main())
    except Refusal as refusal:
        # An answer, so it exits like one. The launcher did exactly what it meant to do and the
        # message says what to change; framing that as a crashed tool invites a blind retry.
        print(f"REFUSED -- nothing was dispatched, and no time was spent.\n\n{refusal}")
        sys.exit(delivered(1, AS_TOOL))
    except DispatchError as error:
        # stdout, not stderr: this is the only thing the agent will see, and it needs to be able to
        # tell "the harness could not run" apart from "your port is wrong". This one really is the
        # harness failing, so it keeps its non-zero exit and reaches the agent as a tool error.
        print(f"the dispatch could not complete: {error}")
        sys.exit(3)
