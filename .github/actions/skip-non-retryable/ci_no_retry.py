#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Decide whether CI jobs marked `ci-no-retry` should be re-run. See ../classify-failure/README.md.

  job-check --job-id ID --run-id ID --attempt N [--triggering-actor LOGIN]
                                           run by a job on a re-run attempt: exit 3 if auto-retry started the
                                           re-run and this job's previous attempt was marked (it emits the
                                           marker again), else 0. A re-run started by anyone else runs the job.
  run-decision --run-id ID --attempt N     run by auto-retry: prints "skip" if every unsuccessful job of
                                           attempt N is marked, else "rerun"; also sets the step output
                                           all-non-retryable=true when it prints "skip"

Both fail open: any API problem means "run the job" / "re-run". Needs GH_TOKEN with actions:read and
checks:read. Locally: GH_TOKEN=$(gh auth token) python3 ci_no_retry.py run-decision --run-id <id> --attempt 1
"""

import argparse
import json
import os
import sys
import urllib.request

MARKER = "ci-no-retry"
SKIP_EXIT = 3
# Conclusions rerun-failed-jobs would not re-run; anything else (failure, timed_out, cancelled, ...) would be.
SETTLED = {"success", "skipped", "neutral"}
SHORT_CIRCUIT_PREFIX = " failed deterministically: "
SHORT_CIRCUIT_SUFFIX = " -- auto-retry cannot change that; re-run it by hand to measure again"
# Who auto-retry's re-runs are triggered as: the owner of RETRY_WORKFLOW_TOKEN in
# .github/workflows/_auto-retry-nightly-workflows.yaml. A re-run started by anyone else (by hand, to measure
# again) is never short-circuited. Known exception: the token owner is a person, so their own manual re-runs
# look like auto-retry and are short-circuited too (moving the token to a GitHub App would remove this).
# If the token changes owner, update this; the auto-retry workflow warns when they differ, and until then
# marked jobs simply re-run as they did before ci-no-retry existed.
AUTO_RETRY_ACTOR = "tt-rkim"


def api(url):
    if not url.startswith("http"):
        url = f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/{url.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def paged(url, key=None):
    out, page = [], 1
    sep = "&" if "?" in url else "?"
    while True:
        body = api(f"{url}{sep}per_page=100&page={page}")
        batch = body[key] if key else body
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def attempt_jobs(repo, run_id, attempt):
    return paged(f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")


def marker(job):
    """The job's ci-no-retry annotation message, or None."""
    # check_run_url rather than assuming the check-run id equals the job id.
    for a in paged(f"{job['check_run_url']}/annotations"):
        if a.get("title") == MARKER:
            return a.get("message") or ""
    return None


def escape(msg):
    return msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def job_check(args):
    repo, run_id, attempt = args.repo, args.run_id, args.attempt
    if attempt <= 1:
        return 0
    if args.triggering_actor.casefold() != args.auto_retry_actor.casefold():  # logins are case-insensitive
        print(f"re-run started by '{args.triggering_actor}', not by auto-retry ('{args.auto_retry_actor}'); running it")
        return 0
    name = api(f"repos/{repo}/actions/jobs/{args.job_id}")["name"]
    prev = [j for j in attempt_jobs(repo, run_id, attempt - 1) if j["name"] == name]
    if len(prev) != 1:
        print(f"found {len(prev)} jobs named '{name}' in attempt {attempt - 1}; running this one")
        return 0
    if prev[0]["conclusion"] in SETTLED:
        print(f"'{name}' did not fail in attempt {attempt - 1}; running it")
        return 0
    msg = marker(prev[0])
    if msg is None:
        print(f"attempt {attempt - 1} of '{name}' is not marked {MARKER}; running it")
        return 0
    # Strip what an earlier short-circuit added so the reason does not nest attempt after attempt.
    reason = msg.split(SHORT_CIRCUIT_PREFIX, 1)[-1].removesuffix(SHORT_CIRCUIT_SUFFIX)
    print(
        f"::error title={MARKER}::"
        + escape(f"Not re-run: attempt {attempt - 1}{SHORT_CIRCUIT_PREFIX}{reason}{SHORT_CIRCUIT_SUFFIX}")
    )
    return SKIP_EXIT


def run_decision(args):
    jobs = attempt_jobs(args.repo, args.run_id, args.attempt)
    unsuccessful = [j for j in jobs if j["conclusion"] not in SETTLED]
    unmarked = [j["name"] for j in unsuccessful if marker(j) is None]
    print(f"{len(unsuccessful) - len(unmarked)} of {len(unsuccessful)} unsuccessful job(s) are marked {MARKER}")
    for name in unmarked:
        print(f"  retryable: {name}")
    skip = bool(unsuccessful) and not unmarked
    print("skip" if skip else "rerun")
    if skip and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write("all-non-retryable=true\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    jc = sub.add_parser("job-check")
    jc.add_argument("--job-id", required=True)
    jc.add_argument("--triggering-actor", default=os.environ.get("GITHUB_TRIGGERING_ACTOR", ""))
    jc.add_argument("--auto-retry-actor", default=AUTO_RETRY_ACTOR)
    rd = sub.add_parser("run-decision")
    for p in (jc, rd):
        p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal"))
        p.add_argument("--run-id", required=True)
        p.add_argument("--attempt", required=True, type=int)
    args = parser.parse_args()
    try:
        return job_check(args) if args.cmd == "job-check" else run_decision(args)
    except Exception as exc:  # fail open: running / re-running is the behaviour without this script
        print(f"::warning title={MARKER}-check-failed::{args.cmd} failed ({type(exc).__name__}: {exc}); not skipping")
        if args.cmd == "run-decision":
            print("rerun")
        return 0


if __name__ == "__main__":
    sys.exit(main())
