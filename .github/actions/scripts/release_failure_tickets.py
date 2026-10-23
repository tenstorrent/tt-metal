#!/usr/bin/env python3
"""File one RELEASE Jira ticket per failed job of a Package and release run.

Reads the run and its jobs from the GitHub API, drops the failures that carry
no test result (a runner that never started; the nothing-to-release precheck on
main), and files -- or de-duplicates onto -- one ticket per remaining job. Each
ticket names the job, its owner from tests/pipeline_reorg, and the root cause
the run's AI Summary recorded for it.

A ticket is keyed on (release ref, job): the same job failing again on the
same ref comments on its open ticket instead of opening another, and the
close-jira job in the workflow closes every ticket for the ref when a run on
it goes green.

Environment:
  REPO / RUN_ID                     owner/repo and the Package and release run id  (required)
  RUN_URL                           link to the run (default: derived from REPO/RUN_ID)
  JIRA_BASE_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN / JIRA_PROJECT_KEY
  JIRA_ISSUE_TYPE                   default: Bug
  JIRA_ASSIGNEE_ACCOUNT_ID          accountId every ticket is assigned to (optional)
  JIRA_DRY_RUN                      print the payloads instead of calling Jira
Needs GH_TOKEN with actions:read for the gh calls.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

from jira_client import _commit_link, _env, _truthy, file_issue  # noqa: E402
from job_owners import lookup  # noqa: E402

JIRA_SUMMARY_MAX = 255

# A job whose runner never came up ran no test: its work steps are skipped and
# only the always() steps run, so it fails having tested nothing.
_INFRA_STEP = re.compile(r"^\s*(set up job|set up runner|initialize containers)\s*$", re.I)


def is_release_ref(branch):
    """Only main (nightly dev release), stable and the version tags reach the board."""
    return branch in ("main", "stable") or re.match(r"^v[0-9]", branch or "") is not None


def leaf(job_name):
    """The last segment of "a / b / c / leaf"."""
    return job_name.split(" / ")[-1].strip()


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _gh(args):
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def fetch_run(repo, run_id):
    return json.loads(_gh([f"repos/{repo}/actions/runs/{run_id}"]))


def fetch_jobs(repo, run_id):
    """Every job of the run, as dicts with name / conclusion / html_url / steps."""
    out = _gh(["--paginate", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", "--jq", ".jobs[]"])
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _runner_never_started(job):
    return any(
        s.get("conclusion") in ("failure", "timed_out") and _INFRA_STEP.match(s.get("name") or "")
        for s in job.get("steps") or []
    )


def select_failed(jobs, branch):
    """(jobs to file, number dropped). Drops failures that carry no test result."""
    kept, dropped = [], 0
    for job in jobs:
        if job.get("conclusion") not in ("failure", "timed_out"):
            continue
        if _runner_never_started(job):
            print(f"::notice::skipping {job['name']} -- runner never started, no test ran")
            dropped += 1
            continue
        # release-precheck exits 1 on main to mean "nothing to release" (HEAD is
        # already tagged); everything downstream is skipped, so nothing was
        # tested. On stable and the tags a precheck failure is a real blocker.
        if branch == "main" and leaf(job["name"]) == "release-precheck":
            print(f"::notice::skipping {job['name']} -- nothing-to-release exit, no test ran")
            dropped += 1
            continue
        kept.append(job)
    return kept, dropped


def root_cause_for(job, summary):
    """The AI Summary row for this job, or None. Matched on job URL, then leaf name."""
    rows = list((summary or {}).get("failed") or []) + list((summary or {}).get("infra_failure") or [])
    for row in rows:
        if row.get("job_url") and row["job_url"] == job.get("html_url"):
            return row
    for row in rows:
        if leaf(row.get("job_name") or "") == leaf(job["name"]):
            return row
    return None


def _one_line(text, limit=300):
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def ticket(job, branch, sha, repo, run_id, run_url, summary=None):
    """(summary, description, labels, dedup_label) for one failed job.

    Section layout mirrors RELEASE-7, the hand-written reference for these
    tickets. "### " and "- " render as headings and bullets.
    """
    name = leaf(job["name"])
    title = f"Release {branch} — {name} failed"
    if len(title) > JIRA_SUMMARY_MAX:
        title = title[: JIRA_SUMMARY_MAX - 3] + "..."

    owner = lookup(name)
    job_line = f"- [{name}]({job['html_url']})"
    if owner:
        job_line += f" — owner: {owner['owner']} ({owner['team']})"

    lines = [
        "### Impact",
        f"{name} failed in Package and release on {branch}.",
        "### Failed job",
        job_line,
        "### Environment",
        f"- Branch: {branch}",
        f"- Commit: {_commit_link(sha, repo)}",
        f"- Run link: [{run_id}]({run_url})",
    ]
    parents = job["name"].split(" / ")[:-1]
    if parents:
        lines.append(f"- Pipeline: {' / '.join(parents)}")

    row = root_cause_for(job, summary)
    cause = row and (row.get("root_cause") or row.get("error_message"))
    if cause:
        note = f"- {_one_line(cause)}"
        tag = "/".join(t for t in (row.get("category"), row.get("subcategory")) if t)
        if tag:
            note += f" [{tag}]"
        # The verbatim error rides along only when it is not already the cause.
        if row.get("root_cause") and row.get("error_message"):
            note += f" — error: {_one_line(row['error_message'], 200)}"
        if str(row.get("log_complete")).lower() == "false":
            note += " (log incomplete)"
        lines += ["### Other information", "Root cause analysis from the run's AI Summary:", note]

    labels = ["ci-failure", "package-and-release", f"package-release-ref:{branch}"]
    dedup = f"package-release-job:{branch}:{slug(name)}"
    return title, "\n".join(lines) + "\n", labels, dedup


def _fetch_summary(repo, run_id):
    """The run's AI Summary JSON, or None. Decoration: never blocks filing."""
    try:
        from ci_digest import fetch_run_summary

        return fetch_run_summary(repo, int(run_id))
    except Exception as e:  # noqa: BLE001 -- any failure here just drops the section
        print(f"::warning::AI Summary unavailable for run {run_id}: {e}")
        return None


def main():
    repo = _env("REPO", required=True)
    run_id = _env("RUN_ID", required=True)
    run_url = _env("RUN_URL", "") or f"https://github.com/{repo}/actions/runs/{run_id}"
    dry = _truthy(_env("JIRA_DRY_RUN"))

    run = fetch_run(repo, run_id)
    branch, sha = run.get("head_branch") or "", run.get("head_sha") or ""
    if not is_release_ref(branch):
        print(f"::notice::run {run_id} is on '{branch}', not a release ref; nothing to file.")
        return

    failed, dropped = select_failed(fetch_jobs(repo, run_id), branch)
    if not failed:
        if dropped:
            print(f"::notice::all {dropped} failed job(s) had no test result; nothing to file.")
        else:
            print("No jobs failed; nothing to file.")
        return

    email, token = _env("JIRA_USER_EMAIL", ""), _env("JIRA_API_TOKEN", "")
    if not dry and not (email and token):
        print("::notice::JIRA_RELEASE_* secrets not configured; skipping Jira filing.")
        return

    summary = _fetch_summary(repo, run_id)
    filed = failed_to_file = 0
    for job in failed:
        title, description, labels, dedup = ticket(job, branch, sha, repo, run_id, run_url, summary)
        try:
            print(
                file_issue(
                    base=_env("JIRA_BASE_URL", required=True),
                    email=email,
                    token=token,
                    project=_env("JIRA_PROJECT_KEY", required=True),
                    summary=title,
                    issue_type=_env("JIRA_ISSUE_TYPE", "Bug"),
                    description=description,
                    labels=labels,
                    dedup_label=dedup,
                    assignee=_env("JIRA_ASSIGNEE_ACCOUNT_ID", "") or None,
                    dry_run=dry,
                )
            )
            filed += 1
        except (SystemExit, Exception) as e:  # jira_client exits on API error
            # One bad ticket must not drop the rest.
            print(f"::warning::could not file a ticket for {leaf(job['name'])}: {e}")
            failed_to_file += 1

    print(f"filed/updated {filed} issue(s); {dropped} failed job(s) had no test result")
    if failed_to_file:
        sys.exit(f"error: {failed_to_file} issue(s) could not be filed")


if __name__ == "__main__":
    main()
