#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-job pass/fail status for the last N scheduled runs on `main` of the
(Tier 1/2/3) Models e2e + unit pipelines. For failed jobs, fetches the
failure annotations (typically from gtest/pytest summary actions) so we
can report the actual error message rather than just "the job failed".

Output is markdown, written to stdout.

Usage:
    python3 tools/model_pipeline_ci_status.py [--limit N] > report.md
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

REPO = "tenstorrent/tt-metal"
BRANCH = "main"
DEFAULT_LIMIT = 10

# Each entry: (workflow_id, display_name, tier, pipeline_type)
WORKFLOWS = [
    ("235399384", "(Tier 1) Models e2e",  1, "e2e"),
    ("235399388", "(Tier 2) Models e2e",  2, "e2e"),
    ("235399383", "(Tier 3) Models e2e",  3, "e2e"),
    ("235399387", "(Tier 1) Models unit", 1, "unit"),
    ("235399386", "(Tier 2) Models unit", 2, "unit"),
    ("235399385", "(Tier 3) Models unit", 3, "unit"),
]


def gh_json(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"# gh failed: {' '.join(args)}\n{r.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def fetch_runs(workflow_id, limit, since=None):
    """Fetch the most recent scheduled runs on `main` for a workflow.

    `since`: optional timezone-aware datetime; runs older than this are dropped.
    `limit` is the cap; if `since` is set we may iterate to find enough.
    """
    runs = gh_json([
        "gh", "run", "list",
        "-R", REPO,
        "-w", workflow_id,
        "--event", "schedule",
        "-b", BRANCH,
        "--limit", str(limit),
        "--json", "databaseId,createdAt,conclusion,status,url,displayTitle,headSha",
    ]) or []
    if since is not None:
        runs = [r for r in runs if datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) >= since]
    return runs


def fetch_jobs(run_id):
    data = gh_json([
        "gh", "api", "--paginate",
        f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
    ])
    if data is None:
        return []
    if isinstance(data, dict):
        return data.get("jobs", [])
    jobs = []
    for page in data:
        jobs.extend(page.get("jobs", []))
    return jobs


def fetch_annotations(check_run_id):
    """Annotations on a check-run usually carry the actual failure message
    (e.g. gtest/pytest summary). check_run_id == job id for GHA jobs."""
    data = gh_json([
        "gh", "api",
        f"repos/{REPO}/check-runs/{check_run_id}/annotations",
    ])
    return data or []


def trim(s, n=300):
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + " …[truncated]"


# Drop known low-signal annotation messages so the report focuses on real
# failure causes. These are infrastructure noise that fires on every job.
NOISE_PATTERNS = [
    re.compile(r"Node\.js 20 actions are deprecated", re.I),
    re.compile(r"Docker network rm failed", re.I),
    re.compile(r"Docker rm fail", re.I),
    re.compile(r"Benchmark data directory .* does not exist", re.I),
    re.compile(r"No files were found with the provided path", re.I),
    re.compile(r"actions/upload-artifact.*deprecated", re.I),
    re.compile(r"set-output command is deprecated", re.I),
]


def is_noise(msg):
    return any(p.search(msg) for p in NOISE_PATTERNS)


def filter_annotations(anns):
    """Return failure-level annotations first, then non-noise warnings."""
    failures = [a for a in anns if a.get("annotation_level") == "failure"]
    warnings = [
        a for a in anns
        if a.get("annotation_level") != "failure" and not is_noise(a.get("message", ""))
    ]
    return failures + warnings


def first_failed_step(job):
    for s in job.get("steps", []):
        if s.get("conclusion") == "failure":
            return s.get("name")
    return None


def fmt_iso(ts):
    if not ts:
        return ""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")


def write_report(out, workflows, limit, since=None, header_window=""):
    out.write(f"# Models CI status — {header_window}\n\n")
    out.write(f"Generated {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.\n\n")
    out.write(
        "Filter: `event=schedule`, `branch=main`. Failure messages are sourced from\n"
        "GitHub check-run annotations (gtest/pytest summary actions). When no annotation\n"
        "is present, only the failed-step name is reported.\n\n"
    )

    overall = Counter()

    for wf_id, wf_name, _tier, _ptype in workflows:
        runs = fetch_runs(wf_id, limit, since=since)
        out.write(f"\n## {wf_name}\n\n")
        if not runs:
            out.write("_No scheduled runs found on `main`._\n")
            continue

        # Per-run summary table
        out.write("| Run date (UTC) | Conclusion | Test jobs (✓ / ✗ / -) | Run URL |\n")
        out.write("|---|---|---|---|\n")
        run_details = []
        for r in runs:
            jobs = fetch_jobs(r["databaseId"])
            test_jobs = [j for j in jobs if re.search(r"\[[a-z0-9_]+\]\s*$", j["name"], re.I)]
            passed = sum(1 for j in test_jobs if j.get("conclusion") == "success")
            failed = sum(1 for j in test_jobs if j.get("conclusion") in ("failure", "timed_out"))
            other = len(test_jobs) - passed - failed
            overall[r.get("conclusion") or "unknown"] += 1
            out.write(
                f"| {fmt_iso(r['createdAt'])} | "
                f"{r.get('conclusion') or r.get('status')} | "
                f"{passed} / {failed} / {other} | "
                f"[link]({r['url']}) |\n"
            )
            run_details.append((r, test_jobs, failed))

        # Per-run job breakdown (every test job, pass + fail).
        out.write("\n### Per-run job breakdown\n")
        for r, test_jobs, failed in run_details:
            wf_conclusion = r.get("conclusion") or r.get("status")
            out.write(
                f"\n**{fmt_iso(r['createdAt'])}** — `{r['headSha'][:9]}` · "
                f"workflow: **{wf_conclusion}** · [run]({r['url']})\n\n"
            )
            if not test_jobs:
                out.write("_No test jobs identified for this run._\n")
                continue
            # Sort: failures first, then alphabetic so failures stand out.
            ordered = sorted(
                test_jobs,
                key=lambda j: (j.get("conclusion") not in ("failure", "timed_out"),
                               j["name"].lower()),
            )
            for j in ordered:
                conclusion = j.get("conclusion") or "(running)"
                short_name = j["name"].split(" / ", 1)[-1]
                if conclusion == "success":
                    out.write(f"- ✅ {short_name}\n")
                elif conclusion in ("failure", "timed_out"):
                    step = first_failed_step(j) or "(unknown step)"
                    out.write(
                        f"- ❌ **{short_name}** — failed at step `{step}` "
                        f"([job log]({j.get('html_url', '')}))\n"
                    )
                    anns = filter_annotations(fetch_annotations(j["id"]))
                    shown = 0
                    for a in anns:
                        if shown >= 3:
                            break
                        msg = a.get("message")
                        if not msg:
                            continue
                        title = a.get("title") or a.get("annotation_level", "")
                        out.write(f"    - _{title}_: {trim(msg, 280)}\n")
                        shown += 1
                    if shown == 0:
                        out.write(
                            "    - _no failure annotation_ (likely runner crash, OOM, "
                            "or the failing pytest didn't emit a JUnit summary)\n"
                        )
                elif conclusion == "skipped":
                    out.write(f"- ⏭️ {short_name} _(skipped)_\n")
                elif conclusion == "cancelled":
                    out.write(f"- 🚫 {short_name} _(cancelled)_\n")
                else:
                    out.write(f"- ❔ {short_name} _({conclusion})_\n")

    out.write(f"\n---\n\nWorkflow-run conclusion totals: {dict(overall)}\n")


def parse_tier(value):
    if value is None or value.lower() == "all":
        return None
    tiers = []
    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        try:
            t = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid tier '{part}' (expected 1, 2, 3, or all)")
        if t not in (1, 2, 3):
            raise argparse.ArgumentTypeError(f"invalid tier '{part}' (expected 1, 2, 3, or all)")
        tiers.append(t)
    return tiers or None


def parse_type(value):
    if value is None:
        return None
    v = value.lower()
    if v in ("both", "all"):
        return None
    if v not in ("e2e", "unit"):
        raise argparse.ArgumentTypeError(f"invalid pipeline type '{value}' (expected e2e, unit, or both)")
    return v


def main():
    ap = argparse.ArgumentParser(
        description="CI status report for the (Tier 1/2/3) Models e2e+unit pipelines, "
                    "scheduled runs on main only.",
    )
    ap.add_argument("--tier", type=parse_tier, default=None,
                    help="filter by tier: 1, 2, 3, all, or comma list (default: all)")
    ap.add_argument("--type", dest="ptype", type=parse_type, default=None,
                    help="pipeline type: e2e, unit, or both (default: both)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max runs per pipeline (default {DEFAULT_LIMIT})")
    ap.add_argument("--days", type=int, default=None,
                    help="only include runs created in the last N days "
                         "(takes precedence over --limit when fewer runs land in the window)")
    ap.add_argument("--out", default="-", help="output file (default: stdout)")
    args = ap.parse_args()

    workflows = WORKFLOWS
    if args.tier is not None:
        workflows = [w for w in workflows if w[2] in args.tier]
    if args.ptype is not None:
        workflows = [w for w in workflows if w[3] == args.ptype]

    if not workflows:
        print("No workflows match the given filters.", file=sys.stderr)
        sys.exit(2)

    since = None
    if args.days is not None:
        since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)

    if args.days is not None:
        header_window = f"scheduled runs on `main` in the last {args.days} days"
    else:
        header_window = f"last {args.limit} scheduled runs on `main` per pipeline"

    if args.out == "-":
        write_report(sys.stdout, workflows, args.limit, since=since, header_window=header_window)
    else:
        with open(args.out, "w") as f:
            write_report(f, workflows, args.limit, since=since, header_window=header_window)
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
