#!/usr/bin/env python3
"""Deprecation reaper.

Reads .github/deprecations.json, and for each tracked deprecation looks up when its
PR merged (the PR's exact mergedAt timestamp, via `gh`). Once mergedAt + grace has
elapsed, the deprecation is "overdue". The script then keeps a single tracking issue
in sync:

  * if there are overdue deprecations, it opens the issue (or refreshes its body) and
    assigns/@-mentions the owners;
  * if there are none, it closes the tracking issue if one is open.

Deprecations whose PR is not merged yet are ignored, so the countdown only starts at
merge time -- you never have to know the merge date in advance.

Run locally with --dry-run to preview without touching GitHub.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal")
MANIFEST = ".github/deprecations.json"
TRACKING_LABEL = "deprecation-reaper"
TRACKING_TITLE = "Deprecations past their scheduled removal date"
MARKER = "<!-- deprecation-reaper:managed -->"


def run(args, check=True):
    res = subprocess.run(args, capture_output=True, text=True)
    if check and res.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(args)}\n{res.stderr}\n")
        sys.exit(1)
    return res.stdout.strip()


def parse_grace(s):
    m = re.fullmatch(r"\s*(\d+)\s*([dwm])\s*", s)
    if not m:
        raise ValueError(f"invalid grace {s!r} (use e.g. 30d, 2w, 1m)")
    n, unit = int(m.group(1)), m.group(2)
    return dt.timedelta(days=n * {"d": 1, "w": 7, "m": 30}[unit])


def pr_merged_at(pr):
    out = run(["gh", "pr", "view", str(pr), "--repo", REPO, "--json", "mergedAt,title,url"])
    data = json.loads(out)
    merged = data.get("mergedAt")
    return (
        dt.datetime.fromisoformat(merged.replace("Z", "+00:00")) if merged else None,
        data.get("title", ""),
        data.get("url", ""),
    )


def find_overdue(manifest, now):
    overdue = []
    for d in manifest.get("deprecations", []):
        merged_at, pr_title, pr_url = pr_merged_at(d["introduced_by_pr"])
        if merged_at is None:
            print(f"  - {d['id']}: PR #{d['introduced_by_pr']} not merged yet -> clock not started")
            continue
        deadline = merged_at + parse_grace(d["grace"])
        if now >= deadline:
            days = (now - deadline).days
            print(f"  - {d['id']}: OVERDUE by {days}d (merged {merged_at.date()}, due {deadline.date()})")
            overdue.append(
                {
                    **d,
                    "merged_at": merged_at,
                    "deadline": deadline,
                    "days_overdue": days,
                    "pr_title": pr_title,
                    "pr_url": pr_url,
                }
            )
        else:
            print(f"  - {d['id']}: not due yet (due {deadline.date()})")
    return overdue


def build_body(overdue):
    header_note = (
        "_Auto-managed by the deprecation-reaper workflow. Do not edit by hand; "
        + "remove the entry from `.github/deprecations.json` once the code is deleted._"
    )
    intro = (
        "The following deprecated APIs have passed their scheduled removal date and "
        + "should be deleted in a follow-up PR:"
    )
    lines = [MARKER, header_note, "", intro, ""]
    owners = set()
    for d in overdue:
        owners.update(d.get("owners", []))
        introduced = (
            f"- Introduced by [#{d['introduced_by_pr']}]({d['pr_url']}) "
            + f"(merged {d['merged_at'].date()}, removal due {d['deadline'].date()})"
        )
        lines += [
            f"### `{d['id']}` — overdue by {d['days_overdue']} day(s)",
            f"- {d['description']}",
            f"- Files: {', '.join('`'+f+'`' for f in d['files'])}",
            introduced,
            f"- Owners: {', '.join('@'+o for o in d.get('owners', []))}",
            "",
        ]
    if owners:
        lines.append("cc " + " ".join("@" + o for o in sorted(owners)))
    return "\n".join(lines), sorted(owners)


def find_tracking_issue():
    out = run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            REPO,
            "--label",
            TRACKING_LABEL,
            "--state",
            "open",
            "--json",
            "number,body",
            "--limit",
            "50",
        ]
    )
    for issue in json.loads(out or "[]"):
        if MARKER in (issue.get("body") or ""):
            return issue["number"]
    return None


def ensure_label(dry):
    if dry:
        print(f"[dry-run] ensure label {TRACKING_LABEL!r} exists")
        return
    run(
        [
            "gh",
            "label",
            "create",
            TRACKING_LABEL,
            "--repo",
            REPO,
            "--color",
            "B60205",
            "--description",
            "Tracks deprecated APIs past their scheduled removal date",
            "--force",
        ],
        check=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)

    now = dt.datetime.now(dt.timezone.utc)
    count = len(manifest.get("deprecations", []))
    print(f"deprecation-reaper: checking {count} tracked deprecation(s) at {now.isoformat()}")
    overdue = find_overdue(manifest, now)
    existing = find_tracking_issue()

    if not overdue:
        print("No overdue deprecations.")
        if existing:
            msg = "All tracked deprecations have been removed or are no longer overdue. Closing."
            if args.dry_run:
                print(f"[dry-run] comment + close issue #{existing}: {msg}")
            else:
                run(["gh", "issue", "comment", str(existing), "--repo", REPO, "--body", msg])
                run(["gh", "issue", "close", str(existing), "--repo", REPO])
        return

    body, owners = build_body(overdue)
    if existing:
        if args.dry_run:
            print(f"[dry-run] update issue #{existing} with refreshed body:\n{body}")
        else:
            run(["gh", "issue", "edit", str(existing), "--repo", REPO, "--body", body])
            if owners:
                run(
                    ["gh", "issue", "edit", str(existing), "--repo", REPO, "--add-assignee", ",".join(owners)],
                    check=False,
                )
        print(f"Updated tracking issue #{existing}.")
    else:
        ensure_label(args.dry_run)
        if args.dry_run:
            print(f"[dry-run] create issue '{TRACKING_TITLE}' (label {TRACKING_LABEL}, assignees {owners}):\n{body}")
        else:
            create = [
                "gh",
                "issue",
                "create",
                "--repo",
                REPO,
                "--title",
                TRACKING_TITLE,
                "--label",
                TRACKING_LABEL,
                "--body",
                body,
            ]
            if owners:
                create += ["--assignee", ",".join(owners)]
            url = run(create)
            print(f"Opened tracking issue: {url}")


if __name__ == "__main__":
    main()
