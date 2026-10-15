#!/usr/bin/env python3
"""Open a tt-metal issue for a drain that a merge just added and the checker flags.

Runs after a push to main. For the pushed range it collects the lines that added a
`tensix_sync()` or `mop_sync()` call under the Blackhole or Wormhole LLK trees, runs
check_redundant_drain.py on the touched headers at the new head, and keeps the findings that sit
on an added line. Each such finding is a drain somebody put in with this merge that orders
nothing where it was placed: typically a workaround for a non-deterministic failure elsewhere.
One issue is opened per merged pull request, listing its sites, assigned to the author of the
pull request, so the workaround is tracked and removed (or replaced by the wait the real hazard
needs) rather than left in place.

An issue is created once per site: a marker in the body names the file, function and head commit,
and an open issue carrying the same file and function is updated instead of duplicated.

Run with --dry-run to print what would be filed.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "check_redundant_drain.py")
REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal")
LABEL = "llk-redundant-drain"
MARKER = "<!-- llk-redundant-drain:{file}:{function} -->"
TREES = ("tt_metal/tt-llk/tt_llk_blackhole/", "tt_metal/tt-llk/tt_llk_wormhole_b0/")
DRAIN = re.compile(r"\b(?:ckernel::)?(?:tensix_sync|mop_sync)\s*\(\s*\)")


def run(args, check=True, input_text=None):
    res = subprocess.run(args, capture_output=True, text=True, input=input_text)
    if check and res.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(args)}\n{res.stderr}\n")
        sys.exit(1)
    return res.stdout


def added_drain_lines(base, head):
    """{file: {line}} of lines in head that this range added and that contain a drain call."""
    files = [
        f
        for f in run(
            ["git", "diff", "--name-only", f"{base}..{head}", "--", *TREES]
        ).split()
        if f.endswith((".h", ".hpp"))
    ]
    added = {}
    for f in files:
        diff = run(["git", "diff", "-U0", f"{base}..{head}", "--", f], check=False)
        line = 0
        for raw in diff.splitlines():
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if m:
                line = int(m.group(1))
                continue
            if raw.startswith("+") and not raw.startswith("+++"):
                if DRAIN.search(raw):
                    added.setdefault(f, set()).add(line)
                line += 1
    return added


def findings_on(files):
    out = run([sys.executable, CHECKER, "--json", *files], check=False)
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


def blame_commit(path, line, head):
    out = run(
        ["git", "blame", "-L", f"{line},{line}", "--porcelain", head, "--", path],
        check=False,
    )
    return out.split("\n", 1)[0].split(" ", 1)[0] if out else ""


def pull_for(sha):
    out = run(["gh", "api", f"repos/{REPO}/commits/{sha}/pulls"], check=False)
    try:
        prs = json.loads(out)
    except ValueError:
        return None
    if not isinstance(prs, list) or not prs:
        return None
    pr = prs[0]
    return {
        "number": pr["number"],
        "title": pr.get("title", ""),
        "url": pr.get("html_url", ""),
        "author": (pr.get("user") or {}).get("login", ""),
    }


def ensure_label(dry):
    exists = run(
        [
            "gh",
            "label",
            "list",
            "--repo",
            REPO,
            "--search",
            LABEL,
            "--json",
            "name",
            "--jq",
            f'.[]|select(.name=="{LABEL}")|.name',
        ],
        check=False,
    ).strip()
    if exists or dry:
        return
    run(
        [
            "gh",
            "label",
            "create",
            LABEL,
            "--repo",
            REPO,
            "--color",
            "D93F0B",
            "--description",
            "A Tensix drain that orders nothing where it was added",
        ],
        check=False,
    )


def open_issue_for(marker):
    out = run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            REPO,
            "--label",
            LABEL,
            "--state",
            "open",
            "--search",
            marker,
            "--json",
            "number,body",
        ],
        check=False,
    )
    try:
        for it in json.loads(out):
            if marker in (it.get("body") or ""):
                return it["number"]
    except ValueError:
        pass
    return None


def body_for(site, pr, head):
    where = f"`{site['file']}:{site['line']}` in `{site['function'].strip()}`"
    what = (
        "a `mop_sync()` that a following `tensix_sync()` already covers"
        if site["kind"] == "REDUNDANT" and site["drain"] == "mop_sync"
        else (
            "a `tensix_sync()` that a following `tensix_sync()` already covers"
            if site["kind"] == "REDUNDANT"
            else "a `tensix_sync()` after which nothing in the function needs the RISC to wait: only Tensix instructions or plain code follow"
        )
    )
    intro = (
        f"{pr['url']} added {what} at {where}."
        if pr
        else f"Commit {head[:11]} added {what} at {where}."
    )
    return "\n".join(
        [
            MARKER.format(file=site["file"], function=site["function"].strip()[:60]),
            intro,
            "",
            "A drain blocks the RISC until the Tensix thread is idle. That only orders something when the RISC is about to "
            "touch state the drained work produces: a mailbox or semaphore, a configuration register through the memory map, "
            "a GPR, a volatile load or store, a store through a pointer. Tensix instructions pushed after it are ordered behind "
            "the earlier ones by the thread anyway. A drain placed where nothing of that follows does not fix the hazard it was "
            "added for; it only changes timing, and it costs a full pipeline drain every time it runs.",
            "",
            "Please either remove it, or replace it with the wait the real hazard needs (a `STALLWAIT` on the unit and "
            "condition that actually conflict, placed where the conflict is), or move the drain to the call site that needs "
            "it. If the drain was added as a workaround for a failure seen elsewhere, that failure is the thing to root-cause; "
            "note it here so the drain is not mistaken for a fix.",
            "",
            f"Reported by `tt_metal/tt-llk/infra/check_redundant_drain.py` ({site['kind'].lower()}: {site['reason']}).",
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, help="commit before the push")
    ap.add_argument("--head", required=True, help="commit pushed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.head = run(["git", "rev-parse", args.head]).strip()

    added = added_drain_lines(args.base, args.head)
    if not added:
        print("no drain calls added in this range")
        return
    sites = [
        f for f in findings_on(sorted(added)) if f["line"] in added.get(f["file"], ())
    ]
    if not sites:
        print(
            f"{sum(len(v) for v in added.values())} drain call(s) added; none flagged"
        )
        return

    by_pr = {}
    for site in sites:
        sha = blame_commit(site["file"], site["line"], args.head) or args.head
        pr = pull_for(sha)
        key = pr["number"] if pr else sha
        by_pr.setdefault(key, (pr, []))[1].append(site)

    ensure_label(args.dry_run)
    for key, (pr, pr_sites) in by_pr.items():
        for site in pr_sites:
            marker = MARKER.format(
                file=site["file"], function=site["function"].strip()[:60]
            )
            title = f"[LLK] Remove the drain added at {os.path.basename(site['file'])}:{site['line']} ({site['function'].strip()[:40]})"
            body = body_for(site, pr, args.head)
            existing = open_issue_for(marker)
            if args.dry_run:
                print(
                    ("would update #%s: " % existing if existing else "would create: ")
                    + title
                )
                print(body + "\n")
                continue
            if existing:
                run(
                    [
                        "gh",
                        "issue",
                        "edit",
                        str(existing),
                        "--repo",
                        REPO,
                        "--body",
                        body,
                    ]
                )
                print(f"updated #{existing}: {title}")
            else:
                cmd = [
                    "gh",
                    "issue",
                    "create",
                    "--repo",
                    REPO,
                    "--title",
                    title,
                    "--body",
                    body,
                    "--label",
                    LABEL,
                ]
                if pr and pr.get("author"):
                    cmd += ["--assignee", pr["author"]]
                print("created " + run(cmd, check=False).strip() + ": " + title)


if __name__ == "__main__":
    main()
