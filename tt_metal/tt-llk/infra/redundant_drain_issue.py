#!/usr/bin/env python3
"""Open a tt-metal issue for the drains a merge just added and the checker flags.

Runs after a push to main, with the repository checked out at the pushed head. For the pushed
range it collects the lines that added a `tensix_sync()` or `mop_sync()` call anywhere in the
repository (Quasar and vendored code excepted), runs check_redundant_drain.py with the reviewed-site
baseline on the touched files, and keeps the findings that sit on an added line. Each such finding
is a drain somebody put in with this merge that orders nothing where it was placed: typically a
workaround for a non-deterministic failure elsewhere.

An added line is one the range adds net of what it removes: the diff is taken once with rename
detection and whitespace ignored, a hunk that removes as many drain calls as it adds (a moved or
re-indented drain) adds none, and a site whose blamed commit is not in the range is dropped.

One issue is opened per merged pull request (or per commit when no pull request is found), listing
every flagged site, assigned to the author of the pull request, so the drain is tracked and removed
(REDUNDANT), replaced by the in-stream wait that does the job (OVERSYNC) or reviewed against its
callers (ADVISORY) rather than left in place. A marker in the body names the pull request (or the
commit); the open issues carrying the label are listed and matched on that marker locally, and a
match is updated with the current list of sites instead of being duplicated. A failed
`gh issue create` is retried without the assignee, naming the author in the body instead, and a
second failure fails the run.

Run with --dry-run to print what would be filed.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_redundant_drain import (  # noqa: E402
    DRAIN,
    EXTS,
    is_quasar,
    is_skipped_silently,
)

CHECKER = os.path.join(HERE, "check_redundant_drain.py")
BASELINE = os.path.join(HERE, "redundant_drain_baseline.txt")
REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal")
LABEL = "llk-redundant-drain"
MARKER = "<!-- llk-redundant-drain:{key} -->"

# The programs this script may start. A process is started only for a name in this table, with its
# arguments as a list and no shell, so nothing here is ever parsed by a shell.
PROGRAMS = {"git": "git", "gh": "gh", "python": sys.executable}


def try_run(program, args):
    """(returncode, stdout, stderr) of a program from the table."""
    res = subprocess.run([PROGRAMS[program], *args], capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def run(program, args, check=True):
    rc, out, err = try_run(program, args)
    if check and rc != 0:
        sys.stderr.write(
            f"command failed: {' '.join([PROGRAMS[program], *args])}\n{err}\n"
        )
        sys.exit(1)
    return out


def commit(ref):
    """The full SHA of `ref`, which must name a commit; an option-shaped value is rejected."""
    return run(
        "git", ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"]
    ).strip()


def added_drain_lines(base, head):
    """{file: {line}} of lines in head that this range added, net, and that contain a drain call.

    One diff with rename detection and whitespace ignored, so a moved or re-indented file is not
    wholly new; within a hunk, drain calls on removed lines cancel drain calls on added lines, so
    a drain that merely changed line or spelling is not added.
    """
    diff = run("git", ["diff", "-U0", "-M", "-w", f"{base}..{head}"])
    added, path, line, hunk_plus, hunk_minus = {}, None, 0, [], 0

    def close_hunk():
        if path and len(hunk_plus) > hunk_minus:
            added.setdefault(path, set()).update(hunk_plus)

    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            close_hunk()
            hunk_plus, hunk_minus = [], 0
            p = raw[4:].strip()
            p = p[2:] if p.startswith("b/") else p
            path = (
                p
                if p.endswith(EXTS) and not is_quasar(p) and not is_skipped_silently(p)
                else None
            )
            continue
        if raw.startswith("--- ") or path is None:
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            close_hunk()
            hunk_plus, hunk_minus = [], 0
            line = int(m.group(1))
            continue
        if raw.startswith("+"):
            if DRAIN.search(raw):
                hunk_plus.append(line)
            line += 1
        elif raw.startswith("-") and DRAIN.search(raw):
            hunk_minus += 1
    close_hunk()
    return added


def findings_on(files):
    # Checked: the checker exits 0 without --strict, so a non-zero status is a crash, and a crash
    # must not read as "none flagged".
    out = run("python", [CHECKER, "--json", "--baseline", BASELINE, *files])
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


def blame_commit(path, line, head):
    out = run(
        "git",
        ["blame", "-L", f"{line},{line}", "--porcelain", head, "--", path],
        check=False,
    )
    return out.split("\n", 1)[0].split(" ", 1)[0] if out else ""


def pull_for(sha):
    out = run("gh", ["api", f"repos/{REPO}/commits/{sha}/pulls"], check=False)
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
        "gh",
        [
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
        "gh",
        [
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


def open_issues():
    """{number: body} of the open issues carrying the label; the marker is matched locally."""
    out = run(
        "gh",
        [
            "issue",
            "list",
            "--repo",
            REPO,
            "--label",
            LABEL,
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "number,body",
        ],
        check=False,
    )
    try:
        return {it["number"]: it.get("body") or "" for it in json.loads(out)}
    except (ValueError, TypeError, KeyError):
        return {}


def describe(site):
    d = f"`{site['drain']}()`"
    if site["kind"] == "REDUNDANT":
        return f"a {d} that a following `{site.get('covered_by') or 'tensix_sync'}()` already covers: remove it"
    if site["kind"] == "OVERSYNC":
        return (
            f"a {d} whose only effect here is to order the Tensix pushes after it behind in-flight work, "
            "which a `STALLWAIT` on the conflicting unit does without stalling the RISC: replace it, "
            "unless a caller relies on it, in which case the caller should drain itself"
        )
    return (
        f"a {d} after which nothing in the function needs the RISC to wait: remove it, or if a "
        "caller relies on it, move the drain to that caller"
    )


VERB = {"REDUNDANT": "Remove", "OVERSYNC": "Replace", "ADVISORY": "Review"}


def issue_key(pr, head):
    return f"pr:{pr['number']}" if pr else f"commit:{head[:11]}"


def marker_for(pr, head):
    return MARKER.format(key=issue_key(pr, head))


def body_for(sites, pr, head, author_note=""):
    origin = pr["url"] if pr else f"Commit {head[:11]}"
    lines = [
        marker_for(pr, head),
        f"{origin} added {len(sites)} drain(s) the redundant-drain checker flags:",
        "",
    ]
    for s in sites:
        lines.append(
            f"- `{s['file']}:{s['line']}` in `{s['function'].strip()}`: {describe(s)} "
            f"({s['kind'].lower()}: {s['reason']})"
        )
    lines += [
        "",
        "A drain blocks the RISC until the Tensix thread is idle. It is needed when the RISC is about to touch state the "
        "drained work produces: a mailbox or semaphore, a configuration register through the memory map, a GPR, a volatile "
        "load or store, a store through a pointer. A Tensix push that writes configuration, a GPR, a semaphore, Dest or a "
        "mutex is ordered behind in-flight work by a `tensix_sync()` before it, but an in-stream `STALLWAIT` on the "
        "conflicting unit does the same without stalling the RISC; a `mop_sync()` orders no push at all. A drain followed "
        "by nothing of that does not fix the hazard it was added for; it only changes timing, and it costs a full pipeline "
        "drain every time it runs.",
        "",
        "If a drain was added as a workaround for a failure seen elsewhere, that failure is the thing to root-cause; note it "
        "here so the drain is not mistaken for a fix. A site the checker cannot see the caller of may still serve that "
        "caller: then move the drain to the caller, or keep it and record it in `redundant_drain_baseline.txt` with the "
        "reason.",
        "",
        "Reported by `tt_metal/tt-llk/infra/check_redundant_drain.py`.",
    ]
    if author_note:
        lines += ["", author_note]
    return "\n".join(lines)


def title_for(sites, pr, head):
    first = f"{os.path.basename(sites[0]['file'])}:{sites[0]['line']}"
    more = f" and {len(sites) - 1} more" if len(sites) > 1 else ""
    origin = f"#{pr['number']}" if pr else f"commit {head[:11]}"
    verbs = sorted(
        {VERB.get(s["kind"], "Review") for s in sites}, key=list(VERB.values()).index
    )
    verb = (
        verbs[0]
        if len(verbs) == 1
        else " or ".join(v.lower() if i else v for i, v in enumerate(verbs))
    )
    noun = "drain" if len(sites) == 1 else "drains"
    return f"[LLK] {verb} the {noun} {origin} added at {first}{more}"


def file_issue(sites, pr, head, existing, dry):
    """Create or update the one issue for this pull request (or commit); returns its number or URL."""
    sites = sorted(sites, key=lambda s: (s["file"], s["line"]))
    marker = marker_for(pr, head)
    title = title_for(sites, pr, head)
    body = body_for(sites, pr, head)
    number = next((n for n, b in existing.items() if marker in b), None)
    if dry:
        print(
            ("would update #%s: " % number if number is not None else "would create: ")
            + title
        )
        print(body + "\n")
        return number
    if number is not None:
        run(
            "gh",
            [
                "issue",
                "edit",
                str(number),
                "--repo",
                REPO,
                "--title",
                title,
                "--body",
                body,
            ],
        )
        print(f"updated #{number}: {title}")
        return number
    base = ["issue", "create", "--repo", REPO, "--title", title, "--label", LABEL]
    author = (pr or {}).get("author")
    if author:
        rc, out, err = try_run("gh", base + ["--body", body, "--assignee", author])
        if rc == 0:
            print(f"created {out.strip()}: {title}")
            return out.strip()
        sys.stderr.write(
            f"could not create the issue assigned to {author}: {err.strip()}\n"
        )
        body = body_for(
            sites, pr, head, author_note=f"Author: @{author} (could not be assigned)."
        )
    rc, out, err = try_run("gh", base + ["--body", body])
    if rc != 0:
        sys.stderr.write(f"gh issue create failed:\n{err}\n")
        sys.exit(1)
    print(f"created {out.strip()}: {title}")
    return out.strip()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, help="commit before the push")
    ap.add_argument(
        "--head", required=True, help="commit pushed; the checkout must be at it"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.base = commit(args.base)
    args.head = commit(args.head)
    checked_out = run("git", ["rev-parse", "--verify", "HEAD^{commit}"]).strip()
    if checked_out != args.head:
        sys.stderr.write(
            f"the checkout is at {checked_out[:11]} but --head is {args.head[:11]}; the sources "
            "and helper trees the checker reads must be those of the head, so check it out first\n"
        )
        sys.exit(1)

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

    in_range = set(run("git", ["rev-list", f"{args.base}..{args.head}"]).split())
    by_pr = {}
    for site in sites:
        sha = blame_commit(site["file"], site["line"], args.head) or args.head
        if sha not in in_range:
            continue  # the line is older than the push: it moved, it was not added
        pr = pull_for(sha)
        key = pr["number"] if pr else sha
        by_pr.setdefault(key, (pr, []))[1].append(site)

    ensure_label(args.dry_run)
    existing = open_issues()
    for _key, (pr, pr_sites) in by_pr.items():
        file_issue(pr_sites, pr, args.head, existing, args.dry_run)


if __name__ == "__main__":
    main()
