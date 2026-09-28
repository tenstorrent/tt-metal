#!/usr/bin/env python3
"""Join a repo's closed issues, PRs and git history into "fix cases": one per bug with the commits that fixed it.

  make_cases.py --issues 'raw/x_issue/*.jsonl' --prs 'raw/x_pr/*.jsonl' --git /path/to/clone [--ref origin/main]
                 --out cases.jsonl [--vendored-prefix tt_metal/tt-llk/]

Inputs come from fetch_search.py / fetch_repo.py (GitHub GraphQL dumps) and a local clone for commits. A case is:
  - a closed issue that looks like a bug (labels, title), with every fix commit linked to it: the closing PR, merged
    PRs that declare they close it, and commits whose message cites it; or
  - a merged PR with no bug issue whose title or body says it fixes something.
For each fix commit the case records files, size, and later history: reverts of it, and later commits that cite the
same issue/PR or touch the same files with a fix-like message (the candidates for "the first fix was incomplete").
Nothing here judges the bug; the triage and deep-read workflows do that. This only assembles evidence.
"""
import argparse
import collections
import glob
import json
import re
import subprocess
import sys

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
p.add_argument("--issues", required=True)
p.add_argument("--prs", required=True)
p.add_argument("--git", required=True)
p.add_argument("--ref", default="origin/main")
p.add_argument("--out", required=True)
p.add_argument("--followup-days", type=int, default=365)
p.add_argument(
    "--xref-git",
    help="another clone whose commits may fix this repo's issues (e.g. where the code moved)",
)
p.add_argument(
    "--xref-pattern",
    default=r"tt-llk(?:/issues/|#)(\d+)",
    help="regex with one group = this repo's issue number, matched in --xref-git commit messages",
)
a = p.parse_args()

BUG_LABEL = re.compile(
    r"(^bug$|_bug$|bug$|^ci-bug$|regression|hang|crash|pcc|correctness|n[d] failure|flaky)",
    re.I,
)
FIXY = re.compile(
    r"\b(fix(es|ed|ing)?|bug|hang(s|ing)?|deadlock|race|crash|segfault|regression|incorrect|wrong|"
    r"broken|overflow|underflow|nan|inf|mismatch|pcc|corrupt\w*|leak|off[- ]by[- ]one|revert)\b",
    re.I,
)
NOT_BUG_TITLE = re.compile(
    r"^\s*(\[?(feat|feature|docs?|chore|refactor|perf|ci|test|build|bump|uplift|add)\b)",
    re.I,
)


def load_jsonl(pattern):
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for ln in open(f):
            if ln.strip():
                x = json.loads(ln)
                rows[x["number"]] = x
    return rows


issues = load_jsonl(a.issues)
prs = load_jsonl(a.prs)
print(f"loaded {len(issues)} issues, {len(prs)} PRs", file=sys.stderr)

# ---- git history (first-parent, so a squash-merged PR is one commit) ----
fmt = "%x1e%H%x1f%ct%x1f%s%x1f%b%x1f"
log = subprocess.run(
    [
        "git",
        "-C",
        a.git,
        "log",
        "--first-parent",
        "--no-merges",
        "--numstat",
        f"--format={fmt}",
        a.ref,
    ],
    capture_output=True,
    text=True,
    errors="replace",
    check=True,
).stdout
commits, order = {}, []
for rec in log.split("\x1e")[1:]:
    parts = rec.split("\x1f")
    if len(parts) < 5:
        continue
    oid, ct, subj, body, stat = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
    files = []
    for ln in stat.strip().splitlines():
        cols = ln.split("\t")
        if len(cols) == 3:
            add = int(cols[0]) if cols[0].isdigit() else 0
            dele = int(cols[1]) if cols[1].isdigit() else 0
            files.append((cols[2], add, dele))
    commits[oid] = {"oid": oid, "t": ct, "subject": subj, "body": body, "files": files}
    order.append(oid)
order.reverse()  # oldest first
pos = {o: i for i, o in enumerate(order)}
print(f"loaded {len(commits)} first-parent commits", file=sys.stderr)

pr_commit = {}
for o, c in commits.items():
    m = re.search(r"\(#(\d+)\)\s*$", c["subject"])
    if m:
        pr_commit.setdefault(int(m.group(1)), o)
for n, pr in prs.items():
    mc = (pr.get("mergeCommit") or {}).get("oid")
    if pr.get("mergedAt") and mc in commits:
        pr_commit[n] = mc

cites = collections.defaultdict(
    set
)  # issue/pr number -> commits whose message cites it
for o, c in commits.items():
    text = c["subject"] + "\n" + c["body"]
    own = re.search(r"\(#(\d+)\)\s*$", c["subject"])
    for m in re.finditer(r"#(\d{2,6})\b", text):
        k = int(m.group(1))
        if own and k == int(own.group(1)):
            continue
        cites[k].add(o)

reverts = collections.defaultdict(list)
for o, c in commits.items():
    m = re.search(r"This reverts commit ([0-9a-f]{7,40})", c["body"])
    if m:
        tgt = next((x for x in commits if x.startswith(m.group(1))), None)
        if tgt:
            reverts[tgt].append(o)
    elif c["subject"].lower().startswith('revert "'):
        inner = c["subject"][8:].rsplit('"', 1)[0]
        mm = re.search(r"\(#(\d+)\)\s*$", inner)
        if mm and int(mm.group(1)) in pr_commit:
            reverts[pr_commit[int(mm.group(1))]].append(o)

by_file = collections.defaultdict(list)
for o in order:
    for f, _, _ in commits[o]["files"]:
        by_file[f].append(o)


def summarize(o):
    c = commits[o]
    return {
        "oid": o,
        "subject": c["subject"][:200],
        "t": c["t"],
        "files": [f for f, _, _ in c["files"]][:60],
        "nfiles": len(c["files"]),
        "adds": sum(x[1] for x in c["files"]),
        "dels": sum(x[2] for x in c["files"]),
    }


def later_history(fix_oids, refs):
    horizon = max(commits[o]["t"] for o in fix_oids) + a.followup_days * 86400
    start = max(pos[o] for o in fix_oids)
    files = {f for o in fix_oids for f, _, _ in commits[o]["files"]}
    cited = set().union(*(cites.get(r, set()) for r in refs)) if refs else set()
    rev = [r for o in fix_oids for r in reverts.get(o, [])]
    touch = set()
    for f in files:
        for o in by_file.get(f, []):
            if (
                pos[o] > start
                and commits[o]["t"] <= horizon
                and FIXY.search(commits[o]["subject"])
            ):
                touch.add(o)
    cited = {o for o in cited if pos[o] > start and o not in fix_oids}
    srt = lambda s: sorted(s, key=lambda o: pos[o])  # noqa: E731
    return {
        "reverts": [summarize(o) for o in srt(set(rev))],
        "citing_later": [summarize(o) for o in srt(cited)][:20],
        "fixlike_same_files": [summarize(o) for o in srt(touch - cited - set(rev))][
            :25
        ],
        "fixlike_same_files_count": len(touch - cited - set(rev)),
    }


def issue_is_bug(x):
    labels = [n["name"] for n in x["labels"]["nodes"]]
    if any(BUG_LABEL.search(lb) for lb in labels):
        return True, "label"
    if FIXY.search(x["title"]) and not NOT_BUG_TITLE.search(x["title"]):
        return True, "title"
    return False, None


cases, used_prs = [], set()
for n, x in sorted(issues.items()):
    if x.get("stateReason") not in ("COMPLETED", None):
        continue
    isbug, why = issue_is_bug(x)
    if not isbug:
        continue
    linked = set()
    for ev in x["timelineItems"]["nodes"]:
        if ev["__typename"] == "ClosedEvent" and ev.get("closer"):
            cl = ev["closer"]
            if cl["__typename"] == "PullRequest" and cl.get("merged"):
                linked.add(cl["number"])
        if ev["__typename"] == "CrossReferencedEvent" and (ev.get("source") or {}).get(
            "merged"
        ):
            pn = ev["source"]["number"]
            pr = prs.get(pn)
            if pr and n in [
                i["number"] for i in pr["closingIssuesReferences"]["nodes"]
            ]:
                linked.add(pn)
    fix = {pr_commit[pn] for pn in linked if pn in pr_commit}
    fix |= {
        o
        for o in cites.get(n, set())
        if FIXY.search(commits[o]["subject"])
        or re.match(rf"^#{n}\b", commits[o]["subject"])
    }
    cases.append(
        {
            "id": f"I{n}",
            "kind": "issue",
            "issue": n,
            "title": x["title"],
            "labels": [l["name"] for l in x["labels"]["nodes"]],
            "bug_signal": why,
            "created": x["createdAt"],
            "closed": x["closedAt"],
            "body": (x.get("body") or "")[:4000],
            "comments": x["comments"]["totalCount"],
            "prs": sorted(linked),
            "fix_oids": sorted(fix, key=lambda o: pos[o]),
        }
    )
    used_prs |= linked

closing = collections.defaultdict(set)
for pn, pr in prs.items():
    for i in pr["closingIssuesReferences"]["nodes"]:
        closing[i["number"]].add(pn)
by_id = {c["issue"]: c for c in cases}
for inum, pns in closing.items():
    c = by_id.get(inum)
    if c:
        for pn in pns:
            if (
                prs[pn].get("mergedAt")
                and pn in pr_commit
                and pr_commit[pn] not in c["fix_oids"]
            ):
                c["prs"] = sorted(set(c["prs"]) | {pn})
                c["fix_oids"] = sorted(
                    set(c["fix_oids"]) | {pr_commit[pn]}, key=lambda o: pos[o]
                )
                used_prs.add(pn)

for pn, pr in sorted(prs.items()):
    if pn in used_prs or not pr.get("mergedAt") or pn not in pr_commit:
        continue
    title = pr["title"]
    if (
        not FIXY.search(title)
        or NOT_BUG_TITLE.search(title)
        or title.lower().startswith("revert")
    ):
        continue
    cases.append(
        {
            "id": f"P{pn}",
            "kind": "pr",
            "issue": None,
            "title": title,
            "labels": [l["name"] for l in pr["labels"]["nodes"]],
            "bug_signal": "pr-title",
            "created": pr["createdAt"],
            "closed": pr["mergedAt"],
            "body": (pr.get("body") or "")[:4000],
            "comments": pr["comments"]["totalCount"],
            "prs": [pn],
            "fix_oids": [pr_commit[pn]],
        }
    )

xref = collections.defaultdict(list)
if a.xref_git:
    xl = subprocess.run(
        [
            "git",
            "-C",
            a.xref_git,
            "log",
            "--first-parent",
            "--format=%H%x1f%s%x1f%b%x1e",
            a.ref,
        ],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    for rec in xl.split("\x1e"):
        parts = rec.strip().split("\x1f")
        if len(parts) >= 2:
            for m in re.finditer(
                a.xref_pattern,
                parts[1] + "\n" + parts[2] if len(parts) > 2 else parts[1],
                re.I,
            ):
                xref[int(m.group(1))].append(
                    {"oid": parts[0], "subject": parts[1][:200]}
                )
for c in cases:
    if c["issue"] in xref:
        c["xref_fix"] = xref[c["issue"]]
    c["fix"] = [summarize(o) for o in c["fix_oids"]]
    c["later"] = (
        later_history(
            c["fix_oids"], [c["issue"]] + c["prs"] if c["issue"] else c["prs"]
        )
        if c["fix_oids"]
        else None
    )
    c["review_threads"] = sum(
        prs[pn]["reviewThreads"]["totalCount"] for pn in c["prs"] if pn in prs
    )
with open(a.out, "w") as fh:
    for c in cases:
        fh.write(json.dumps(c) + "\n")
k = collections.Counter(c["kind"] for c in cases)
nf = sum(1 for c in cases if c["fix_oids"])
nr = sum(1 for c in cases if c["later"] and c["later"]["reverts"])
print(
    f"{len(cases)} cases ({dict(k)}); {nf} with fix commits; {nr} with a revert",
    file=sys.stderr,
)
