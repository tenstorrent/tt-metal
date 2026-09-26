#!/usr/bin/env python3
"""Track what has been DONE about each confirmed finding, durably and outside the derived report.

`CONFIRMED.md` is regenerated from `verdicts/*.json` by consolidate.py, so any note written into it is
lost on the next wave. Dispositions live here instead, in `dispositions.json`, keyed by the same
`file:line` identity consolidate.py uses to dedup findings. consolidate.py reads this file and stamps
the state into CONFIRMED.md, and writes OPEN.md containing only the findings still needing work.

  disposition.py [--run DIR] list                                  # counts per state
  disposition.py list pr_open                           # everything in one state
  disposition.py show tt_metal/common/work_split.cpp:318
  disposition.py set <file:line> pr_open --pr 51234 --issue 51233 -m "fix + regression test"
  disposition.py set <file:line> not_a_bug -m "guarded by caller, see X"
  disposition.py sync                                   # ask GitHub; flip merged PRs to merged
  disposition.py check                                  # dispositions that match no current finding

States:
  in_progress  being worked on now (no PR yet)      -> hidden from OPEN.md, listed under Active
  pr_open      fix PR is open                       -> hidden from OPEN.md, listed under Active
  merged       PR merged; finding is fixed          -> done, never resurface
  wont_fix     real but deliberately not fixing     -> done, never resurface
  not_a_bug    turned out to be a false positive    -> done, never resurface
  duplicate    same defect as another finding       -> done, never resurface (use --of <file:line>)
  pr_closed    PR closed without merging            -> back in OPEN.md under "needs attention"
  already_filed  an open GitHub issue/PR already reports it (filed_check.py) -> done, never filed again
  filed_closed   reported and closed on GitHub, but still present in the tree -> OPEN.md "needs attention"
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_dir, state  # noqa: E402

HERE = run_dir()
PATH = f"{HERE}/dispositions.json"
REPO = state(HERE)["repo"]

ACTIVE = ("in_progress", "pr_open")
DONE = (
    "merged",
    "wont_fix",
    "not_a_bug",
    "duplicate",
    "already_filed",
    "fixed_upstream",
)
ATTENTION = ("pr_closed", "filed_closed")
# "open" is not a stored state: it clears the state so the finding stays actionable in OPEN.md,
# while keeping any severity override / note recorded alongside it.
STATES = ACTIVE + DONE + ATTENTION + ("open",)
SEVERITIES = ("high", "medium", "low")


def load():
    if not os.path.exists(PATH):
        return {}
    with open(PATH) as fh:
        return json.load(fh)


def save(d):
    tmp = PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    os.replace(tmp, PATH)  # atomic: a crash mid-write cannot truncate the real file


def findings():
    p = f"{HERE}/CONFIRMED.json"
    return json.load(open(p)) if os.path.exists(p) else []


def key_of(f):
    return f"{f['file']}:{f['line']}"


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve(k, known):
    """Accept an exact file:line key, or a unique suffix/substring of one."""
    if k in known:
        return k
    hits = [x for x in known if x.endswith(k) or k in x]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(
            f"no confirmed finding matches '{k}' (expected <file>:<line> as printed in CONFIRMED.md)"
        )
    sys.exit("ambiguous — matches:\n  " + "\n  ".join(sorted(hits)))


def cmd_set(a):
    if a.state not in STATES:
        sys.exit(f"unknown state '{a.state}' — one of: {', '.join(STATES)}")
    fs = {key_of(f): f for f in findings()}
    d = load()
    k = resolve(a.key, set(fs) | set(d))
    e = d.get(k, {})
    # snapshot identity so `check` can detect line drift after the audit tree is re-pointed
    if k in fs and "summary" not in e:
        e["summary"] = fs[k]["summary"][:160]
        e["severity"] = fs[k]["severity"]
    if a.state == "open":
        e.pop("state", None)  # stays actionable in OPEN.md
    else:
        e["state"] = a.state
    e["updated"] = now()
    if a.pr:
        e["pr"] = int(str(a.pr).lstrip("#"))
    if a.issue:
        e["issue"] = int(str(a.issue).lstrip("#"))
    if a.of:
        e["duplicate_of"] = resolve(a.of, set(fs))
    if a.severity:
        if a.severity not in SEVERITIES:
            sys.exit(
                f"unknown severity '{a.severity}' — one of: {', '.join(SEVERITIES)}"
            )
        if not a.message:
            sys.exit(
                "--severity requires -m: record WHY the audit's severity was wrong, or it gets re-derived"
            )
        e["severity_override"] = a.severity
    if a.message:
        e["note"] = a.message
    d[k] = e
    save(d)
    sev = (
        f" [{e['severity']} -> {e['severity_override']}]"
        if e.get("severity_override")
        else ""
    )
    print(
        f"{k} -> {e.get('state', 'open')}{sev}"
        + (f" (PR #{e['pr']})" if e.get("pr") else "")
    )


def cmd_list(a):
    d = load()
    if a.state:
        rows = [(k, v) for k, v in sorted(d.items()) if v.get("state") == a.state]
        for k, v in rows:
            pr = f" PR #{v['pr']}" if v.get("pr") else ""
            print(f"{v['state']:12} {k}{pr}  {v.get('note','')}")
        print(f"\n{len(rows)} in state {a.state}")
        return
    total = len(findings())
    counts = {s: sum(1 for v in d.values() if v.get("state") == s) for s in STATES}
    for s in STATES:
        if counts[s]:
            print(f"  {s:12} {counts[s]}")
    handled = sum(counts[s] for s in ACTIVE + DONE)
    print(
        f"\n{total} confirmed · {handled} handled (active+done) · {total - handled} still open"
    )


def cmd_show(a):
    fs = {key_of(f): f for f in findings()}
    d = load()
    k = resolve(a.key, set(fs) | set(d))
    if k in fs:
        f = fs[k]
        print(f"[{f['severity'].upper()}] {k} — {f['category']}\n\n{f['summary']}\n")
        print(f"Failure scenario: {f['failure_scenario']}\n")
        print(f"Evidence: {f['evidence']}\n")
        print(f"Suggested fix: {f['suggested_fix']}\n")
        print(
            f"Verifiers: {f.get('votes_kept','?')}/{f.get('votes_total','?')} · batch {f['batch']}\n"
        )
    print("Disposition: " + (json.dumps(d[k], indent=1) if k in d else "none (open)"))


def cmd_sync(a):
    """Ask GitHub about every recorded PR and move the finding's state to match reality."""
    d = load()
    todo = {
        k: v for k, v in d.items() if v.get("pr") and v.get("state") not in ("merged",)
    }
    if not todo:
        print("no open PRs to sync")
        return
    changed = 0
    for k, v in sorted(todo.items()):
        pr = v["pr"]
        r = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--repo",
                REPO,
                "--json",
                "state,mergedAt,url,title",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(
                f"  ?  PR #{pr} ({k}): gh failed — {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown'}"
            )
            continue
        info = json.loads(r.stdout)
        st = info["state"]
        new = {"MERGED": "merged", "CLOSED": "pr_closed", "OPEN": "pr_open"}.get(
            st, v["state"]
        )
        if new != v.get("state"):
            v["state"], v["updated"] = new, now()
            if st == "MERGED":
                v["merged_at"] = info.get("mergedAt")
            v["pr_url"] = info["url"]
            changed += 1
            print(f"  -> PR #{pr} {st}: {k} now {new}")
        else:
            print(f"     PR #{pr} {st}: {k} unchanged")
    if changed:
        save(d)
    print(
        f"\n{changed} finding(s) updated. Run consolidate.py to refresh CONFIRMED.md / OPEN.md."
    )


def cmd_check(a):
    """Dispositions whose file:line no longer matches a confirmed finding (line drift, typo, re-run)."""
    fs = {key_of(f): f for f in findings()}
    d = load()
    orphans = [k for k in d if k not in fs]
    if not orphans:
        print(f"all {len(d)} dispositions match a confirmed finding")
        return
    print(
        f"{len(orphans)} disposition(s) match NO confirmed finding — line drift or a typo:\n"
    )
    for k in sorted(orphans):
        v = d[k]
        print(f"  {k}  [{v.get('state')}]  {v.get('summary','')[:90]}")
        f, _, ln = k.rpartition(":")
        near = [x for x in fs if x.startswith(f + ":")]
        if near:
            print(
                f"      same file, current lines: {', '.join(sorted(x.rpartition(':')[2] for x in near))}"
            )
    print(
        "\nFix with: disposition.py set <correct file:line> <state> ... then remove the stale key by hand."
    )


p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
sub = p.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("set", help="record what was done about a finding")
s.add_argument("key")
s.add_argument("state")
s.add_argument("--pr")
s.add_argument("--issue")
s.add_argument("--of", help="for duplicate: the other file:line")
s.add_argument(
    "--severity",
    help="override the audit's severity (high|medium|low); requires -m with the reason",
)
s.add_argument("-m", "--message")
s.set_defaults(fn=cmd_set)
s = sub.add_parser("list", help="counts, or all findings in one state")
s.add_argument("state", nargs="?")
s.set_defaults(fn=cmd_list)
s = sub.add_parser("show", help="full finding + its disposition")
s.add_argument("key")
s.set_defaults(fn=cmd_show)
sub.add_parser("sync", help="query GitHub; flip merged/closed PRs").set_defaults(
    fn=cmd_sync
)
sub.add_parser(
    "check", help="find dispositions that match no current finding"
).set_defaults(fn=cmd_check)
a = p.parse_args()
a.fn(a)
