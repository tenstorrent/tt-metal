#!/usr/bin/env python3
"""Append-only bring-up log with staleness checking.

File: bringup/<model>/LOG.md (any path via --log). One entry per section, newest last, never edited in place:

    ## E261006-1843-a3f
    - date: 2026-10-06T18:43:02
    - commit: a9593c4477a
    - branch: ssinghal/mimo
    - host: 45
    - track: prefill
    - status: verified            (verified = measured/observed by the writing agent at this commit; inferred = reasoned, copied, or not re-measured)
    - claim: B=16 ISL 3720 TTFT 9358 ms, 6360 tok/s
    - evidence: /mnt/.../logs/grid_b16_G4k.log
    - paths: models/demos/foo/tt, models/demos/foo/demo   (files the claim depends on; a change to any of them since `commit` makes the entry STALE)
    - supersedes: E261005-1100-b12      (optional, comma list)
    - data: {"ttft_ms": 9358}           (optional JSON, machine-readable numbers; scorecard.py reads this)

RULE: an agent must not quote a number from a STALE, SUPERSEDED or inferred entry as a fact. Re-measure and `add` a new entry
(--supersedes the old id), or label the number "stale/inferred" explicitly. `check` and `show --fresh-only` implement that rule.

Commands: add | check | show.  `python3 log_tool.py <cmd> --help`.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitlib  # noqa: E402

HEADER = "# Bring-up LOG (append-only; see models/common/bringup_kit/README.md)\n"
FIELDS = ["date", "commit", "branch", "host", "track", "status", "claim", "evidence", "paths", "supersedes", "data"]
ENTRY_RE = re.compile(r"^## (E[\w-]+)\s*$")
FIELD_RE = re.compile(r"^- (\w+): ?(.*)$")


def parse(text):
    entries, cur = [], None
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if m:
            cur = {"id": m.group(1)}
            entries.append(cur)
            continue
        m = FIELD_RE.match(line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip()
    for e in entries:
        e["paths_list"] = [p.strip() for p in e.get("paths", "").split(",") if p.strip()]
        e["supersedes_list"] = [p.strip() for p in e.get("supersedes", "").split(",") if p.strip()]
        try:
            e["data_obj"] = json.loads(e["data"]) if e.get("data") else {}
        except ValueError:
            e["data_obj"] = {}
    return entries


def load(path):
    return parse(open(path).read()) if os.path.exists(path) else []


def entry_state(e, repo, entries, ref="HEAD"):
    """-> (state, reasons). state in FRESH | STALE | SUPERSEDED. Quotable as a fact = FRESH and status == verified."""
    reasons = []
    by = [x["id"] for x in entries if e["id"] in x["supersedes_list"]]
    if by:
        return "SUPERSEDED", [f"superseded by {','.join(by)}"]
    c = e.get("commit", "")
    if not kitlib.resolve(repo, c):
        reasons.append(f"commit {c} unknown to this repo")
    elif not kitlib.is_ancestor(repo, c, ref):
        reasons.append(f"commit {c} is not an ancestor of {ref}")
    else:
        ch = kitlib.changed_paths(repo, c, e["paths_list"], ref) if e["paths_list"] else []
        if ch:
            reasons.append(f"{len(ch)} file(s) changed since {c}: {', '.join(ch[:3])}{' ...' if len(ch) > 3 else ''}")
    return ("STALE", reasons) if reasons else ("FRESH", [])


def quotable(e, state):
    return state == "FRESH" and e.get("status") == "verified"


def new_id(claim, commit):
    now = datetime.datetime.now()
    h = hashlib.sha1(f"{claim}{commit}{now.isoformat()}{os.getpid()}".encode()).hexdigest()[:3]
    return f"E{now:%y%m%d-%H%M}-{h}"


def cmd_add(a):
    repo = a.repo
    log_rel_dirty_ignore = [os.path.relpath(os.path.abspath(a.log), kitlib.git(repo, "rev-parse", "--show-toplevel"))]
    dirty = kitlib.dirty_files(repo, ignore_prefixes=log_rel_dirty_ignore + list(a.ignore_dirty))
    if dirty and not a.allow_dirty:
        kitlib.die(
            f"tree is dirty ({', '.join(dirty[:5])}): commit first so the entry can name a real commit (or --allow-dirty, status forced to inferred)"
        )
    status = "inferred" if dirty else a.status
    if a.commit:
        full = kitlib.resolve(repo, a.commit)
        if not full:
            kitlib.die(f"--commit {a.commit} does not exist")
        commit = kitlib.git(repo, "rev-parse", "--short=11", full)
    else:
        br = a.branch or kitlib.git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        tip = kitlib.resolve(repo, br)
        if tip is None or tip != kitlib.resolve(repo, "HEAD"):
            kitlib.die(
                f"HEAD is not the tip of branch {br}: check out the commit you measured, or pass --commit <existing hash>"
            )
        commit = kitlib.head(repo)
    branch = a.branch or kitlib.git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if status == "verified":
        if not a.evidence:
            kitlib.die("a verified entry needs --evidence <path to log/json/test output>")
        if not os.path.exists(a.evidence):
            kitlib.die(f"evidence path {a.evidence} does not exist (use --status inferred if there is none)")
        if not a.paths:
            kitlib.die("a verified entry needs --paths (files the claim depends on; use '.' for the whole tree)")
    if a.data:
        json.loads(a.data)
    existing = load(a.log)
    ids = {e["id"] for e in existing}
    for s in a.supersedes:
        if s not in ids:
            kitlib.die(f"--supersedes {s}: no such entry in {a.log}")
    eid = new_id(a.claim, commit)
    vals = {
        "date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": commit,
        "branch": branch,
        "host": a.host or socket.gethostname().split(".")[0],
        "track": a.track,
        "status": status,
        "claim": " ".join(a.claim.split()),
        "evidence": a.evidence or "-",
        "paths": ", ".join(a.paths),
        "supersedes": ", ".join(a.supersedes),
        "data": a.data or "",
    }
    block = f"\n## {eid}\n" + "".join(f"- {k}: {vals[k]}\n" for k in FIELDS if vals[k] != "")
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(HEADER)
        f.write(block)
    print(eid)
    return 0


def _states(a):
    entries = load(a.log)
    return entries, {e["id"]: entry_state(e, a.repo, entries, a.ref) for e in entries}


def cmd_check(a):
    entries, st = _states(a)
    bad = 0
    for e in entries:
        s, why = st[e["id"]]
        tag = s if e.get("status") == "verified" or s != "FRESH" else "FRESH(inferred: do not quote)"
        print(
            f"{e['id']:<18} {tag:<8} {e.get('track','-'):<10} {e.get('claim','')[:70]}"
            + (f"  <- {'; '.join(why)}" if why else "")
        )
        bad += s != "FRESH"
    print(
        f"{len(entries)} entries, {bad} stale/superseded. Do not quote numbers from STALE, SUPERSEDED or inferred entries as facts."
    )
    return 1 if bad and a.strict else 0


def cmd_show(a):
    entries, st = _states(a)
    for e in entries:
        s, why = st[e["id"]]
        if a.fresh_only and s != "FRESH":
            continue
        if a.verified_only and e.get("status") != "verified":
            continue
        if a.track and e.get("track") != a.track:
            continue
        note = "" if quotable(e, s) else "   [NOT QUOTABLE as fact: " + (s if s != "FRESH" else "inferred") + "]"
        print(
            f"{e['id']}  {e.get('date','')}  {e.get('commit','')}  {e.get('host','')}  {e.get('track','')}  {e.get('status','')}\n    {e.get('claim','')}\n    evidence: {e.get('evidence','')}{note}"
        )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--log", required=True, help="path of LOG.md, e.g. bringup/mimo_v26/LOG.md")
        s.add_argument("--repo", default=".", help="git worktree the log lives in (default .)")

    s = sub.add_parser("add", help="append an entry (refuses on a dirty tree or HEAD != branch tip unless --commit)")
    common(s)
    s.add_argument("--claim", required=True, help="one line: the fact, with numbers")
    s.add_argument("--track", required=True, help="e.g. correctness, prefill, decode, memory, infra")
    s.add_argument("--status", choices=["verified", "inferred"], default="verified")
    s.add_argument("--evidence", help="path to the log/json that shows it (must exist for verified)")
    s.add_argument("--paths", nargs="*", default=[], help="repo paths the claim depends on (staleness trigger)")
    s.add_argument("--supersedes", nargs="*", default=[], help="ids of older entries this replaces")
    s.add_argument("--data", help="JSON with the machine-readable numbers (read by scorecard.py)")
    s.add_argument("--host", help="host the measurement ran on (default: this host)")
    s.add_argument("--branch", help="branch whose tip must equal HEAD (default: current)")
    s.add_argument("--commit", help="measured commit if it is not HEAD (must exist)")
    s.add_argument("--allow-dirty", action="store_true", help="allow a dirty tree; forces status inferred")
    s.add_argument(
        "--ignore-dirty",
        nargs="*",
        default=[],
        help="path prefixes ignored by the dirty check (the LOG file itself always is)",
    )
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("check", help="flag STALE/SUPERSEDED entries")
    common(s)
    s.add_argument("--ref", default="HEAD")
    s.add_argument("--strict", action="store_true", help="exit 1 if any entry is stale or superseded")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("show", help="print entries")
    common(s)
    s.add_argument("--ref", default="HEAD")
    s.add_argument("--fresh-only", action="store_true", help="hide stale and superseded entries")
    s.add_argument("--verified-only", action="store_true")
    s.add_argument("--track")
    s.set_defaults(fn=cmd_show)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
