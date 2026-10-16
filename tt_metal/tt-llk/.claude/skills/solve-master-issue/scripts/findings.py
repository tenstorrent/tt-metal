#!/usr/bin/env python3
"""findings.py — a local findings file as the issue source (instead of a GitHub master issue).

Formats:
  JSON   a list of records (or {"findings": [...]}) with at least file, line, summary; optional severity,
         category, evidence, failure_scenario, suggested_fix, duplicate_of, also_at, disposition.
         This is the bug-audit skill's CONFIRMED.json. Work state is written into record["disposition"]
         (a dict; existing keys such as severity notes are kept).
  MD     a table with a `Status` column and a `Location` column holding `path:line` in backticks
         (the bug-audit skill's OPEN.md). Work state replaces the Status cell.

Item key = `<file>:<line>` with any absolute worktree prefix stripped (so the key is stable across
checkouts). Records whose duplicate_of names a key are that key's duplicates: they are never worked on
their own and receive every mark their canonical gets.

Subcommands
  list  <file> [--all]            open items as JSON lines (canonical only; --all adds state, dups, filed)
  mark  <file> <key>... --state S [--issue URL] [--pr URL] [--branch B] [--worktree W] [--sha SHA]
                                  [--group G] [--note TEXT]     write state (+ dups), atomically
  reconcile <file> [--repo o/r]   GitHub vs file: items with a PR/issue carrying `audit-key: <key>`
                                  but no mark → marked; marked PRs closed unmerged → reported
  sync  <file> [--sync-cmd CMD] [-m MSG]
                                  commit + push the file if it is inside a git repo with an upstream;
                                  else run CMD (e.g. the pack-and-push script of a mirror repo);
                                  else say it is local-only. Serialized with a lock, safe from any step.

States: in-progress · issue · pr · done · skipped · needs-user · already_filed · filed_closed
(the last two come from the bug-audit filed check and are never worked here).
Open = canonical record with no state. `--all` lists everything with its state.
"""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

PREFIX_RE = re.compile(
    r"^/.*?/(?=(tt_metal|ttnn|models|tests|tt-llk|tt_llk_[a-z0-9_]+)/)"
)
WORK_STATES = {"in-progress", "issue", "pr", "done", "skipped", "needs-user"}
FILED_STATES = {"already_filed", "filed_closed"}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def norm(path):
    return PREFIX_RE.sub("", path.strip().strip("`"))


def key_of(rec):
    """`file:line`, plus `#n` when several distinct records sit on one line (the audit's same_line case).
    Suffixes are assigned by assign_keys() in file order; a dup's duplicate_of names the bare key.
    """
    return rec.get("_key") or f"{norm(rec['file'])}:{rec['line']}"


def assign_keys(recs):
    seen = {}
    for r in recs:
        k = f"{norm(r['file'])}:{r['line']}"
        n = seen.get(k, 0) + 1
        seen[k] = n
        r["_key"] = k if n == 1 else f"{k}#{n}"
    return recs


def strip_keys(recs):
    for r in recs:
        r.pop("_key", None)


# ---------------------------------------------------------------- JSON files
def load_json(path):
    with open(path) as fh:
        data = json.load(fh)
    recs = data["findings"] if isinstance(data, dict) and "findings" in data else data
    if not isinstance(recs, list):
        sys.exit(f"{path}: expected a list of records")
    return data, assign_keys(recs)


def state_of(rec):
    d = rec.get("disposition")
    if isinstance(d, dict):
        return d.get("state")
    if isinstance(d, str) and d:
        return d
    return None


def save_atomic(path, text):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".findings-", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------ MD files
MD_ROW = re.compile(r"^\|")


def md_table(lines):
    """(header_idx, columns) of the first table that has Status + Location columns."""
    for i, ln in enumerate(lines):
        if MD_ROW.match(ln) and "Status" in ln and "Location" in ln:
            cols = [c.strip() for c in ln.strip().strip("|").split("|")]
            return i, cols
    sys.exit("no table with Status and Location columns")


def md_rows(lines):
    hi, cols = md_table(lines)
    si, li = cols.index("Status"), cols.index("Location")
    rows = []
    for i in range(hi + 2, len(lines)):
        if not MD_ROW.match(lines[i]):
            break
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) < len(cols):
            continue
        locs = re.findall(r"`([^`]+:\d+)`", cells[li])
        if not locs:
            continue
        rec = {
            "file": locs[0].rsplit(":", 1)[0],
            "line": int(locs[0].rsplit(":", 1)[1]),
        }
        rec["also_at"] = locs[1:]
        for c, v in zip(cols, cells):
            if c not in ("Status", "Location"):
                rec[c.lower()] = v
        rows.append((i, si, cells, rec, cells[si]))
    return rows


# ----------------------------------------------------------------- commands
def cmd_list(a):
    if a.file.endswith(".json"):
        _, recs = load_json(a.file)
        dups = {}
        for r in recs:
            if r.get("duplicate_of"):
                dups.setdefault(norm(r["duplicate_of"]), []).append(key_of(r))
        for r in recs:
            k = key_of(r)
            if r.get("duplicate_of"):
                continue
            st = state_of(r)
            if not a.all and st is not None:
                continue
            out = {
                "key": k,
                "state": st,
                "severity": r.get("severity"),
                "category": r.get("category"),
                "summary": r.get("summary"),
                "failure_scenario": r.get("failure_scenario"),
                "suggested_fix": r.get("suggested_fix"),
                "evidence": r.get("evidence"),
                "also_at": r.get("also_at") or [],
                "duplicates": dups.get(k.split("#")[0], []) if "#" not in k else [],
                "disposition": (
                    r.get("disposition")
                    if isinstance(r.get("disposition"), dict)
                    else None
                ),
            }
            print(json.dumps(out))
    else:
        lines = open(a.file).read().splitlines()
        for _, _, _, rec, status in md_rows(lines):
            st = None if status.lower() in ("open", "") else status
            if not a.all and st is not None:
                continue
            print(
                json.dumps(
                    {
                        "key": key_of(rec),
                        "state": st,
                        **{k: v for k, v in rec.items() if k not in ("file", "line")},
                    }
                )
            )


def cmd_mark(a):
    if a.state not in WORK_STATES:
        sys.exit(f"state must be one of {sorted(WORK_STATES)}")
    keys = set()
    for k in a.keys:
        k, _, suf = k.partition("#")
        path, _, line = k.rpartition(":")
        keys.add(f"{norm(path)}:{line}" + (f"#{suf}" if suf else ""))
    bare = {k.split("#")[0] for k in keys}
    fields = {
        k: v
        for k, v in (
            ("issue", a.issue),
            ("pr", a.pr),
            ("branch", a.branch),
            ("worktree", a.worktree),
            ("sha", a.sha),
            ("group", a.group),
            ("note", a.note),
        )
        if v
    }
    fields["host"] = os.uname().nodename
    with lock(a.file):
        if a.file.endswith(".json"):
            data, recs = load_json(a.file)
            hit = 0
            for r in recs:
                k = key_of(r)
                if k in keys or (
                    r.get("duplicate_of") and norm(r["duplicate_of"]) in bare
                ):
                    d = (
                        r.get("disposition")
                        if isinstance(r.get("disposition"), dict)
                        else {}
                    )
                    if d.get("state") in FILED_STATES and a.state != "skipped":
                        d["note_prev_state"] = d["state"]
                    d.update(fields)
                    d["state"] = a.state
                    d["updated"] = now()
                    r["disposition"] = d
                    hit += 1
            strip_keys(recs)
            save_atomic(a.file, json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        else:
            lines = open(a.file).read().splitlines()
            hit = 0
            links = " ".join(v for k, v in fields.items() if k in ("issue", "pr"))
            cell = f"{a.state} {links}".strip()
            for i, si, cells, rec, _ in md_rows(lines):
                if key_of(rec) in keys:
                    cells[si] = cell
                    lines[i] = "| " + " | ".join(cells) + " |"
                    hit += 1
            save_atomic(a.file, "\n".join(lines) + "\n")
    missing = len(keys) - min(hit, len(keys))
    print(
        f"marked {hit} record(s) {a.state}"
        + (f"; {missing} key(s) not found" if hit < len(keys) else "")
    )
    if hit == 0:
        sys.exit(1)


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"gh {' '.join(args[:3])}… failed: {r.stderr.strip()[:300]}")
    return r.stdout


def cmd_reconcile(a):
    repo = a.repo or os.environ.get("GH_REPO", "tenstorrent/tt-metal")
    me = gh("api", "user", "-q", ".login").strip()
    items = {}
    for ln in subprocess.run(
        [sys.executable, "-I", __file__, "list", a.file, "--all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines():
        o = json.loads(ln)
        items[o["key"]] = o
    tok = re.compile(r"audit-key:\s*`?([^\s`]+:\d+)`?")
    prs = json.loads(
        gh(
            "pr",
            "list",
            "--repo",
            repo,
            "--author",
            me,
            "--state",
            "all",
            "--limit",
            "300",
            "--json",
            "number,state,url,body,title,headRefName,mergedAt",
        )
    )
    issues = json.loads(
        gh(
            "api",
            "--paginate",
            f"search/issues?q=author:{me}+repo:{repo}+is:issue+%22audit-key%22&per_page=100",
            "-q",
            "[.items[]|{number,state,html_url,body,title}]",
        )
    )
    if issues and isinstance(issues[0], list):  # --paginate concatenates arrays
        issues = [x for page in issues for x in page]
    drift = []
    for kind, rows in (("pr", prs), ("issue", issues)):
        for row in rows:
            for k in set(
                tok.findall((row.get("body") or "") + " " + (row.get("title") or ""))
            ):
                it = items.get(k)
                if it is None:
                    continue
                url = row.get("url") or row.get("html_url")
                st = it.get("state")
                disp = it.get("disposition") or {}
                if kind == "pr":
                    closed_unmerged = row["state"] == "CLOSED" and not row.get(
                        "mergedAt"
                    )
                    if closed_unmerged:
                        if disp.get("pr") == url or st in (None, "in-progress", "pr"):
                            drift.append(
                                (
                                    "REPORT",
                                    k,
                                    f"PR {url} closed without merge — decide, do not re-open",
                                )
                            )
                        continue
                    if disp.get("pr") != url:
                        drift.append(("MARK", k, "pr", url, row["headRefName"]))
                else:
                    if disp.get("issue") != url and st not in ("pr", "done"):
                        drift.append(("MARK", k, "issue", url, None))
    for d in drift:
        if d[0] == "MARK":
            _, k, kind, url, branch = d
            args = [
                sys.executable,
                "-I",
                __file__,
                "mark",
                a.file,
                k,
                "--state",
                kind,
                f"--{kind}",
                url,
                "--note",
                "reconciled from GitHub on resume",
            ]
            if branch:
                args += ["--branch", branch]
            subprocess.run(args, check=False)
            print(f"RECONCILED {k} -> {kind} {url}")
        else:
            print(f"{d[0]} {d[1]}: {d[2]}")
    stale = [k for k, it in items.items() if it.get("state") == "in-progress"]
    for k in stale:
        d = items[k].get("disposition") or {}
        print(
            f"IN-PROGRESS {k}: branch={d.get('branch')} worktree={d.get('worktree')} host={d.get('host')} "
            f"since {d.get('updated')} — resume it from the worktree if it exists, else restart it"
        )
    if not drift and not stale:
        print("consistent: file marks match GitHub")


def lock(path):
    lf = open(os.path.abspath(path) + ".lock", "w")
    fcntl.flock(lf, fcntl.LOCK_EX)
    return lf


def cmd_sync(a):
    path = os.path.abspath(a.file)
    d = os.path.dirname(path)
    msg = a.message or f"solve-master-issue: update {os.path.basename(path)} {now()}"
    with lock(path):
        top = subprocess.run(
            ["git", "-C", d, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if top.returncode == 0:
            subprocess.run(["git", "-C", d, "add", "--", path], check=True)
            if (
                subprocess.run(
                    ["git", "-C", d, "diff", "--cached", "--quiet", "--", path]
                ).returncode
                == 0
            ):
                print("no changes")
                return
            subprocess.run(
                ["git", "-C", d, "commit", "-q", "-m", msg, "--", path], check=True
            )
            up = subprocess.run(
                [
                    "git",
                    "-C",
                    d,
                    "rev-parse",
                    "--abbrev-ref",
                    "--symbolic-full-name",
                    "@{u}",
                ],
                capture_output=True,
                text=True,
            )
            if up.returncode:
                print("committed; no upstream configured — not pushed")
                return
            remote, _, branch = up.stdout.strip().partition("/")
            r = subprocess.run(
                ["git", "-C", d, "push", "-q", remote, f"HEAD:{branch}"],
                capture_output=True,
                text=True,
            )
            print(
                "PUSH_OK"
                if r.returncode == 0
                else f"push failed: {r.stderr.strip()[:300]}"
            )
            if r.returncode:
                sys.exit(1)
        elif a.sync_cmd:
            r = subprocess.run(
                a.sync_cmd + " " + json.dumps(msg),
                shell=True,
                capture_output=True,
                text=True,
            )
            print((r.stdout.strip() or r.stderr.strip())[-400:])
            if r.returncode:
                sys.exit(1)
        else:
            print(f"{path} is not in a git repo and no --sync-cmd given: local only")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("list")
    s.add_argument("file")
    s.add_argument("--all", action="store_true")
    s.set_defaults(f=cmd_list)
    s = sp.add_parser("mark")
    s.add_argument("file")
    s.add_argument("keys", nargs="+")
    s.add_argument("--state", required=True)
    for o in ("issue", "pr", "branch", "worktree", "sha", "group", "note"):
        s.add_argument(f"--{o}")
    s.set_defaults(f=cmd_mark)
    s = sp.add_parser("reconcile")
    s.add_argument("file")
    s.add_argument("--repo")
    s.set_defaults(f=cmd_reconcile)
    s = sp.add_parser("sync")
    s.add_argument("file")
    s.add_argument("--sync-cmd")
    s.add_argument("-m", "--message")
    s.set_defaults(f=cmd_sync)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
