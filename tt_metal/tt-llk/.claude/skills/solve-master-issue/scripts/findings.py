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
  list  <file> [--all]            workable items as JSON lines: no state yet, or `planned` (canonical
                                  only; --all adds every state incl. filed; duplicates ride along, never listed). Each line carries the
                                  group / arch / owner written by the plan, and an `arch_guess` from paths.
  mark  <file> <key>... --state S [--issue URL] [--pr URL] [--branch B] [--worktree W] [--sha SHA]
                                  [--group G] [--arch A] [--owner O] [--note TEXT]
                                  write state (+ dups), atomically
  groups <file> [--claims DIR]    one JSON line per group: items, states, arch, owner, who holds the claim
  next  <file> --owner O --arch <arch>[,<arch>] --claims DIR [-n N] [--take-other]
                                  the next unclaimed planned groups this worker should take, in order
                                  (--arch = the architectures this box can run; any names the harness
                                  takes: wormhole, blackhole, quasar, …; several boxes per arch are fine)
  claim <file> <group> --owner O --claims DIR
                                  atomic claim of a group (mkdir on the shared dir): marks its planned
                                  items in-progress with --owner; exit 1 and TAKEN if another worker has it
  release <file> <group> --claims DIR
                                  give a claimed group back (items without a PR return to planned)
  reconcile <file> [--repo o/r]   GitHub vs file: items with a PR/issue carrying `audit-key: <key>`
                                  but no mark → marked; marked PRs closed unmerged → reported
  sync  <file> [--sync-cmd CMD] [-m MSG]
                                  commit + push the file if it is inside a git repo with an upstream
                                  (pull --rebase first); else run CMD (e.g. the pack-and-push script of a
                                  mirror repo); else say it is local-only. Serialized with its own lock,
                                  so marks are never blocked behind a slow push.

States: planned · in-progress · issue · pr · done · skipped · needs-user · already_filed · filed_closed
(the last two come from the bug-audit filed check and are never worked here).
Workable = canonical record with no state or `planned`. `--all` lists everything with its state.

Several workers (one per box) share the file through a directory every box mounts: the plan phase marks
groups (`planned` + --group + --arch), each worker takes groups with `next` → `claim`, and the claim
directory (`--claims`, normally `$MASTER_ISSUE_DIR/<run>/claims`) is the arbiter: `mkdir` is atomic on
POSIX filesystems including network mounts, so two workers can never hold one group. Locks on the file
use the same mkdir primitive (`<file>.lock.d`), with stale breaking after LOCK_STALE_S.
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

PREFIX_RE = re.compile(
    r"^/.*?/(?=(tt_metal|ttnn|models|tests|tt-llk|tt_llk_[a-z0-9_]+)/)"
)
WORK_STATES = {
    "planned",
    "in-progress",
    "issue",
    "pr",
    "done",
    "skipped",
    "needs-user",
}
FILED_STATES = {"already_filed", "filed_closed"}
WORKABLE = {None, "planned"}
ARCH_RE = re.compile(
    r"^[a-z][a-z0-9_]*$"
)  # architecture names are whatever the harness takes
LOCK_STALE_S = int(os.environ.get("FINDINGS_LOCK_STALE_S", "900"))
LOCK_WAIT_S = int(os.environ.get("FINDINGS_LOCK_WAIT_S", "600"))


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


ARCH_DIR_RE = re.compile(
    r"(?:^|/)tt_llk_([a-z0-9]+?)(?:_b\d)?(?:/|$)"
)  # tt_llk_wormhole_b0 -> wormhole


def arch_set(text):
    """Parse an arch spec: 'wormhole', 'blackhole,wormhole', 'neutral' -> a set (empty = neutral)."""
    if not text or text == "neutral":
        return set()
    out = set()
    for t in text.split(","):
        t = t.strip()
        if t == "neutral" or not t:
            continue
        if not ARCH_RE.match(t):
            sys.exit(f"bad architecture name {t!r} (lowercase letters, digits, _)")
        out.add(t)
    return out


def arch_str(archs):
    return ",".join(sorted(archs)) if archs else "neutral"


def arch_guess(rec):
    """From the paths alone: the tt_llk_<arch> trees the sites live in (comma-joined), or neutral."""
    paths = [rec.get("file", "")]
    for x in rec.get("also_at") or []:
        paths.append(x if isinstance(x, str) else x.get("file", ""))
    a = set()
    for p in paths:
        m = ARCH_DIR_RE.search(p)
        if m:
            a.add(m.group(1))
        else:
            for name in ("wormhole", "blackhole", "quasar"):
                if name in p:
                    a.add(name)
    return arch_str(a)


# ---------------------------------------------------------------- JSON files
def load_json(path):
    with open(path) as fh:
        data = json.load(fh)
    recs = data["findings"] if isinstance(data, dict) and "findings" in data else data
    if not isinstance(recs, list):
        sys.exit(f"{path}: expected a list of records")
    return data, assign_keys(recs)


def disp(rec):
    d = rec.get("disposition")
    return d if isinstance(d, dict) else {}


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


def save_json(path, data, recs):
    strip_keys(recs)
    save_atomic(path, json.dumps(data, indent=1, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- locking
class DirLock:
    """mkdir-based lock: atomic on every POSIX filesystem incl. NFS/weka (flock is not, across hosts).
    Holder info in <dir>/owner; a lock older than LOCK_STALE_S is broken (its holder died mid-step).
    """

    def __init__(self, path, stale=LOCK_STALE_S, wait=LOCK_WAIT_S):
        self.dir = path
        self.stale = stale
        self.wait = wait

    def __enter__(self):
        deadline = time.time() + self.wait
        while True:
            try:
                os.mkdir(self.dir)
                break
            except FileExistsError:
                try:
                    age = time.time() - os.stat(self.dir).st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale:
                    try:
                        who = open(os.path.join(self.dir, "owner")).read().strip()
                    except OSError:
                        who = "?"
                    print(
                        f"breaking stale lock {self.dir} ({int(age)}s old, {who})",
                        file=sys.stderr,
                    )
                    shutil.rmtree(self.dir, ignore_errors=True)
                    continue
                if time.time() > deadline:
                    sys.exit(f"could not take {self.dir} within {self.wait}s")
                time.sleep(0.5 + (os.getpid() % 10) / 20)
        with open(os.path.join(self.dir, "owner"), "w") as fh:
            fh.write(f"{socket.gethostname()} pid={os.getpid()} {now()}\n")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)


def lock(path):
    return DirLock(os.path.abspath(path) + ".lock.d")


def sync_lock(path):
    return DirLock(os.path.abspath(path) + ".sync.lock.d", wait=3600)


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
def item_view(r, k, dups):
    d = disp(r)
    return {
        "key": k,
        "state": state_of(r),
        "group": d.get("group"),
        "arch": d.get("arch"),
        "owner": d.get("owner"),
        "arch_guess": arch_guess(r),
        "severity": r.get("severity"),
        "category": r.get("category"),
        "summary": r.get("summary"),
        "failure_scenario": r.get("failure_scenario"),
        "suggested_fix": r.get("suggested_fix"),
        "evidence": r.get("evidence"),
        "also_at": r.get("also_at") or [],
        "duplicates": dups.get(k.split("#")[0], []) if "#" not in k else [],
        "disposition": d or None,
    }


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
            if not a.all and st not in WORKABLE:
                continue
            print(json.dumps(item_view(r, k, dups)))
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


def parse_keys(keys):
    out = set()
    for k in keys:
        k, _, suf = k.partition("#")
        path, _, line = k.rpartition(":")
        out.add(f"{norm(path)}:{line}" + (f"#{suf}" if suf else ""))
    return out


def apply_mark(recs, keys, state, fields):
    """Set state+fields on the keyed records and their duplicates; returns the hit count."""
    bare = {k.split("#")[0] for k in keys}
    hit = 0
    for r in recs:
        k = key_of(r)
        if k in keys or (r.get("duplicate_of") and norm(r["duplicate_of"]) in bare):
            d = disp(r)
            if d.get("state") in FILED_STATES and state != "skipped":
                d["note_prev_state"] = d["state"]
            d.update(fields)
            d["state"] = state
            d["updated"] = now()
            r["disposition"] = d
            hit += 1
    return hit


def cmd_mark(a):
    if a.state not in WORK_STATES:
        sys.exit(f"state must be one of {sorted(WORK_STATES)}")
    if a.arch:
        a.arch = arch_str(
            arch_set(a.arch)
        )  # canonical: sorted, comma-joined, or neutral
    keys = parse_keys(a.keys)
    fields = {
        k: v
        for k, v in (
            ("issue", a.issue),
            ("pr", a.pr),
            ("branch", a.branch),
            ("worktree", a.worktree),
            ("sha", a.sha),
            ("group", a.group),
            ("arch", a.arch),
            ("owner", a.owner),
            ("note", a.note),
        )
        if v
    }
    fields["host"] = os.uname().nodename
    with lock(a.file):
        if a.file.endswith(".json"):
            data, recs = load_json(a.file)
            hit = apply_mark(recs, keys, a.state, fields)
            save_json(a.file, data, recs)
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


# ------------------------------------------------------------ groups/claims
def group_table(recs):
    """group -> {items, states, arch, owner} over canonical records that carry a group."""
    g = {}
    for r in recs:
        if r.get("duplicate_of"):
            continue
        d = disp(r)
        if not d.get("group"):
            continue
        e = g.setdefault(
            d["group"],
            {
                "group": d["group"],
                "items": [],
                "states": {},
                "arch": set(),
                "owner": None,
            },
        )
        k = key_of(r)
        e["items"].append(k)
        e["states"][k] = d.get("state")
        e["arch"] |= arch_set(d.get("arch") or arch_guess(r))
        e["owner"] = e["owner"] or d.get("owner")
    for e in g.values():
        e["arch"] = arch_str(e["arch"])  # the union over the group's items
    return g


def claim_owner(claims, group):
    if not claims or not os.path.isdir(os.path.join(claims, group)):
        return None
    try:
        with open(os.path.join(claims, group, "owner.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"owner": "?"}


def cmd_groups(a):
    _, recs = load_json(a.file)
    for e in group_table(recs).values():
        c = claim_owner(a.claims, e["group"])
        e["claimed_by"] = (c or {}).get("owner")
        print(json.dumps(e))


def rank_for(group_arch, mine):
    """0 = every arch the group needs is on this box · 1 = neutral (any box) · 2 = some legs here, the
    rest remote · 3 = nothing here (every leg remote; left to a box that has it unless --take-other).
    """
    need = arch_set(group_arch)
    if not need:
        return 1
    if need <= mine:
        return 0
    return 2 if need & mine else 3


def spread_key(owner, group):
    """Per-worker order inside a rank: a stable hash of worker+group, so n workers on the same
    architecture walk the same rank in different orders and seldom try one group at the same time.
    """
    import hashlib

    return hashlib.sha1(f"{owner}\0{group}".encode()).hexdigest()


def cmd_next(a):
    mine = arch_set(a.arch)
    if not mine:
        sys.exit("--arch must name at least one architecture this box can run")
    _, recs = load_json(a.file)
    cands = []
    for e in group_table(recs).values():
        if any(s != "planned" for s in e["states"].values()):
            continue  # already started by someone (or finished)
        if claim_owner(a.claims, e["group"]):
            continue
        r = rank_for(e["arch"], mine)
        if r == 3 and not a.take_other:
            continue
        cands.append((r, e))
    cands.sort(key=lambda t: (t[0], spread_key(a.owner, t[1]["group"])))
    for r, e in cands[: a.n]:
        e["rank"] = r
        print(json.dumps(e))


def cmd_claim(a):
    os.makedirs(a.claims, exist_ok=True)
    cdir = os.path.join(a.claims, a.group)
    try:
        os.mkdir(cdir)
    except FileExistsError:
        who = claim_owner(a.claims, a.group) or {}
        print(
            f"TAKEN {a.group} by {who.get('owner', '?')} ({who.get('host', '?')}, {who.get('time', '?')})"
        )
        sys.exit(1)
    with open(os.path.join(cdir, "owner.json"), "w") as fh:
        json.dump(
            {
                "owner": a.owner,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "time": now(),
                "group": a.group,
            },
            fh,
        )
    with lock(a.file):
        data, recs = load_json(a.file)
        keys = {
            key_of(r)
            for r in recs
            if not r.get("duplicate_of")
            and disp(r).get("group") == a.group
            and state_of(r) in WORKABLE
        }
        hit = apply_mark(
            recs,
            keys,
            "in-progress",
            {"owner": a.owner, "host": os.uname().nodename, "group": a.group},
        )
        save_json(a.file, data, recs)
    print(f"CLAIMED {a.group} for {a.owner}: {hit} record(s) in-progress")
    if hit == 0:
        print("warning: the group had no planned items", file=sys.stderr)


def cmd_release(a):
    cdir = os.path.join(a.claims, a.group)
    with lock(a.file):
        data, recs = load_json(a.file)
        keys = {
            key_of(r)
            for r in recs
            if not r.get("duplicate_of")
            and disp(r).get("group") == a.group
            and state_of(r) == "in-progress"
            and not disp(r).get("pr")
        }
        hit = apply_mark(recs, keys, "planned", {"note": "released"})
        for r in recs:
            d = disp(r)
            if d.get("group") == a.group and d.get("state") == "planned":
                d.pop("owner", None)
        save_json(a.file, data, recs)
    shutil.rmtree(cdir, ignore_errors=True)
    print(f"RELEASED {a.group}: {hit} record(s) back to planned")


# -------------------------------------------------------------- reconcile
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
                d = it.get("disposition") or {}
                if kind == "pr":
                    closed_unmerged = row["state"] == "CLOSED" and not row.get(
                        "mergedAt"
                    )
                    if closed_unmerged:
                        if d.get("pr") == url or st in (
                            None,
                            "planned",
                            "in-progress",
                            "pr",
                        ):
                            drift.append(
                                (
                                    "REPORT",
                                    k,
                                    f"PR {url} closed without merge — decide, do not re-open",
                                )
                            )
                        continue
                    if d.get("pr") != url:
                        drift.append(("MARK", k, "pr", url, row["headRefName"]))
                else:
                    if d.get("issue") != url and st not in ("pr", "done"):
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
            f"IN-PROGRESS {k}: owner={d.get('owner')} host={d.get('host')} branch={d.get('branch')} "
            f"worktree={d.get('worktree')} since {d.get('updated')} — the owner resumes it from the "
            f"worktree if it exists, else restarts it; another worker leaves it alone"
        )
    if not drift and not stale:
        print("consistent: file marks match GitHub")


# ------------------------------------------------------------------- sync
def cmd_sync(a):
    path = os.path.abspath(a.file)
    d = os.path.dirname(path)
    msg = a.message or f"solve-master-issue: update {os.path.basename(path)} {now()}"
    with sync_lock(path):
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
            # another box may have pushed its marks meanwhile: integrate first, then push
            r = None
            for _attempt in (1, 2):
                pull = subprocess.run(
                    ["git", "-C", d, "pull", "-q", "--rebase", remote, branch],
                    capture_output=True,
                    text=True,
                )
                if pull.returncode:
                    subprocess.run(
                        ["git", "-C", d, "rebase", "--abort"], capture_output=True
                    )
                    print(f"pull --rebase failed: {pull.stderr.strip()[:300]}")
                    sys.exit(1)
                r = subprocess.run(
                    ["git", "-C", d, "push", "-q", remote, f"HEAD:{branch}"],
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    print("PUSH_OK")
                    return
            print(f"push failed: {r.stderr.strip()[:300]}")
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
    for o in (
        "issue",
        "pr",
        "branch",
        "worktree",
        "sha",
        "group",
        "arch",
        "owner",
        "note",
    ):
        s.add_argument(f"--{o}")
    s.set_defaults(f=cmd_mark)
    s = sp.add_parser("groups")
    s.add_argument("file")
    s.add_argument("--claims")
    s.set_defaults(f=cmd_groups)
    s = sp.add_parser("next")
    s.add_argument("file")
    s.add_argument("--owner", required=True)
    s.add_argument("--arch", required=True)
    s.add_argument("--claims", required=True)
    s.add_argument("-n", type=int, default=5)
    s.add_argument("--take-other", action="store_true")
    s.set_defaults(f=cmd_next)
    s = sp.add_parser("claim")
    s.add_argument("file")
    s.add_argument("group")
    s.add_argument("--owner", required=True)
    s.add_argument("--claims", required=True)
    s.set_defaults(f=cmd_claim)
    s = sp.add_parser("release")
    s.add_argument("file")
    s.add_argument("group")
    s.add_argument("--claims", required=True)
    s.set_defaults(f=cmd_release)
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
