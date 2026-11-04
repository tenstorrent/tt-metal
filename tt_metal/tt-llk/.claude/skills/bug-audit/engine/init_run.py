#!/usr/bin/env python3
"""Create a bug-audit run: enumerate the files in scope, assign priorities, pack line-bounded batches.

  init_run.py --root /path/to/checkout --out /path/to/run-dir --repo owner/name \
      [--prio 'A=src/kernels/**,src/core/**' --prio 'B=tools/**'] [--default-prio C] \
      [--ext .c,.cc,.cpp,.h,.hpp,.py] [--include 'src/area/**'] [--exclude 'third_party/**'] \
      [--since <commit>] [--max-files 20] [--max-lines 300] \
      [--knowledge references/classes-universal.md,references/classes-<domain>.md]   # a repo pack only to re-measure it

--knowledge defaults to the universal classes, plus the Tenstorrent classes for a tenstorrent/ repo.
--pack defaults to packs/<repo name>.md when it exists: files in its hot areas that no --prio glob matched go to A.

--root must be a git checkout pinned at the commit you mean to audit (a dedicated worktree is best), so
recorded file:line findings stay valid for the life of the run. The commit is recorded in state.json.
--since limits the scope to files changed between <commit> and the audited commit (diff mode).
Priorities are glob lists matched in the order given; unmatched files get --default-prio.
Batches never mix priorities, keep directories contiguous, and hold at most --max-files files and
--max-lines lines (default 300, or the --batch-lines budget for their priority), so an agent can read, and actually
analyse, every line of every file it is assigned. A file longer than the budget is a batch of its own.
"""
import argparse
import datetime
import fnmatch
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spawn  # noqa: E402

from common import save  # noqa: E402

DEFAULT_EXT = (
    ".c,.cc,.cpp,.cxx,.h,.hh,.hpp,.hxx,.inl,.ipp,.cu,.py,.rs,.go,.java,.js,.ts"
)

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
p.add_argument("--root", required=True)
p.add_argument("--out", required=True)
p.add_argument(
    "--repo",
    required=True,
    help="owner/name on GitHub, used for dedup and dispositions",
)
p.add_argument("--name", default="")
p.add_argument(
    "--prio",
    action="append",
    default=[],
    help="LETTER=glob,glob (repeatable, first match wins)",
)
p.add_argument("--default-prio", default="C")
p.add_argument("--ext", default=DEFAULT_EXT)
p.add_argument(
    "--include",
    action="append",
    default=[],
    help="area mode: only files matching one of these globs",
)
p.add_argument("--exclude", action="append", default=[])
p.add_argument("--since", help="diff mode: only files changed since this commit")
p.add_argument("--max-files", type=int, default=20)
p.add_argument("--max-lines", type=int, default=300)
p.add_argument(
    "--batch-lines",
    default="",
    help="per-priority line budget overriding --max-lines, e.g. 'C=1500'. Less code per hunter means deeper "
    "reading: on full-size batches, hunters found real bugs at 300 lines that they missed at 800 and above",
)
p.add_argument(
    "--knowledge",
    default=None,
    help="comma list of knowledge files the hunters must read. Default: references/classes-universal.md, plus "
    "references/classes-tenstorrent.md when --repo is a tenstorrent/ repo. 'none' hands hunters no class list",
)
p.add_argument(
    "--pack",
    default=None,
    help="repo pack whose 'Hot areas' promote files no --prio glob matched to priority A. Default: "
    "packs/<repo name>.md when it exists; 'none' disables",
)
p.add_argument(
    "--recurse-submodules",
    action="store_true",
    help="audit files inside submodules too (git ls-files lists a submodule as ONE entry, so without this "
    "they are out of scope; the run records which ones)",
)
p.add_argument(
    "--prior-run",
    action="append",
    default=[],
    help="earlier run dirs: their confirmed and refuted findings for these files are handed to the hunters",
)
a = p.parse_args()

root = os.path.abspath(a.root)
out = os.path.abspath(a.out)
if os.path.exists(os.path.join(out, "state.json")):
    sys.exit(
        f"{out} already holds a run; resume it instead of re-initialising (re-init would orphan its verdicts)"
    )
# the class lists are the hunters' checklist; without them the hunt prompt names no bug classes at all
skill = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if a.knowledge is None:
    knowledge = ["references/classes-universal.md"]
    if a.repo.lower().startswith("tenstorrent/"):
        knowledge.append("references/classes-tenstorrent.md")
elif a.knowledge.strip().lower() == "none":
    knowledge = []
else:
    knowledge = [k.strip() for k in a.knowledge.split(",") if k.strip()]
missing = [
    k
    for k in knowledge
    if not os.path.isfile(k if os.path.isabs(k) else os.path.join(skill, k))
]
if missing:
    sys.exit(
        f"knowledge file(s) not found (relative paths resolve against {skill}): {', '.join(missing)}"
    )
if knowledge:
    print(f"knowledge handed to every hunter: {', '.join(knowledge)}")
else:
    print(
        "WARNING: hunters get NO bug-class list, only the generic hunt prompt",
        file=sys.stderr,
    )


def hot_areas(path):
    """The directories listed under the pack's '## Hot areas' heading (each counted by a fixed file's parent dir)."""
    m = re.search(r"^## Hot areas\n(.*?)(?=^## |\Z)", open(path).read(), re.M | re.S)
    return re.findall(r"`([^`]+)`", m.group(1)) if m else []


if a.pack is None:
    auto = os.path.join(skill, "packs", a.repo.split("/")[-1] + ".md")
    pack = auto if os.path.isfile(auto) else None
elif a.pack.strip().lower() == "none":
    pack = None
else:
    pack = a.pack if os.path.isabs(a.pack) else os.path.join(skill, a.pack)
    if not os.path.isfile(pack):
        sys.exit(f"--pack not found: {pack}")
hot = set(hot_areas(pack)) if pack else set()
if pack and not hot:
    sys.exit(f"{pack} has no '## Hot areas' list; pass --pack none to run without one")


def git(*args):
    return spawn.run(
        "git", ["-C", root, *args], check=True, capture_output=True, text=True
    ).stdout


commit = git("rev-parse", "HEAD").strip()
dirty = git("status", "--porcelain", "--untracked-files=no").strip()
if dirty:
    print(
        "WARNING: the audited tree has uncommitted changes; findings will cite lines that are not in any commit",
        file=sys.stderr,
    )

exts = tuple(e.strip() for e in a.ext.split(",") if e.strip())
# -z: without it git C-quotes any path with non-ASCII or special characters (in quotes, each such byte an octal escape), and the quoted name
# fails the extension test, so the file would silently fall out of scope
submodules = [
    ln.split("\t", 1)[1]
    for ln in git("ls-files", "-s", "-z").split("\0")
    if ln.startswith("160000")
]
listing = (
    git("ls-files", "-z", "--recurse-submodules")
    if a.recurse_submodules
    else git("ls-files", "-z")
)
files = [f for f in listing.split("\0") if f and f.endswith(exts)]
if submodules and not a.recurse_submodules:
    print(
        f"WARNING: {len(submodules)} submodule(s) are OUT of scope (each is one gitlink entry): "
        + ", ".join(submodules[:8])
        + (" ..." if len(submodules) > 8 else "")
        + ". Pass --recurse-submodules to include them, or audit each as its own run.",
        file=sys.stderr,
    )
if a.since:
    # -z for the same reason as ls-files above
    changed = {
        p
        for p in git("diff", "--name-only", "-z", f"{a.since}...{commit}").split("\0")
        if p
    }
    if a.recurse_submodules:
        # a diff names a changed submodule only by its path: diff inside it between its two pins, and recurse

        def pin(repo, rev, s):
            """The commit submodule s (a path inside repo) is pinned to at rev, or None if it is not there."""
            entry = (
                git("-C", repo, "ls-tree", "-z", rev, "--", s).split("\0")[0].split()
            )
            return entry[2] if entry else None

        def gitlinks(repo, rev):
            out_ = git("-C", repo, "ls-tree", "-r", "-z", rev)
            return {
                e.split("\t", 1)[1] for e in out_.split("\0") if e.startswith("160000 ")
            }

        def changed_in(path, old, new):
            """Changed files inside the submodule at path (from the root) between pins old and new."""
            try:
                names = (
                    git("-C", path, "diff", "--name-only", "-z", old, new)
                    if old
                    else git("-C", path, "ls-files", "-z")
                )
                nested = gitlinks(path, new) if new else set()
            except subprocess.CalledProcessError:
                # a pin is not in the submodule's clone: every file in it counts as changed
                print(
                    f"WARNING: {path}: cannot diff {(old or 'none')[:11]}..{(new or 'none')[:11]} (not fetched); "
                    "all its files are in scope",
                    file=sys.stderr,
                )
                return {f for f in files if f.startswith(path + "/")}
            got = set()
            for p in filter(None, names.split("\0")):
                if p in nested:
                    got |= changed_in(
                        f"{path}/{p}", old and pin(path, old, p), pin(path, new, p)
                    )
                else:
                    got.add(f"{path}/{p}")
            return got

        base = git("merge-base", a.since, commit).strip()
        for s in submodules:
            if s in changed:
                changed |= changed_in(s, pin(".", base, s), pin(".", commit, s))
    files = [f for f in files if f in changed]
# comma lists, like --prio: a whole list taken as one glob matches nothing, silently
include = [g.strip() for x in a.include for g in x.split(",") if g.strip()]
exclude = [g.strip() for x in a.exclude for g in x.split(",") if g.strip()]
if include:
    files = [f for f in files if any(fnmatch.fnmatch(f, g) for g in include)]
files = [f for f in files if not any(fnmatch.fnmatch(f, g) for g in exclude)]
if not files:
    sys.exit(
        "no file in scope: check --include, --exclude, --ext and --since against the tree"
    )

prio_rules = []
for spec in a.prio:
    letter, _, globs = spec.partition("=")
    prio_rules.append(
        (letter.strip(), [g.strip() for g in globs.split(",") if g.strip()])
    )


promoted = set()


def prio_of(f):
    for letter, globs in prio_rules:
        if any(fnmatch.fnmatch(f, g) for g in globs):
            return letter
    # an explicit --prio glob always wins; the pack only lifts files the user left at the default
    if os.path.dirname(f) in hot:
        promoted.add(f)
        return "A"
    return a.default_prio


def nlines(f):
    try:
        with open(os.path.join(root, f), "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


rows = sorted(((prio_of(f), f, nlines(f)) for f in files), key=lambda r: (r[0], r[1]))
if pack:
    dirs = sorted({os.path.dirname(f) for f in promoted})
    print(
        f"pack {os.path.relpath(pack, skill)}: {len(promoted)} file(s) in {len(dirs)} hot area(s) promoted to A"
        + (
            ": " + ", ".join(dirs[:6]) + (" ..." if len(dirs) > 6 else "")
            if dirs
            else ""
        )
    )
budget = {
    k.strip(): int(v)
    for k, _, v in (x.partition("=") for x in a.batch_lines.split(",") if "=" in x)
}
batches, cur, cur_lines, cur_prio = [], [], 0, None


def flush():
    global cur, cur_lines
    if cur:
        n = sum(1 for b in batches if b["prio"] == cur_prio)
        batches.append(
            {
                "batch": f"{cur_prio}-{n:04d}",
                "prio": cur_prio,
                "files": [r[1] for r in cur],
                "lines": cur_lines,
            }
        )
    cur, cur_lines = [], 0


for pr, f, n in rows:
    if pr != cur_prio:
        flush()
        cur_prio = pr
    if cur and (
        len(cur) >= a.max_files or cur_lines + n > budget.get(cur_prio, a.max_lines)
    ):
        flush()
    cur.append((pr, f, n))
    cur_lines += n
flush()

os.makedirs(os.path.join(out, "batches"), exist_ok=True)
for sub in ("findings", "done", "verdicts", "raw_wave_outputs"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
save(os.path.join(out, "batches", "manifest.json"), batches)
with open(os.path.join(out, "ledger.tsv"), "w") as fh:
    fh.write("file\tprio\tlines\tbatch\tstatus\tfindings\n")
    for b in batches:
        for f in b["files"]:
            fh.write(f"{f}\t{b['prio']}\t{nlines(f)}\t{b['batch']}\tpending\t0\n")
# prior-run reconciliation: per batch, what earlier runs confirmed or refuted in these files
if a.prior_run:
    import json as _json

    in_batch = {f: b["batch"] for b in batches for f in b["files"]}
    known = {}
    for pr in a.prior_run:
        vdir = os.path.join(os.path.abspath(pr), "verdicts")
        for fn in (os.listdir(vdir) if os.path.isdir(vdir) else []):
            if not fn.endswith(".json"):
                continue
            for f in _json.load(open(os.path.join(vdir, fn))).get("findings", []):
                b = in_batch.get(f.get("file"))
                if b and f.get("status") in ("confirmed", "refuted"):
                    known.setdefault(b, []).append(
                        {
                            k: f.get(k)
                            for k in ("file", "line", "category", "summary", "status")
                        }
                        | {"why": (f.get("reasons") or [""])[0][:400], "prior_run": pr}
                    )
    os.makedirs(os.path.join(out, "known"), exist_ok=True)
    for b, lst in known.items():
        save(os.path.join(out, "known", f"{b}.json"), lst)
    print(
        f"prior runs: known findings for {len(known)} batches ({sum(len(v) for v in known.values())} entries)"
    )
save(
    os.path.join(out, "state.json"),
    {
        "name": a.name or os.path.basename(out),
        "repo": a.repo,
        "root": root,
        "commit": commit,
        "since": a.since,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "knowledge": knowledge,
        "pack": os.path.relpath(pack, skill) if pack else None,
        "hot_promoted": len(promoted),
        "submodules": {"found": submodules, "included": bool(a.recurse_submodules)},
        "prior_runs": [os.path.abspath(x) for x in a.prior_run],
        "corpus": {"files": len(files), "lines": sum(r[2] for r in rows)},
        "batching": {
            "max_files": a.max_files,
            "max_lines": a.max_lines,
            "per_priority_lines": budget,
            "batches": len(batches),
        },
        "waves": [],
        "in_flight": [],
    },
)
per = {}
for b in batches:
    per.setdefault(b["prio"], [0, 0])
    per[b["prio"]][0] += 1
    per[b["prio"]][1] += len(b["files"])
print(
    f"run {out}: {len(files)} files, {sum(r[2] for r in rows)} lines, {len(batches)} batches @ {commit[:11]}"
)
for k in sorted(per):
    print(f"  prio {k}: {per[k][0]} batches, {per[k][1]} files")
