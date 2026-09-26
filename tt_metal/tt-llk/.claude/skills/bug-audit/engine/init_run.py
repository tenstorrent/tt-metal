#!/usr/bin/env python3
"""Create a bug-audit run: enumerate the files in scope, assign priorities, pack line-bounded batches.

  init_run.py --root /path/to/checkout --out /path/to/run-dir --repo owner/name \
      [--prio 'A=src/kernels/**,src/core/**' --prio 'B=tools/**'] [--default-prio C] \
      [--ext .c,.cc,.cpp,.h,.hpp,.py] [--include 'src/area/**'] [--exclude 'third_party/**'] \
      [--since <commit>] [--max-files 20] [--max-lines 3500] \
      [--knowledge references/classes-universal.md,packs/<repo>.md]

--root must be a git checkout pinned at the commit you mean to audit (a dedicated worktree is best), so
recorded file:line findings stay valid for the life of the run. The commit is recorded in state.json.
--since limits the scope to files changed between <commit> and the audited commit (diff mode).
Priorities are glob lists matched in the order given; unmatched files get --default-prio.
Batches never mix priorities, keep directories contiguous, and hold at most --max-files files and
--max-lines lines (or the --batch-lines budget for their priority; priority A defaults to 1500), so an agent can
read, and actually analyse, every line of every file it is assigned.
"""
import argparse
import datetime
import fnmatch
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
p.add_argument("--max-lines", type=int, default=3500)
p.add_argument(
    "--batch-lines",
    default="A=1500",
    help="per-priority line budget overriding --max-lines, e.g. 'A=1500,B=3500'. Less code per hunter means "
    "deeper reading; the post-fix miss analysis found skimming, not coverage, was the main cause of misses",
)
p.add_argument(
    "--knowledge",
    default="",
    help="comma list of knowledge files the hunters must read",
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


def git(*args):
    return subprocess.run(
        ["git", "-C", root, *args], check=True, capture_output=True, text=True
    ).stdout


commit = git("rev-parse", "HEAD").strip()
dirty = git("status", "--porcelain", "--untracked-files=no").strip()
if dirty:
    print(
        "WARNING: the audited tree has uncommitted changes; findings will cite lines that are not in any commit",
        file=sys.stderr,
    )

exts = tuple(e.strip() for e in a.ext.split(",") if e.strip())
submodules = [
    ln.split("\t", 1)[1]
    for ln in git("ls-files", "-s").splitlines()
    if ln.startswith("160000")
]
listing = (
    git("ls-files", "--recurse-submodules") if a.recurse_submodules else git("ls-files")
)
files = [f for f in listing.splitlines() if f.endswith(exts)]
if submodules and not a.recurse_submodules:
    print(
        f"WARNING: {len(submodules)} submodule(s) are OUT of scope (each is one gitlink entry): "
        + ", ".join(submodules[:8])
        + (" ..." if len(submodules) > 8 else "")
        + ". Pass --recurse-submodules to include them, or audit each as its own run.",
        file=sys.stderr,
    )
if a.since:
    changed = set(git("diff", "--name-only", f"{a.since}...{commit}").splitlines())
    files = [f for f in files if f in changed]
if a.include:
    files = [f for f in files if any(fnmatch.fnmatch(f, g) for g in a.include)]
files = [f for f in files if not any(fnmatch.fnmatch(f, g) for g in a.exclude)]

prio_rules = []
for spec in a.prio:
    letter, _, globs = spec.partition("=")
    prio_rules.append(
        (letter.strip(), [g.strip() for g in globs.split(",") if g.strip()])
    )


def prio_of(f):
    for letter, globs in prio_rules:
        if any(fnmatch.fnmatch(f, g) for g in globs):
            return letter
    return a.default_prio


def nlines(f):
    try:
        with open(os.path.join(root, f), "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


rows = sorted(((prio_of(f), f, nlines(f)) for f in files), key=lambda r: (r[0], r[1]))
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
knowledge = [k.strip() for k in a.knowledge.split(",") if k.strip()]
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
