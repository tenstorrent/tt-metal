#!/usr/bin/env python3
"""Pick the benchmark holdout set, then the deep-read set (which never includes the holdout).

  select.py holdout --cases cases.jsonl --triage triage.jsonl --out holdout.jsonl [--n 60] [--seed 7]
                    [--code-glob 'tt_metal/**' ...] [--test-glob '*/tests/*' ...]
  select.py screened --cases picked.jsonl --screen <holdout-screen output> --out holdout.jsonl --n 60 [--triage x]
  select.py deep --cases cases.jsonl --triage triage.jsonl --exclude holdout.jsonl --out-dir deep/ \
                 [--min-priority 2] [--per-batch 4] [--max 0] [--include-test-bugs]

holdout: a seeded random sample of confirmed code bugs whose fix is small (<= 3 code files, <= 120 changed lines), was
not reverted, and touches code rather than only tests. It also rejects no-functional-change commits (the --reject
pattern on the title or fix subject), include-only or comment-only diffs (with --git), and anything a deep read judged
not real. These filters are cheap and deliberately narrow ("rename", "enable" and "typo" also appear in real fixes).
They do NOT replace the validity screen: run holdout-screen-wave.js on an oversampled pick (--n about 1.4x the target)
and keep the first N valid cases (select.py screened). A post-mortem found 9 of 75 unscreened picks were not defects.
The output is in the format bench.py expects. Everything downstream must exclude it, or the recall benchmark measures
memory instead of skill.
deep: triaged code bugs at or above --min-priority, plus every case whose fix was reverted or re-fixed (those are
the incomplete-fix candidates, whatever their priority). Written as batch files of full cases with their triage.
"""
import argparse
import collections
import fnmatch
import json
import os
import random
import re
import sys

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["holdout", "screened", "deep"])
p.add_argument("--screen", help="screened: the holdout-screen workflow's output JSON")
p.add_argument("--cases", required=True)
p.add_argument("--triage", help="required for holdout and deep")
p.add_argument("--out")
p.add_argument("--out-dir")
p.add_argument(
    "--exclude",
    action="append",
    default=[],
    help="jsonl files of cases to exclude (by id AND by fix commit): old holdouts, the deep-read store",
)
p.add_argument("--n", type=int, default=60)
p.add_argument("--seed", type=int, default=7)
p.add_argument(
    "--test-glob",
    action="append",
    default=[
        "tests/*",
        "*/tests/*",
        "*test_*",
        "*_test.*",
        ".github/*",
        "*/unit_tests/*",
    ],
)
p.add_argument("--min-priority", type=int, default=2)
p.add_argument("--per-batch", type=int, default=4)
p.add_argument("--max", type=int, default=0)
p.add_argument("--include-test-bugs", action="store_true")
p.add_argument(
    "--only-ids", help="file of case ids to restrict to (e.g. an area filter)"
)
p.add_argument(
    "--git",
    help="holdout: local clone, used to reject fixes whose diff only touches includes/comments/blank lines",
)
p.add_argument(
    "--deep",
    action="append",
    default=[],
    help="deep-read stores; cases a deep read judged not real are excluded",
)
p.add_argument(
    "--reject",
    default=r"\b(clean ?up|clang-tidy|lint|nfc|no functional|non-functional|cosmetic)\b",
    help="holdout: reject cases whose title or fix subject matches (no-functional-change heuristics)",
)
p.add_argument(
    "--sample",
    type=int,
    default=0,
    help="deep: also add this many cases from priority --min-priority - 1, stratified round-robin by component",
)
a = p.parse_args()

cases = {c["id"]: c for c in map(json.loads, filter(str.strip, open(a.cases)))}
if a.cmd != "screened" and not a.triage:
    p.error("--triage is required for holdout and deep")
tri = (
    {t["id"]: t for t in map(json.loads, filter(str.strip, open(a.triage)))}
    if a.triage
    else {}
)
excl, excl_fix = set(), set()
for f in a.exclude:
    for x in filter(str.strip, open(f)):
        r = json.loads(x)
        excl.add(str(r["id"]))
        if r.get("fix_commit"):
            excl_fix.add(
                r["fix_commit"]
            )  # an issue case and its PR case can share one fix commit
only = set(open(a.only_ids).read().split()) if a.only_ids else None
is_test = lambda f: any(fnmatch.fnmatch(f, g) for g in a.test_glob)  # noqa: E731
CODE_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".inl", ".py", ".rs", ".go")

not_real, deep_fix = set(), set()
for f in a.deep:
    for d in map(json.loads, filter(str.strip, open(f))):
        if d.get("is_real_bug") == "no":
            not_real.add(d["id"])
        if (
            d["id"] in cases
        ):  # a deep-read fix may be in the pack, so no holdout case may share it
            deep_fix |= {fx["oid"] for fx in cases[d["id"]]["fix"]}


def trivial_diff(oid, files):
    """True when every changed line is an include, a comment or blank: nothing an auditor could have flagged."""
    import subprocess

    d = subprocess.run(
        ["git", "-C", a.git, "show", "-U0", "--format=", oid, "--", *files],
        capture_output=True,
        text=True,
    ).stdout
    body = [
        ln[1:].strip()
        for ln in d.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    return all(not x or x.startswith(("#include", "//", "/*", "*", "#")) for x in body)


if a.cmd == "holdout":
    reject = re.compile(a.reject, re.I)
    pool = []
    for cid, t in tri.items():
        c = cases.get(cid)
        if (
            not c
            or t["verdict"] != "code-bug"
            or not c["fix"]
            or (only and cid not in only)
            or cid in not_real
        ):
            continue
        if any(fx["oid"] in excl_fix or fx["oid"] in deep_fix for fx in c["fix"]):
            continue  # same fix as an excluded or deep-read case under another id
        if reject.search(c["title"]) or any(
            reject.search(fx["subject"]) for fx in c["fix"]
        ):
            continue
        later = c.get("later") or {}
        if later.get("reverts"):
            continue
        fx = c["fix"][-1]
        code = [f for f in fx["files"] if f.endswith(CODE_EXT) and not is_test(f)]
        if (
            not code
            or len(code) > 3
            or fx["adds"] + fx["dels"] > 120
            or fx["nfiles"] > 6
        ):
            continue
        if a.git and trivial_diff(fx["oid"], code):
            continue
        pool.append(
            {
                "id": cid,
                "fix_commit": fx["oid"],
                "files": code,
                "class": (t["classes"] or ["?"])[0],
                "summary": t["mechanism"],
                "component": t["component"],
                "title": c["title"],
            }
        )
    pool.sort(key=lambda x: x["id"])
    pick = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    with open(a.out, "w") as fh:
        for x in pick:
            fh.write(json.dumps(x) + "\n")
    print(f"holdout: {len(pick)} of {len(pool)} eligible -> {a.out}")
elif a.cmd == "screened":
    raw = json.load(open(a.screen))
    r = raw.get("result", raw)
    r = json.loads(r) if isinstance(r, str) else r
    ok = {v["id"]: v for v in r["verdicts"]}
    rows = [json.loads(x) for x in open(a.cases) if x.strip()]
    keep = [x for x in rows if ok.get(x["id"], {}).get("valid")][: a.n]
    with open(a.out, "w") as fh:
        for x in keep:
            fh.write(json.dumps(x) + "\n")
    print(
        f"screened: kept {len(keep)} valid of {len(rows)} ({sum(1 for v in ok.values() if not v['valid'])} rejected) -> {a.out}"
    )
else:
    sel = []
    for cid, t in tri.items():
        if cid in excl or cid in not_real or (only and cid not in only):
            continue
        c = cases.get(cid)
        if not c:
            continue
        later = c.get("later") or {}
        suspicious = bool(later.get("reverts") or later.get("citing_later"))
        ok_verdict = t["verdict"] == "code-bug" or (
            a.include_test_bugs and t["verdict"] == "test-bug"
        )
        if (ok_verdict and t["deep_priority"] >= a.min_priority) or (
            suspicious and t["verdict"] in ("code-bug", "test-bug", "unclear")
        ):
            sel.append((-(t["deep_priority"] + 2 * suspicious), cid))
    sel.sort()
    if a.sample:
        chosen = {cid for _, cid in sel}
        pool = collections.defaultdict(list)
        for cid, t in tri.items():
            if (
                cid in chosen
                or cid in excl
                or cid not in cases
                or (only and cid not in only)
            ):
                continue
            if (
                t["verdict"] == "code-bug"
                and t["deep_priority"] == a.min_priority - 1
                and cases[cid]["fix"]
            ):
                pool[t["component"].lower()].append(cid)
        rnd = random.Random(a.seed)
        for v in pool.values():
            rnd.shuffle(v)
        comps = sorted(pool, key=lambda k: -len(pool[k]))
        added = 0
        while added < a.sample and any(pool.values()):
            for k in comps:
                if pool[k] and added < a.sample:
                    sel.append((0, pool[k].pop()))
                    added += 1
    if a.max:
        sel = sel[: a.max]
    ids = [cid for _, cid in sel]
    ids.sort(
        key=lambda i: (tri[i]["component"], i)
    )  # keep an area's cases together in a batch
    os.makedirs(a.out_dir, exist_ok=True)
    paths = []
    for i in range(0, len(ids), a.per_batch):
        path = os.path.join(
            os.path.abspath(a.out_dir), f"deep-{i // a.per_batch:04d}.json"
        )
        batch = []
        for cid in ids[i : i + a.per_batch]:
            c = dict(cases[cid])
            c["triage"] = tri[cid]
            batch.append(c)
        json.dump({"cases": batch}, open(path, "w"))
        paths.append(path)
    json.dump(paths, sys.stdout)
    print(
        f"\ndeep: {len(ids)} cases in {len(paths)} batches -> {a.out_dir}",
        file=sys.stderr,
    )
