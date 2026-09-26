#!/usr/bin/env python3
"""Recall benchmark: audit real past bugs at the commit BEFORE their fix, and measure how many the audit finds.

  bench.py prepare --cases CASES.jsonl --repo-path /path/to/clone --out RUN_DIR --repo owner/name \
                   [--knowledge a.md,b.md] [--limit N] [--trees-dir DIR]
  bench.py score [--run RUN_DIR] [--window 10] [--cases CASES.jsonl]   (--cases: drop cases since marked valid=false)
  bench.py judge-inputs --arm LABEL=RUN_DIR [--arm ...] --repo-path CLONE --out-dir DIR [--cases CASES.jsonl]
  bench.py judged --dir DIR --raw <judge-wave.js output>

CASES.jsonl has one held-out fix per line: {"id", "fix_commit", "files": [paths], "class", "summary"}. The mining step
writes it (packs/<repo>-holdout.jsonl) and keeps those cases OUT of the knowledge pack — a pack that contains the
answer would score a meaningless 100%.

`prepare` makes one detached worktree per case at <fix_commit>^ under RUN_DIR/trees/, and one batch per case holding
the files the fix changed (the hunter still greps the whole tree). Run it through the normal wave loop (next_wave.py
-> audit-wave.js -> persist_wave.py). Prepare the same cases twice, with and without --knowledge, to measure what
the knowledge adds.
`score` counts a case as FOUND when a finding in its batch lands in a file the fix changed, within --window lines of a
line the fix changed (pre-fix numbering). Recall is reported for confirmed findings and for any candidate at all,
so a verifier that kills true positives shows up as a gap between the two.
`score` is a quick proxy: line proximity both over-credits (a different bug near the fixed lines) and under-credits
(the right defect reported at an unchanged line). For numbers you will act on, use SEMANTIC scoring:
`judge-inputs` writes one blinded input per case holding every arm's findings under opaque ids, then
`judge-wave.js` asks two independent judges, plus a third on disagreement, "does this finding describe the defect this
fix removed?", and `judged` turns the matches into recall per arm.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, manifest, run_dir, save, state  # noqa: E402


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout


def changed_lines(repo, commit, files):
    """Pre-fix line numbers the fix touched, per file (deleted/changed lines, or the insertion point for pure adds)."""
    diff = git(repo, "diff", "-U0", f"{commit}^", commit, "--", *files)
    res, cur = {}, None
    for ln in diff.splitlines():
        if ln.startswith("--- "):
            cur = ln[6:] if ln.startswith("--- a/") else None
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? ", ln)
        if m and cur:
            start, n = int(m.group(1)), int(m.group(2) or 1)
            res.setdefault(cur, []).extend(range(start, start + max(n, 1)))
    return res


if len(sys.argv) > 1 and sys.argv[1] == "prepare":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--cases", required=True)
    p.add_argument("--repo-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--knowledge", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--trees-dir",
        help="where the per-case worktrees live (default RUN_DIR/trees); share it between arms",
    )
    a = p.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "state.json")):
        sys.exit(f"{out} already holds a run")
    for sub in ("batches", "findings", "done", "verdicts", "raw_wave_outputs", "trees"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cases = [json.loads(x) for x in open(a.cases) if x.strip()]
    skipped = [c["id"] for c in cases if c.get("valid") is False or c.get("exclude")]
    # valid=false: post-mortem found no defect; exclude: contaminated (shares a fix with knowledge or rule sources)
    cases = [c for c in cases if c.get("valid") is not False and not c.get("exclude")]
    if skipped:
        print(f"  skipping {len(skipped)} cases marked valid=false")
    if a.limit:
        cases = cases[: a.limit]
    batches, truth = [], {}
    for c in cases:
        tree = os.path.join(
            os.path.abspath(a.trees_dir) if a.trees_dir else os.path.join(out, "trees"),
            str(c["id"]),
        )
        if not os.path.exists(tree):
            git(a.repo_path, "worktree", "add", "--detach", tree, f"{c['fix_commit']}^")
        files = [f for f in c["files"] if os.path.exists(os.path.join(tree, f))]
        if not files:
            print(
                f"  skip {c['id']}: no fixed file exists before the fix (pure addition)"
            )
            continue
        b = f"BENCH-{c['id']}"
        lines = sum(sum(1 for _ in open(os.path.join(tree, f), "rb")) for f in files)
        batches.append(
            {"batch": b, "prio": "A", "files": files, "lines": lines, "root": tree}
        )
        truth[b] = {
            "case": c,
            "changed": changed_lines(a.repo_path, c["fix_commit"], files),
        }
    save(os.path.join(out, "batches", "manifest.json"), batches)
    save(os.path.join(out, "truth.json"), truth)
    save(
        os.path.join(out, "state.json"),
        {
            "name": os.path.basename(out),
            "repo": a.repo,
            "root": os.path.abspath(a.repo_path),
            "commit": "per-case",
            "created": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "knowledge": [k for k in a.knowledge.split(",") if k],
            "bench": True,
            "waves": [],
            "in_flight": [],
        },
    )
    print(
        f"bench {out}: {len(batches)} cases prepared (knowledge: {a.knowledge or 'none'})"
    )
elif len(sys.argv) > 1 and sys.argv[1] == "score":
    out = run_dir()
    window = (
        int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 10
    )
    truth = load(os.path.join(out, "truth.json"), {})
    man = manifest(out)
    # cases later found not to be defects (valid=false in the holdout file) are dropped from older runs too
    invalid = set()
    if "--cases" in sys.argv:
        invalid = {
            c["id"]
            for c in map(
                json.loads,
                filter(str.strip, open(sys.argv[sys.argv.index("--cases") + 1])),
            )
            if c.get("valid") is False or c.get("exclude")
        }
    found_conf = found_any = judged_cases = 0
    per_class = {}
    for b, t in sorted(truth.items()):
        v = load(os.path.join(out, "verdicts", f"{b}.json"), {"findings": []})[
            "findings"
        ]
        hunted = os.path.exists(os.path.join(out, "done", f"{b}.done"))
        if not hunted or t["case"].get("valid") is False or t["case"]["id"] in invalid:
            continue
        judged_cases += 1

        def hit(f):
            ls = t["changed"].get(f["file"], [])
            return any(abs(f["line"] - x) <= window for x in ls)

        any_hit = any(hit(f) for f in v)
        conf_hit = any(hit(f) for f in v if f["status"] == "confirmed")
        found_any += any_hit
        found_conf += conf_hit
        k = t["case"].get("class", "?")
        pc = per_class.setdefault(k, [0, 0, 0])
        pc[0] += 1
        pc[1] += conf_hit
        pc[2] += any_hit
        print(
            f"  {'FOUND' if conf_hit else ('cand ' if any_hit else 'miss ')} {b} [{k}] {t['case'].get('summary', '')[:90]}"
        )
    if not judged_cases:
        sys.exit("no benchmark case has been hunted yet")
    print(
        f"\nrecall (confirmed): {found_conf}/{judged_cases} = {100 * found_conf / judged_cases:.0f}%"
    )
    print(
        f"recall (any candidate): {found_any}/{judged_cases} = {100 * found_any / judged_cases:.0f}%"
    )
    print("\nper class: cases / confirmed hits / candidate hits")
    for k, (n, c, x) in sorted(per_class.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:40s} {n:3d} {c:3d} {x:3d}")
elif len(sys.argv) > 1 and sys.argv[1] == "judge-inputs":
    # Semantic scoring, step 1: one blinded input per case holding every arm's findings for that case.
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        help="LABEL=RUN_DIR (repeatable; all arms share cases)",
    )
    p.add_argument("--repo-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases", help="holdout file; cases marked valid=false are skipped")
    a = p.parse_args()
    arms = [x.split("=", 1) for x in a.arm]
    invalid = set()
    if a.cases:
        invalid = {
            c["id"]
            for c in map(json.loads, filter(str.strip, open(a.cases)))
            if c.get("valid") is False or c.get("exclude")
        }
    truth = load(os.path.join(arms[0][1], "truth.json"), {})
    os.makedirs(a.out_dir, exist_ok=True)
    import hashlib

    paths, keymap = [], {}
    for b, t in sorted(truth.items()):
        c = t["case"]
        if c.get("valid") is False or c["id"] in invalid:
            continue
        found = []
        for label, run in arms:
            if not os.path.exists(os.path.join(run, "done", f"{b}.done")):
                continue
            for i, f in enumerate(
                load(os.path.join(run, "verdicts", f"{b}.json"), {"findings": []})[
                    "findings"
                ]
            ):
                # opaque, deterministic id: judges must not know which arm produced a finding
                fid = "F" + hashlib.sha1(f"{label}|{b}|{i}".encode()).hexdigest()[:8]
                keymap[fid] = {"arm": label, "batch": b, "status": f["status"]}
                found.append(
                    {
                        "id": fid,
                        "file": f["file"],
                        "line": f["line"],
                        "category": f["category"],
                        "summary": f["summary"],
                        "failure_scenario": f["failure_scenario"],
                    }
                )
        found.sort(key=lambda x: x["id"])  # shuffles arms together
        diff = subprocess.run(
            [
                "git",
                "-C",
                a.repo_path,
                "show",
                "-U4",
                c["fix_commit"],
                "--",
                *c["files"],
            ],
            capture_output=True,
            text=True,
        ).stdout[:24000]
        path = os.path.join(os.path.abspath(a.out_dir), f"judge-{c['id']}.json")
        json.dump(
            {
                "case_id": c["id"],
                "summary": c.get("summary", ""),
                "title": c.get("title", ""),
                "fix_diff": diff,
                "findings": found,
            },
            open(path, "w"),
            indent=1,
        )
        paths.append(path)
    save(os.path.join(a.out_dir, "keymap.json"), keymap)
    json.dump({"inputs": paths}, sys.stdout)
    print(
        f"\n{len(paths)} judge inputs, {len(keymap)} findings across {len(arms)} arms",
        file=sys.stderr,
    )
elif len(sys.argv) > 1 and sys.argv[1] == "judged":
    # Semantic scoring, step 2: recall per arm from the judge workflow's matches.
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument(
        "--dir", required=True, help="the judge-inputs --out-dir (holds keymap.json)"
    )
    p.add_argument("--raw", required=True, help="judge-wave.js output JSON")
    a = p.parse_args()
    keymap = load(os.path.join(a.dir, "keymap.json"))
    raw = load(a.raw)
    r = raw.get("result", raw)
    r = json.loads(r) if isinstance(r, str) else r
    # one judge run may cover several benchmark dirs; keep only the cases whose input lives in --dir
    here = os.path.abspath(a.dir)
    cases = {
        x["case_id"]: x
        for x in r["cases"]
        if os.path.dirname(os.path.abspath(x["path"])) == here
    }
    failed = [
        f
        for f in r.get("failed", [])
        if f and os.path.dirname(os.path.abspath(f)) == here
    ]
    if failed:
        print(
            f"!! {len(failed)} cases in this dir were not judged (a judge died); rerun them before trusting recall"
        )
    arms = sorted({v["arm"] for v in keymap.values()})
    per = {arm: {"conf": set(), "any": set()} for arm in arms}
    for cid, x in cases.items():
        for fid in x["matching"]:
            k = keymap.get(fid)
            if not k:
                continue
            per[k["arm"]]["any"].add(cid)
            if k["status"] == "confirmed":
                per[k["arm"]]["conf"].add(cid)
    n = len(cases)
    disagreements = sum(x.get("disagreements", 0) for x in cases.values())
    print(
        f"{n} cases judged; {disagreements} finding-level judge disagreements resolved by a third judge"
    )
    for arm in arms:
        c, y = len(per[arm]["conf"]), len(per[arm]["any"])
        print(
            f"  {arm:24s} confirmed {c}/{n} = {100 * c / n:.0f}%   any candidate {y}/{n} = {100 * y / n:.0f}%"
        )
    save(
        os.path.join(a.dir, "judged-summary.json"),
        {arm: {k: sorted(v) for k, v in d.items()} for arm, d in per.items()},
    )
else:
    sys.exit(__doc__)
