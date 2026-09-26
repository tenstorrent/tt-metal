#!/usr/bin/env python3
"""Before filing: find which open findings are ALREADY reported on GitHub, by anyone, as an issue or a PR.

  filed_check.py [--run DIR] candidates [--limit 12] [--max N]   # search GitHub per finding; prints filed-wave.js args
  filed_check.py [--run DIR] persist <raw>             # record the judges' matches as dispositions
  filed_check.py [--run DIR] show

`candidates` searches the run's repo, issues and PRs, open and closed, all authors, for each open canonical finding.
It searches by the file's name and by the key identifiers in the finding (backticked or CamelCase/snake_case tokens).
It keeps the candidate items (title and trimmed body) for the judges. filed-wave.js then decides, per finding,
whether any candidate reports the SAME defect. `persist` records each match with disposition.py semantics:
  already_filed  an OPEN issue or PR already covers it         -> closed in OPEN.md, never filed again
  filed_closed   a CLOSED issue or a merged PR covered it, yet the defect is still in the audited tree
                 (the fix was incomplete, regressed, or never landed) -> "needs attention", with the link
  fixed_upstream a PR fixing it merged AFTER the audited commit    -> closed; re-check on current main before any work
A finding with no match stays open and is safe to file. Arch copies (dedup.py) travel with their canonical finding,
so a Wormhole+Blackhole bug is checked, and later filed, once, with every site listed.
GitHub's search API allows about 30 requests a minute; this pauses between findings.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, load, run_dir, save, state  # noqa: E402

out = run_dir()
st = state(out)
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
cdir = os.path.join(out, "filed_candidates")
os.makedirs(cdir, exist_ok=True)
STOP = {
    "the",
    "and",
    "for",
    "with",
    "when",
    "that",
    "this",
    "from",
    "into",
    "only",
    "never",
    "every",
    "true",
    "false",
    "uint32_t",
    "int",
    "bool",
    "void",
    "const",
    "auto",
    "size_t",
    "std",
    "ttnn",
    "tt_metal",
}


def idents(f):
    text = f["summary"] + " " + f.get("evidence", "")[:600]
    ticks = re.findall(r"`([^`]{3,60})`", text)
    toks = re.findall(r"\b(?:[a-z]+_[a-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]+)\b", text)
    seen = []
    for t in ticks + toks:
        t = re.sub(r"[^A-Za-z0-9_:]", "", t).split("::")[-1]
        if len(t) > 4 and t.lower() not in STOP and t not in seen:
            seen.append(t)
    return seen[:3]


def gh_search(kind, query, limit):
    """REST search (works with every gh version). Any failure other than a rate limit is FATAL: silently returning
    "no candidates" would make an already-filed bug look new."""
    q = f"repo:{st['repo']} is:{'issue' if kind == 'issues' else 'pr'} {query}"
    for attempt in range(6):
        r = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "GET",
                "search/issues",
                "-f",
                f"q={q}",
                "-f",
                f"per_page={limit}",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            items = json.loads(r.stdout).get("items", [])
            return [
                {
                    "number": it["number"],
                    "title": it["title"],
                    "state": it["state"].upper(),
                    "url": it["html_url"],
                    "body": it.get("body") or "",
                    "merged": bool((it.get("pull_request") or {}).get("merged_at")),
                }
                for it in items
            ]
        if "rate limit" in (r.stderr + r.stdout).lower() or "403" in r.stderr:
            time.sleep(65)
            continue
        sys.exit(f"GitHub search failed for {q!r}: {r.stderr.strip()[-300:]}")
    sys.exit(
        f"GitHub search kept hitting the rate limit for {q!r}; rerun later (finished findings are cached)"
    )


if argv[0] == "candidates":
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 12
    conf = load(os.path.join(out, "CONFIRMED.json"), [])
    disp = load(os.path.join(out, "dispositions.json"), {})
    todo = [
        f
        for f in conf
        if not f.get("duplicate_of") and not (disp.get(key_of(f)) or {}).get("state")
    ]
    if "--max" in argv:
        todo = todo[: int(argv[argv.index("--max") + 1])]
    paths = []
    for i, f in enumerate(todo):
        k = key_of(f)
        p = os.path.join(cdir, re.sub(r"[^A-Za-z0-9_.-]", "_", k) + ".json")
        if os.path.exists(p):
            paths.append(p)
            continue
        base = os.path.basename(f["file"])
        found = {}
        for q in [base] + [f"{base} {t}" for t in idents(f)[:2]] + idents(f)[:1]:
            for kind in ("issues", "prs"):
                for it in gh_search(kind, q, limit):
                    found.setdefault(
                        (kind, it["number"]),
                        {
                            **it,
                            "kind": kind[:-1],
                            "body": (it.get("body") or "")[:3000],
                        },
                    )
            time.sleep(2.2)
        rec = {
            "key": k,
            "finding": {
                x: f.get(x)
                for x in (
                    "file",
                    "line",
                    "category",
                    "summary",
                    "failure_scenario",
                    "evidence",
                )
            },
            "also_at": f.get("also_at", []),
            "candidates": list(found.values()),
        }
        json.dump(rec, open(p, "w"), indent=1)
        paths.append(p)
        print(
            f"[{i + 1}/{len(todo)}] {k}: {len(found)} candidate issues/PRs",
            file=sys.stderr,
        )
    with_cands = [p for p in paths if load(p)["candidates"]]
    json.dump({"inputs": with_cands}, sys.stdout)
    print(
        f"\n{len(todo)} open findings searched; {len(with_cands)} have GitHub candidates to judge",
        file=sys.stderr,
    )
elif argv[0] == "persist":
    raw = load(argv[1])
    r = raw.get("result", raw)
    r = json.loads(r) if isinstance(r, str) else r
    disp = load(os.path.join(out, "dispositions.json"), {})
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_open = n_closed = 0
    for res in r.get("results", []):
        m = next((c for c in res.get("matches", []) if c.get("same_bug")), None)
        if not m:
            continue
        e = disp.get(res["key"], {})
        is_open = m.get("state", "").upper() == "OPEN"
        state_ = "already_filed" if is_open else "filed_closed"
        note = (
            "already reported"
            if is_open
            else "reported and closed, but still present in the audited tree"
        )
        if m["kind"] == "pr" and not is_open:
            # a PR merged AFTER the audited commit means the bug is fixed upstream, not "still present"
            pr = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{st['repo']}/pulls/{m['number']}",
                    "--jq",
                    ".merge_commit_sha // empty",
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            pinned = (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        st["root"],
                        "rev-parse",
                        "--verify",
                        "-q",
                        f"{st['commit']}^{{commit}}",
                    ],
                    capture_output=True,
                ).returncode
                == 0
            )  # bench runs have no single pinned commit
            if pr and pinned:
                ancestor_rc = subprocess.run(
                    [
                        "git",
                        "-C",
                        st["root"],
                        "merge-base",
                        "--is-ancestor",
                        pr,
                        st["commit"],
                    ],
                    capture_output=True,
                ).returncode
                if (
                    ancestor_rc == 1
                ):  # 1 = not an ancestor; other codes = unknown object, leave as filed_closed
                    state_, note = (
                        "fixed_upstream",
                        f"fixed by a PR merged after the audited commit ({pr[:11]})",
                    )
        e["state"] = state_
        e["issue" if m["kind"] == "issue" else "pr"] = m["number"]
        e["note"] = f"{note}: {m['url']} ({m['why'][:200]})"
        e["updated"] = now
        disp[res["key"]] = e
        n_open += is_open
        n_closed += not is_open
    save(os.path.join(out, "dispositions.json"), disp)
    print(
        f"{n_open} already filed (open), {n_closed} reported-and-closed or fixed upstream; run consolidate.py"
    )
elif argv[0] == "show":
    disp = load(os.path.join(out, "dispositions.json"), {})
    for k, v in sorted(disp.items()):
        if v.get("state") in ("already_filed", "filed_closed", "fixed_upstream"):
            print(f"{v['state']:14s} {k}  {v.get('note', '')[:140]}")
else:
    sys.exit(__doc__)
