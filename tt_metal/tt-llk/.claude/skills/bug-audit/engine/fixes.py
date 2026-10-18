#!/usr/bin/env python3
"""Site-grounded suggested fixes for the open confirmed findings.

  fixes.py [--run DIR] prepare --to-dir DIR [--batch 20] [--all]   # write fix-wave.js input batches, print its args
  fixes.py [--run DIR] persist <raw_output.json>                  # merge the written fixes into suggested_fixes.json

A finding's own suggested_fix is written by the hunter before verification; a history-sibling lead has only a
placeholder. `prepare` gathers every OPEN finding (and each site merged into one) with the verifiers' write-ups, which
trace the defect in the current code, so fix-wave.js can write a fix grounded in the site. `--all` includes closed
ones. `persist` stores them in suggested_fixes.json, which consolidate.py renders in place of the placeholder; it is
durable, so re-running consolidate.py keeps them. Run consolidate.py first, and again after `persist`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, load, run_dir, save, state  # noqa: E402

out = run_dir()
st = state(out)
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
FIXES = os.path.join(out, "suggested_fixes.json")
OPEN_STATES = (
    None,
    "pr_closed",
    "filed_closed",
)  # consolidate.py's "open" and "needs attention"


def writeups(f):
    rc = f.get("recheck") or {}
    return (
        [r for r in rc.get("reasons", []) if r.startswith("[confirmed]")]
        or rc.get("reasons")
        or f.get("reasons")
        or []
    )


def item(f, site=None):
    return {
        "key": site or key_of(f),
        "site": site or key_of(f),
        "source": f.get("source", "hunt"),
        "category": f.get("category"),
        "summary": f.get("summary"),
        "failure_scenario": f.get("failure_scenario"),
        "evidence": f.get("evidence"),
        "verifier_writeups": writeups(f),
    }


if argv[0] == "prepare":
    if "--to-dir" not in argv:
        sys.exit("prepare needs --to-dir DIR")
    d = os.path.abspath(argv[argv.index("--to-dir") + 1])
    size = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else 20
    conf = load(os.path.join(out, "CONFIRMED.json"))
    if conf is None:
        sys.exit("no CONFIRMED.json: run consolidate.py first")
    items = []
    for f in conf:
        if f.get("duplicate_of"):
            continue  # written with its canonical finding, as a merged site
        if "--all" not in argv and (
            (f.get("disposition") or {}).get("state") not in OPEN_STATES
        ):
            continue
        items.append(item(f))
        # a merged site can fail and be fixed differently from its canonical one, so each gets its own fix
        for m in f.get("merged_sites", []):
            site = m["site"].split(" (")[0]
            twin = next((x for x in conf if key_of(x) == site), None)
            items.append(item(twin or {**f, **m}, site))
    os.makedirs(d, exist_ok=True)
    n = 0
    for n, i in enumerate(range(0, len(items), size), 1):
        save(os.path.join(d, f"f{n - 1:04d}.json"), {"items": items[i : i + size]})
    print(f"# {len(items)} site(s) in {n} batch(es)", file=sys.stderr)
    print(json.dumps({"input_dir": d, "n": n, "root": st["root"]}))
elif argv[0] == "persist":
    raw = load(argv[1])
    raw = (
        raw.get("result", raw) if isinstance(raw, dict) else raw
    )  # workflow output wraps the return value
    got = {
        x["key"]: x["fix"].strip()
        for x in raw.get("fixes", [])
        if x.get("fix", "").strip()
    }
    fixes = load(FIXES, {})
    fixes.update(got)
    save(FIXES, fixes)
    missing = raw.get("missing") or []
    print(
        f"stored {len(got)} fix(es); {len(fixes)} in suggested_fixes.json"
        + (f"; {len(missing)} batch(es) died -- rerun them" if missing else "")
        + ". Run consolidate.py."
    )
else:
    sys.exit(__doc__)
