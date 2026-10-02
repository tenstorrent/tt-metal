#!/usr/bin/env python3
"""Second look at the candidates a wave could not settle, and at a sample of the ones it killed.

  recheck.py [--run DIR] queue [--refuted-sample 0.1] [--seed 1] [--max N] [--to-dir DIR]
  recheck.py [--run DIR] persist <raw_output.json>
  recheck.py [--run DIR] report

`queue` adds to recheck.json every needs_recheck and uncertain candidate, plus a seeded random sample of the refuted
ones, and prints the Workflow args for engine/recheck-wave.js. The refuted sample is how the run measures its own
false-negative rate: `report` prints how many sampled refutations were overturned. A high reversal rate means the
verifiers are killing real bugs, and the whole refuted pile needs a recheck, not a sample.
`persist` records each outcome; consolidate.py then lets it override the wave verdict.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, load, run_dir, save, seeded_order, state  # noqa: E402

out = run_dir()
st = state(out)
path = os.path.join(out, "recheck.json")
rc = load(path, {})
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)


def opt(name, default, cast):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


if argv[0] == "queue":
    rows = []
    for fn in sorted(os.listdir(os.path.join(out, "verdicts"))):
        if fn.endswith(".json"):
            rows += load(os.path.join(out, "verdicts", fn), {}).get("findings", [])
    confirmed = {key_of(f) for f in rows if f["status"] == "confirmed"}
    refuted = sorted(
        {
            key_of(f): f
            for f in rows
            if f["status"] == "refuted" and key_of(f) not in confirmed
        }.items()
    )
    k = round(len(refuted) * opt("--refuted-sample", 0.1, float))
    sample = dict(seeded_order(refuted, opt("--seed", 1, int), lambda kv: kv[0])[:k])
    added = 0
    for f in rows:
        key = key_of(f)
        why = {
            "needs_recheck": "a verifier died",
            "uncertain": "no verifier could settle it",
        }.get(f["status"])
        if str((f.get("reasons") or [""])[0]).startswith("[deferred]"):
            why = "deferred at the wave's agent limit"
        if why is None and key in sample:
            why = "refuted-sample"
        if why and key not in rc and key not in confirmed:
            rc[key] = {
                "why": why,
                "outcome": "queued",
                "finding": {
                    x: f[x]
                    for x in (
                        "file",
                        "line",
                        "category",
                        "severity",
                        "summary",
                        "failure_scenario",
                        "evidence",
                        "batch",
                    )
                },
                "prior_reasons": f.get("reasons", []),
            }
            added += 1
    save(path, rc)
    todo = [k for k, v in rc.items() if v["outcome"] == "queued"][
        : opt("--max", 10**9, int)
    ]
    if "--to-dir" in argv:
        # large waves: one small file per item, so the workflow args stay tiny (dir + count)
        d = os.path.abspath(argv[argv.index("--to-dir") + 1])
        os.makedirs(d, exist_ok=True)
        index = {}
        for i, k in enumerate(todo):
            p = os.path.join(d, f"c{i:04d}.json")
            save(p, rc[k])
            index[p] = k
        save(os.path.join(d, "index.json"), index)
        print(
            json.dumps({"run": out, "root": st["root"], "items_dir": d, "n": len(todo)})
        )
    else:
        print(
            json.dumps({"run": out, "root": st["root"], "items": [rc[k] for k in todo]})
        )
    print(
        f"# queued {added} new; {len(todo)} handed to this recheck wave",
        file=sys.stderr,
    )
elif argv[0] == "persist":
    raw = load(argv[1])
    r = raw.get("result", raw) if isinstance(raw, dict) else raw
    if isinstance(r, str):
        r = json.loads(r)
    n = 0
    index = {}
    for item in r.get("items", []):
        if "path" in item and not index:
            index = load(os.path.join(os.path.dirname(item["path"]), "index.json"), {})
        key = index.get(item.get("path")) or key_of(item["finding"])
        if key in rc:
            # the waves it saw: a later pass that re-confirms this site is new evidence the recheck never judged
            rc[key].update(
                outcome=item["outcome"],
                votes=item["votes"],
                reasons=item["reasons"],
                after_wave=len(st.get("waves", [])),
            )
            n += 1
    save(path, rc)
    print(f"recorded {n} recheck outcomes; run consolidate.py")
elif argv[0] == "report":
    samp = [
        v
        for v in rc.values()
        if v["why"] == "refuted-sample" and v["outcome"] != "queued"
    ]
    flips = [v for v in samp if v["outcome"] == "confirmed"]
    other = [
        v
        for v in rc.values()
        if v["why"] != "refuted-sample" and v["outcome"] != "queued"
    ]
    print(
        f"refuted sample: {len(samp)} rechecked, {len(flips)} overturned to confirmed"
        + (f" ({100 * len(flips) / len(samp):.0f}% reversal)" if samp else "")
    )
    for v in flips:
        print(f"  OVERTURNED {key_of(v['finding'])}: {v['finding']['summary'][:120]}")
    print(
        f"uncertain / died: {len(other)} rechecked -> "
        + ", ".join(
            f"{o} {sum(1 for v in other if v['outcome'] == o)}"
            for o in ("confirmed", "refuted", "uncertain")
        )
    )
    print(f"still queued: {sum(1 for v in rc.values() if v['outcome'] == 'queued')}")
else:
    sys.exit(__doc__)
