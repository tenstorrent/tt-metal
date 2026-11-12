#!/usr/bin/env python3
"""Re-rate every confirmed finding's severity with one fixed rubric (severity-wave.js), so filed severities are
comparable.

  severity.py [--run DIR] prepare --to-dir DIR [--batch 10] [--all]
  severity.py [--run DIR] persist OUTPUT.json

A hunter picks high / medium / low on its own judgement, and verification checks whether a bug is real, not how bad
it is. `prepare` writes the canonical confirmed findings (merged-in duplicates are rated through their canonical
entry) in batches for severity-wave.js and prints its args. Findings that already carry a severity override are
skipped unless --all is given. `persist` records each rating as a disposition severity override with its one-line
reason, keeping the disposition's state; consolidate.py then shows the rated severity, and the hunter's as
`severity_audit`. It records only one valid rating per prepared item (high, medium or low, with a reason), names
every skipped, duplicated or invalid one, and exits non-zero until all are rated.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, load, run_dir, save  # noqa: E402

out = run_dir()
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
DISP = os.path.join(out, "dispositions.json")


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def confirming_reasons(f):
    """Why the finding stands confirmed: the recheck's reasons when the recheck confirmed it (the wave's were
    unsettled), else the wave verifiers'."""
    rc = f.get("recheck") or {}
    if rc.get("outcome") == "confirmed" and rc.get("reasons"):
        return rc["reasons"]
    return f.get("reasons") or []


def other_sites(f):
    """The merged copies of this defect, and any other confirmed defect on its line: each one can raise the entry's
    severity in consolidate.py, so the rater sees its own claim and failure scenario, not only its location.
    """
    sites = [
        {
            "site": m["site"],
            "relation": m.get("relation", ""),
            "claim": m["summary"],
            "failure_scenario": m["failure_scenario"],
        }
        for m in f.get("merged_sites", [])
    ]
    sites += [
        {
            "site": key_of(f),
            "relation": "another confirmed defect on the same line",
            "claim": m["summary"],
            "failure_scenario": m.get("failure_scenario", ""),
        }
        for m in f.get("same_line", [])
        if m.get("status") == "confirmed"
    ]
    return sites


if argv[0] == "prepare":
    to_dir = opt("--to-dir")
    if not to_dir:
        sys.exit("prepare needs --to-dir DIR")
    size = int(opt("--batch", 10))
    conf = load(os.path.join(out, "CONFIRMED.json"))
    if conf is None:
        sys.exit("no CONFIRMED.json: run consolidate.py first")
    disp = load(DISP, {})
    todo = [
        f
        for f in conf
        if not f.get("duplicate_of")
        and (
            "--all" in argv or not (disp.get(key_of(f)) or {}).get("severity_override")
        )
    ]
    os.makedirs(to_dir, exist_ok=True)
    for fn in os.listdir(to_dir):  # re-running replaces the previous batches
        if fn.startswith("b") and fn.endswith(".json"):
            os.remove(os.path.join(to_dir, fn))
    n = 0
    for i in range(0, len(todo), size):
        items = [
            {
                "key": key_of(f),
                "location": key_of(f)
                + "".join(f" + {x['site']}" for x in f.get("merged_sites", [])),
                "claim": f["summary"],
                "failure_scenario": f["failure_scenario"],
                "evidence": f.get("evidence", ""),
                "reasons": confirming_reasons(f)[:2],
                # the rating replaces the whole entry's severity, so the rater sees each site it covers
                "other_sites": other_sites(f),
            }
            for f in todo[i : i + size]
        ]
        save(os.path.join(to_dir, f"b{n:04d}.json"), {"items": items})
        n += 1
    print(json.dumps({"input_dir": os.path.abspath(to_dir), "n": n}))
    print(
        f"# {len(todo)} finding(s) in {n} batch(es); run severity-wave.js with these args",
        file=sys.stderr,
    )
elif argv[0] == "persist":
    if len(argv) < 2:
        sys.exit("persist needs the workflow output JSON")
    raw = load(argv[1])
    raw = raw.get("result", raw) if isinstance(raw, dict) else raw
    if isinstance(raw, str):
        raw = json.loads(raw)
    known = {key_of(f) for f in load(os.path.join(out, "CONFIRMED.json"), [])}
    disp = load(DISP, {})
    now = datetime.datetime.now().strftime("%F %T")
    # an agent can skip an item, rate one twice, or write a value outside the rubric: record only one clean rating
    # per key, and name everything else
    by_key, bad = {}, []
    for r in raw.get("ratings", []):
        k = r.get("key")
        if k not in known:
            bad.append(f"{k}: not a confirmed finding")
        elif r.get("severity") not in ("high", "medium", "low"):
            bad.append(f"{k}: severity {r.get('severity')!r} is not high/medium/low")
        elif not str(r.get("why") or "").strip():
            bad.append(f"{k}: no reason given")
        else:
            by_key.setdefault(k, []).append(r)
    for k, rs in list(by_key.items()):
        if len({r["severity"] for r in rs}) > 1:
            bad.append(f"{k}: rated {len(rs)} times, differently")
            del by_key[k]
    asked = set()
    if raw.get("input_dir") and os.path.isdir(raw["input_dir"]):
        for fn in os.listdir(raw["input_dir"]):
            if fn.startswith("b") and fn.endswith(".json"):
                asked |= {
                    i["key"] for i in load(os.path.join(raw["input_dir"], fn))["items"]
                }
    rated_bad = {b.split(": ", 1)[0] for b in bad}
    omitted = sorted(asked - set(by_key) - rated_bad)
    for k, rs in by_key.items():
        e = disp.setdefault(k, {})
        e["severity_override"] = rs[0]["severity"]
        e["severity_note"] = "re-rated (rubric): " + rs[0]["why"]
        e["updated"] = now
    save(DISP, disp)
    print(f"recorded {len(by_key)} rating(s)")
    for b in bad:
        print(f"  NOT RECORDED {b}")
    for k in omitted:
        print(f"  NOT RATED {k}: the rater skipped it")
    if raw.get("missing"):
        print(f"  batches whose rater died: {raw['missing']}")
    if not raw.get("input_dir"):
        print("  (no input_dir in the output: skipped items could not be checked)")
    if bad or omitted or raw.get("missing"):
        sys.exit(
            "re-run severity.py prepare (rated findings are skipped) and severity-wave.js for the rest"
        )
    print("next: consolidate.py")
else:
    sys.exit(__doc__)
