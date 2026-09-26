#!/usr/bin/env python3
"""Print the Workflow args for the next wave, and mark those batches in flight.

  next_wave.py [--run DIR] N [PRIOS] [--dry]      e.g. next_wave.py 120 AB
  next_wave.py [--run DIR] N [PRIOS] --second-pass  re-issue COMPLETE batches that have had only one hunt

Picks batches with no done marker, not already in flight, in priority order (batches sent back by the read check
come first). Prints the JSON object to hand to Workflow as `args` for engine/audit-wave.js. Unless --dry, records
the batches in state.json's in_flight list so a second wave launched while this one runs cannot re-issue them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, manifest, marker_set, run_dir, save, state  # noqa: E402

out = run_dir()
dry = "--dry" in sys.argv
second = "--second-pass" in sys.argv
argv = [x for x in sys.argv[1:] if x not in ("--dry", "--second-pass")]
if not argv:
    sys.exit(__doc__)
n = int(argv[0])
prios = argv[1] if len(argv) > 1 else None
st = state(out)
man = manifest(out)
done = marker_set(out, "done", ".done")
in_flight = set(st.get("in_flight", []))
reread = load(os.path.join(out, "reread.json"), {})

if second:
    # a batch has had a second hunt once persist_wave has archived its first one into findings/<b>.history.jsonl
    once = {
        b
        for b in done
        if not os.path.exists(os.path.join(out, "findings", f"{b}.history.jsonl"))
    }
    todo = [
        b
        for b in man
        if b in once
        and b not in in_flight
        and (prios is None or man[b]["prio"] in prios)
    ]
else:
    todo = [
        b
        for b in man
        if b not in done
        and b not in in_flight
        and (prios is None or man[b]["prio"] in prios)
    ]
todo.sort(key=lambda b: (b not in reread, man[b]["prio"], b))
pick = todo[:n]
if not dry:
    st["in_flight"] = sorted(in_flight | set(pick))
    save(os.path.join(out, "state.json"), st)
here = os.path.dirname(os.path.abspath(__file__))
skill = os.path.dirname(here)
knowledge = [
    k if os.path.isabs(k) else os.path.join(skill, k) for k in st.get("knowledge", [])
]
roots = {b: man[b]["root"] for b in pick if man[b].get("root")}
rdirs = {os.path.dirname(r) for r in roots.values()}
if (
    roots
    and len(rdirs) == 1
    and all(os.path.basename(r) == b.replace("BENCH-", "") for b, r in roots.items())
):
    roots = {"roots_dir": rdirs.pop()}
else:
    roots = {"roots": roots} if roots else {}
extra = {}
if os.path.isdir(os.path.join(out, "known")):
    extra["known_dir"] = os.path.join(out, "known")
if (st.get("execution") or {}).get("enabled") and os.path.isdir(
    os.path.join(out, "exec", "signals")
):
    extra["exec_signals_dir"] = os.path.join(out, "exec", "signals")
print(
    json.dumps(
        {
            "run": out,
            "root": st["root"],
            "batches": pick,
            "knowledge": knowledge,
            **roots,
            **extra,
        }
    )
)
print(
    f"# {len(pick)} batches ({sum(1 for b in pick if b in reread)} re-hunts); {len(todo) - len(pick)} left after this"
    + ("" if dry else "; marked in flight"),
    file=sys.stderr,
)
