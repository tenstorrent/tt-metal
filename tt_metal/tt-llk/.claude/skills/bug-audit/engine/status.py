#!/usr/bin/env python3
"""Ground-truth progress of a run. Reads only on-disk state, so it is right even after a session died mid-wave.

  status.py [--run DIR]

A batch is COMPLETE when it has a done marker (every file passed the read check). Everything else is:
  pending        never hunted, or sent back by the read check (listed in reread.json)
  in flight      handed to a workflow that has not been persisted yet
If no workflow is actually running, everything in flight is ORPHANED: clear state.json's in_flight list and re-issue
those batches (their hunt results were never persisted).
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, manifest, marker_set, run_dir, state  # noqa: E402

out = run_dir()
st = state(out)
man = manifest(out)
done = marker_set(out, "done", ".done")
in_flight = set(st.get("in_flight", [])) - done
reread = load(os.path.join(out, "reread.json"), {})
pending = [b for b in man if b not in done and b not in in_flight]

nfiles = lambda bs: sum(len(man[b]["files"]) for b in bs)  # noqa: E731
tot = nfiles(man)
print(
    f"run {st['name']}  tree {st['root']} @ {st['commit'][:11]}"
    + (f"  (diff since {st['since']})" if st.get("since") else "")
)
print(
    f"batches  complete {len(done)} | in flight {len(in_flight)} | pending {len(pending)} ({len(reread)} sent back by the read check) | of {len(man)}"
)
print(
    f"files    audited {nfiles(done)}/{tot} ({100 * nfiles(done) / max(tot, 1):.1f}%)"
)
status = collections.Counter()
for fn in os.listdir(os.path.join(out, "verdicts")):
    if fn.endswith(".json"):
        for f in load(os.path.join(out, "verdicts", fn), {}).get("findings", []):
            status[f["status"]] += 1
print(
    "verdicts " + ", ".join(f"{k} {v}" for k, v in sorted(status.items()))
    if status
    else "verdicts none yet"
)
recheck = load(os.path.join(out, "recheck.json"), {})
if recheck:
    c = collections.Counter(v.get("outcome", "queued") for v in recheck.values())
    print("recheck  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
if in_flight:
    print(
        "\nIN FLIGHT (orphaned if no workflow is running): "
        + " ".join(sorted(in_flight))
    )
print("\nremaining by priority:")
for p in sorted({m["prio"] for m in man.values()}):
    rem = [b for b in pending if man[b]["prio"] == p]
    if rem:
        print(f"  {p}: {len(rem):4d} batches, {nfiles(rem):5d} files   first: {rem[0]}")
if not pending and not in_flight:
    print(
        "  none — the hunt is complete. Before calling the run done: work the recheck queue (recheck.py) and"
        " run the recall benchmark (see SKILL.md)."
    )
