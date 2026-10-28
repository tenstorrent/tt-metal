#!/usr/bin/env python3
"""pcs_tables.py SECTIONS.json...: per-row summary tables (markdown) from pcs_sections.py output: for the busiest E0
and E1 of each row (most time in work sections; for multicast, the mean over all busy ERISCs of that kind), cycles per
packet by category (section share x the row's cycles/packet)."""
import collections
import json
import sys

E0_CATS = [("send", ("worker: send", "visit: send")),
           ("credits", ("worker: credit", "upstream credit", "completions and bookkeeping")),
           ("checks", ("send check", "completion check", "connection poll", "idle test", "group read")),
           ("loop", ("loop and round", "per-round", "round end", "connection event"))]
E1_CATS = [("poll", ("E1 poll",)),
           ("decode+admit", ("E1 decode", "E1 admission", "1D route decision")),
           ("local", ("local delivery",)),
           ("forward", ("E1 forward",)),
           ("completions", ("completions:",)),
           ("retire+atomic", ("E1 retire", "deferred atomic", "packet bookkeeping")),
           ("loop", ("loop and round", "per-round", "step other"))]
NAMES = {"ew4352": "1-hop E/W write 4352 B", "ew16": "1-hop E/W write 16 B", "ns4352": "1-hop N/S write 4352 B",
         "ewscatter": "1-hop E/W scatter 4352 B", "ewfused": "1-hop E/W fused write+atomic", "ewatomic":
         "1-hop E/W atomic inc", "hop3": "3-hop E/W write (middle routers)", "mcast": "multicast all-to-all (mean)",
         "fwd": "forwarded write (middle router, 2 flows)", "bidir": "1-hop E/W bidirectional",
         "lin1": "1D 1-hop write", "lin3": "1D 3-hop write (middle routers)", "linmcast": "1D multicast (mean)"}


def cats(sections, table):
    out = collections.OrderedDict((c, 0.0) for c, _ in table)
    for s, f in sections.items():
        if s.endswith("check") and "checks" in out:  # "worker: send check" is a check, not a send
            out["checks"] += f
            continue
        for c, keys in table:
            if any(k in s for k in keys):
                out[c] += f
                break
    return out


rows = [r for p in sys.argv[1:] for r in json.load(open(p))]
for erisc, table in ((0, E0_CATS), (1, E1_CATS)):
    print(f"\n| Row (cycles/packet) | {'E0' if erisc == 0 else 'E1'} work | " + " | ".join(c for c, _ in table) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in table) + " |")
    for r in rows:
        busy = [t for t in r["targets"] if t.get("work_share", 0) > 0.15 and t["target"]["erisc"] == erisc]
        if not busy:
            continue
        cpp = r["cycles_per_packet"]
        if "mcast" in r["row"]:
            agg = collections.Counter()
            for t in busy:
                for s, f in t["sections"].items():
                    agg[s] += f / len(busy)
            secs, ws = agg, sum(t["work_share"] for t in busy) / len(busy)
        else:
            t = max(busy, key=lambda t: t["work_share"])
            secs, ws = t["sections"], t["work_share"]
        c = cats(secs, table)
        print(f"| {NAMES.get(r['row'], r['row'])} ({cpp:.1f}) | {100 * ws:.0f}% | " +
              " | ".join(f"{v * cpp:.1f}" for v in c.values()) + " |")
