#!/usr/bin/env python3
"""srcdiff.py A.lst B.lst: rxlist.py --chain listings of two builds; instruction counts per source position (the
innermost frame's file:line, router_receiver.hpp / router_routing.hpp lines collapsed to the innermost function, since
those files are the ones edited) where they differ."""
import collections, re, sys
EDITED = ("router_receiver.hpp", "router_routing.hpp", "router_completions.hpp", "router_common.hpp")
def counts(p):
    c = collections.Counter()
    for l in open(p):
        m = re.match(r"^([0-9a-f]+) (..) (\S+)\s+(\S*)\s+(\S+:\d+|\?)\s+(.*)$", l.rstrip("\n"))
        if not m:
            continue
        src, chain = m.group(5), m.group(6)
        inner = chain.split(" < ")[0] if chain else "?"
        f = src.split(":")[0]
        key = f"{f}:{inner}" if f in EDITED else f"{src} {inner}"
        c[key] += 1
    return c
a, b = counts(sys.argv[1]), counts(sys.argv[2])
tot = 0
for k in sorted(set(a) | set(b), key=lambda k: -(abs(b[k] - a[k]))):
    if a[k] != b[k]:
        tot += b[k] - a[k]
        print(f"{b[k] - a[k]:+4d}  {a[k]:4d} -> {b[k]:4d}  {k[:150]}")
print("total", tot)
