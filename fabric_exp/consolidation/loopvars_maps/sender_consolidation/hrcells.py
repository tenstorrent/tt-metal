#!/usr/bin/env python3
"""hrcells.py A.txt B.txt: the headroom cells (configs passing, per overlay and family) of two headroom_<variant>.txt
files side by side, and which differ."""
import re, sys


def cells(p):
    out, k = {}, None
    for l in open(p):
        m = re.match(r"== \S+ \+ (\w+)", l)
        if m:
            k = m.group(1)
            continue
        m = re.match(r"\s+(\S+)\s+(\d+/\d+) pass", l)
        if m and k:
            out[(k, m.group(1))] = m.group(2)
    return out


a, b = cells(sys.argv[1]), cells(sys.argv[2])
diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print(f"{len(a)} cells; differ: {len(diff)}")
for (kind, fam), x, y in diff:
    print(f"  {kind:4s} {fam:22s} {x} -> {y}")
