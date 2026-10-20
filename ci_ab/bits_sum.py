#!/usr/bin/env python3
"""bits_sum.py <job log>: per case, the output hashes of every arm and run; prints cases whose hashes differ between arms."""
import re, sys
from collections import defaultdict
H = defaultdict(dict)
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"== out (\S+) (r\d+) case done (\S+) (\S+)", l)
    if m: H[m.group(3)][(m.group(1), m.group(2))] = m.group(4)
bad = [c for c, v in H.items() if len(set(v.values())) > 1]
print(f"cases {len(H)}, hash differs between arms or runs: {len(bad)}"); [print(" ", c, sorted(set(H[c].values()))) for c in bad]
