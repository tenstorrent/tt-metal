#!/usr/bin/env python3
"""fndiff.py ELF_A ELF_B: instruction counts of the receiver loop function by innermost inline function, A vs B
(only the functions whose counts differ), plus totals."""
import collections, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def counts(elf):
    out = subprocess.run([sys.executable, os.path.join(HERE, "listing.py"), elf], capture_output=True, text=True).stdout
    c = collections.Counter()
    for l in out.splitlines()[1:]:
        chain = l.split()[-1] if l.split() else ""
        c[chain.split("<")[0]] += 1
    return c


a, b = counts(sys.argv[1]), counts(sys.argv[2])
rows = sorted((k for k in set(a) | set(b) if a[k] != b[k]), key=lambda k: b[k] - a[k])
for k in rows:
    print(f"  {k:44s} {a[k]:5d} {b[k]:5d} {b[k] - a[k]:+5d}")
print(f"  {'total':44s} {sum(a.values()):5d} {sum(b.values()):5d} {sum(b.values()) - sum(a.values()):+5d}")
