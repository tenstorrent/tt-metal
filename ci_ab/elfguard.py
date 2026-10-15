#!/usr/bin/env python3
"""Scratch, not for merge. usage: elfguard.py <log> <armA> <armB> [<armC> ...]; reads the "ELF <arm> <name> <key> <proc>=<hash>"
lines elfhash.py printed, pairs builds by (name, key) and prints, for armA against every other arm, the builds in both
arms, how many differ and the kernel names that differ ("RAB ELFDIFF")."""
import collections, re, sys
log, arms = sys.argv[1], sys.argv[2:]
b = {a: {} for a in arms}
for l in open(log, errors="ignore"):
    l = re.sub(r"^\S+Z ", "", l.rstrip("\n"))
    p = l.split(" ")
    if p[0] == "ELF" and p[1] in b and len(p) > 4:
        b[p[1]][(p[2], p[3])] = " ".join(p[4:])
for o in arms[1:]:
    both = [k for k in b[arms[0]] if k in b[o]]
    diff = [k for k in both if b[arms[0]][k] != b[o][k]]
    names = collections.Counter(k[0] for k in diff)
    print(f"RAB ELFPAIRS {arms[0]}_vs_{o} builds {len(b[arms[0]])}/{len(b[o])} paired {len(both)} differ {len(diff)}")
    for n, c in sorted(names.items()):
        print(f"RAB ELFDIFF {arms[0]}_vs_{o} {n} {c}")
