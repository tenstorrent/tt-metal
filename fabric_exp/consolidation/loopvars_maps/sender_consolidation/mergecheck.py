#!/usr/bin/env python3
"""mergecheck.py OPT LIST FUNC [FUNC...]: for a merge M of two parents P and Q over a common base (stage 0: M = v4txc0,
P = v4txo8 over hrnew = v4rxg3 under the job-464 host rule, Q = v4rxnsf2 over v4rxg3), per image in LIST and per
function (name prefix), checks that M's change from Q equals P's change from its base, and M's change from P equals Q's
change from its base, as multisets of instructions with registers normalised and call/branch targets by symbol (a
function both parents changed holds both changes and nothing else). Prints the counts and the first failures."""
import collections, re, sys
src = open(__file__.replace("mergecheck.py", "fnsplit.py")).read()
ns = {}
exec(src.split("def one(rel)")[0], ns)
funcs = ns["funcs"]
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
REG = re.compile(r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b")
o, lst, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
M, P, PB, Q, QB = "v4txc0", "v4txo8", "hrnew", "v4rxnsf2", "v4rxg3"


def norm(lines):
    return [REG.sub("R", re.sub(r"\[clone [^\]]*\]", "", x)) for x in lines]


def delta(a, b):
    ca, cb = collections.Counter(norm(a)), collections.Counter(norm(b))
    return ca - cb, cb - ca


ok = collections.Counter()
bad = []
for rel in open(lst).read().split():
    f = {b: funcs(f"{CB}/{b}_{o}/{rel}") for b in (M, P, PB, Q, QB)}
    for pre in prefixes:
        ks = [k for k in f[M] if k.startswith(pre)]
        for k in ks:
            g = {b: f[b].get(k, []) for b in f}
            good = delta(g[Q], g[M]) == delta(g[PB], g[P]) and delta(g[P], g[M]) == delta(g[QB], g[Q])
            ok[(pre, good)] += 1
            if not good:
                bad.append((rel, k))
for (pre, good), n in sorted(ok.items()):
    print(f"{o} {pre}: {'merge of both changes' if good else 'OTHER'} in {n} images")
for rel, k in bad[:10]:
    print("  OTHER:", rel, k[:100])
