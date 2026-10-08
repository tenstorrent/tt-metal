#!/usr/bin/env python3
"""For each build that differs between two caches, compare the multiset of Tensix instructions (with operands) over the whole
ELF minus the dest-reuse call sites; equal means every change outside those sites only moves instructions between sites (inlining,
outlining, merging). usage: nondr_check.py <cache A> <cache B> <kernel regex> [--shift=N]"""
import collections, re, sys
import elfcmp, elfsite

DR = re.compile(r"(reduce_to_one_b1\.hpp:6[0-4]\d|rmsnorm\.hpp:15\d|gated_reduce\.hpp:18\d|eltwise_mul\.hpp:21\d)")
a, b, kre = sys.argv[1:4]
shift = next((int(x.split("=")[1]) for x in sys.argv[4:] if x.startswith("--shift=")), 0)
da, pa = elfcmp.load(a, None, kre)
db, pb = elfcmp.load(b, None, kre)
res = collections.Counter()
for k in sorted(set(da) & set(db)):
    if da[k] == db[k] or not k[2].startswith("trisc"):
        continue
    x = pa[(k, next(iter(da[k])))]; y = pb[(k, next(iter(db[k])))]
    sa, sb = elfsite.sites(x), elfsite.sites(y, shift=shift)
    ta = collections.Counter(i for s, (t, c) in sa.items() if not DR.search(s) for i in t)
    tb = collections.Counter(i for s, (t, c) in sb.items() if not DR.search(s) for i in t)
    res[(k[2], ta == tb)] += 1
    if ta != tb:
        print("NONDR DIFF", k[2], x, dict(ta - tb), dict(tb - ta))
print(dict(res))
