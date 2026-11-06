#!/usr/bin/env python3
"""covcmp_multiset.py BUILD_DIR_A BUILD_DIR_B [--all]: per case, the multiset of router image digests (loadable bytes,
cov.image) of two cov builds that may come from different host builds (their hash directories differ). Equal multisets per
case = every image of A has an identical image in B, as many times. --all: every kernel, not routers only."""
import collections, glob, hashlib, importlib.util, os, sys
args = sys.argv[1:]
A, B, everything = args[0], args[1], "--all" in args
os.environ.setdefault("TT_COV_TREE", "/home/aagarwal/tt-metal-gate-tip")
sys.argv = ["cov"]
spec = importlib.util.spec_from_file_location("cov", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cov.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def sets(d):
    out = collections.defaultdict(collections.Counter)
    for x in os.listdir(d):
        if "__kernels__" not in x or (not everything and "__fabric_erisc_router__" not in x):
            continue
        e = glob.glob(os.path.join(d, x, "*.elf"))
        if e:
            out[x.split("__kernels__")[0]][hashlib.md5(m.image(e[0])).hexdigest()] += 1
    return out
sa, sb = sets(A), sets(B)
tot = collections.Counter()
for case in sorted(set(sa) | set(sb)):
    a, b = sa.get(case, collections.Counter()), sb.get(case, collections.Counter())
    same = a == b
    tot["same" if same else "differ"] += 1
    tot["images"] += sum(a.values())
    print(f"{case:24s} {'IDENTICAL' if same else 'DIFFER':9s} images {sum(a.values())}/{sum(b.values())}"
          + ("" if same else f"  only in A {sum((a - b).values())}, only in B {sum((b - a).values())}"))
print("TOTAL", dict(tot))
