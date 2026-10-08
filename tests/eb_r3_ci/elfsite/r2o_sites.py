#!/usr/bin/env python3
"""#58724 fourth review: attribute the differences between two JIT caches to call sites (elfsite.py) with the PR's
reduce_to_one_b1.hpp lines mapped back to the PR head's (ec7714f90f8). ReduceToOneB1's dest-reuse add is one site, R2O:add: the
head's API calls (lines 610-633) and the PR's helpers (264-301) and their calls. Prints per site the builds whose Tensix
instructions or instruction count changed, and per build whether the Tensix instruction multiset outside R2O:add is unchanged.
usage: r2o_sites.py <cache A> <cache B> <kernel regex> [--a=head|pr] [--b=head|pr] [--shift=N (kernel-file lines of B later)]"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elfcmp  # noqa: E402
import elfsite  # noqa: E402

R2O = "unified_kernels/reduce_to_one_b1.hpp"
HEAD_ADD = {610, 613, 614, 622, 623, 632, 633}
# PR line -> head line: (first PR line, last PR line, offset); None marks R2O:add
PR_MAP = [(1, 263, 0), (264, 301, None), (302, 647, -38), (648, 648, None), (649, 650, -38), (651, 651, None),
          (652, 658, -37), (659, 659, None), (660, 667, -36), (668, 668, None), (669, 10**6, -35)]


def norm(site, side):
    # the twin includes the head's copy as reduce_to_one_b1_head.hpp
    m = re.match(r"^(.*unified_kernels/reduce_to_one_b1)(?:_head)?\.hpp:(\d+)$", site)
    if not m:
        return site
    f, ln = f"{m.group(1)}.hpp", int(m.group(2))
    if side == "head":
        return "R2O:add" if ln in HEAD_ADD else f"{f}:{ln}"
    for lo, hi, off in PR_MAP:
        if lo <= ln <= hi:
            return "R2O:add" if off is None else f"{f}:{ln + off}"
    return f"{f}:{ln}"


def nsites(elf, side, shift):
    out = collections.defaultdict(lambda: [[], 0])
    for s, (tt, cnt) in elfsite.sites(elf, shift=shift).items():
        k = norm(s, side)
        out[k][0] += tt
        out[k][1] += cnt
    return out


def main():
    a, b, kre = sys.argv[1:4]
    opt = dict(x[2:].split("=") for x in sys.argv[4:] if x.startswith("--"))
    sa_side, sb_side, shift = opt.get("a", "head"), opt.get("b", "pr"), int(opt.get("shift", 0))
    da, pa = elfcmp.load(a, None, kre)
    db, pb = elfcmp.load(b, None, kre)
    keys = sorted(set(da) & set(db))
    print(f"builds compared {len(keys)} (only in A {len(set(da) - set(db))}, only in B {len(set(db) - set(da))})")
    tt_sites, cnt_sites = collections.Counter(), collections.Counter()
    nbuilds, outside_same, outside_diff = collections.Counter(), collections.Counter(), collections.Counter()
    for k in keys:
        if da[k] == db[k]:
            continue
        nbuilds[k[2]] += 1
        x = nsites(pa[(k, next(iter(da[k])))], sa_side, 0)
        y = nsites(pb[(k, next(iter(db[k])))], sb_side, shift)
        for s in set(x) | set(y):
            tx, cx = x.get(s, ([], 0))
            ty, cy = y.get(s, ([], 0))
            if collections.Counter(tx) != collections.Counter(ty):
                tt_sites[(k[2], s)] += 1
            elif cx != cy:
                cnt_sites[(k[2], s)] += 1
        ox = collections.Counter(i for s, (t, c) in x.items() if s != "R2O:add" for i in t)
        oy = collections.Counter(i for s, (t, c) in y.items() if s != "R2O:add" for i in t)
        (outside_same if ox == oy else outside_diff)[k[2]] += 1
    print(f"differing builds per thread: {dict(nbuilds)}")
    print("sites whose Tensix instructions changed:")
    for (th, s), n in sorted(tt_sites.items()):
        print(f"  {th} {s}: {n} builds")
    print("sites with only an instruction-count change:")
    for (th, s), n in sorted(cnt_sites.items()):
        print(f"  {th} {s}: {n} builds")
    for th in sorted(nbuilds):
        print(f"Tensix multiset outside R2O:add, {th}: unchanged in {outside_same[th]} builds, changed in {outside_diff[th]}")


if __name__ == "__main__":
    main()
