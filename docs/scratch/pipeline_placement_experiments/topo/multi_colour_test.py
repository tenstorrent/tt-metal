# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exact feasibility of the tight multi-mesh ring and line cases (multi2_cases.hpp) by the colour test.

Every mesh is a 4x8 grid of host slices, all 32 used (tight), the pipeline a ring or chain of
one-slice stages. With one cable per adjacent mesh pair (k = 1) the ring must use every cable exactly
once, so each mesh is crossed by one Hamiltonian path between its two cable slices. On a 4x8 grid
such a path exists iff the two slices have opposite colours (Itai, Papadimitriou, Szwarcfiter 1982;
the extra conditions only bite on grids 1 to 3 wide). So k = 1 is decided exactly. For k > 1 a
choice of one cable per pair that passes the test proves feasibility; no such choice is NOT a proof
of infeasibility, because a tight mesh may be visited in two segments over two cable pairs.
Reproduces multi2_make's LCG, so the instances are the ones the C++ harnesses run.
Usage: multi_colour_test.py [multi2|multi3]
"""
import sys
from itertools import product

M32 = 0xFFFFFFFF


def lcg(s):
    s = (s * 1664525 + 1013904223) & M32
    return s, s >> 8


def make(n, k, seed, used, ring):
    s = (seed * 7919 + n * 31 + k) & M32
    pairs = n if ring else n - 1
    cables = []
    for i in range(pairs):
        j = (i + 1) % n
        ra, rb = [], []
        while len(ra) < k:
            s, v = lcg(s)
            r = v % 4
            if r not in ra:
                ra.append(r)
        while len(rb) < k:
            s, v = lcg(s)
            r = v % 4
            if r not in rb:
                rb.append(r)
        for t in range(k):
            cables.append((i, ra[t], 7, j, rb[t], 0))
    name = f"{n} 32-rank meshes {'ring' if ring else 'line'}, {k} cable/pair, {used}/32 per mesh, seed {seed}"
    return name, n, k, used, ring, cables


def multi2():
    out = []
    for n in (2, 4, 8):
        for k in (1, 4):
            for seed in (0, 1):
                out.append(make(n, k, seed, 32, True))
    for n in (2, 4):
        for k in (1, 4):
            out.append(make(n, k, 0, 32, False))
    for n in (2, 4):
        out.append(make(n, 2, 0, 24, True))
    return out


def multi3():
    out = []
    for n in (2, 4, 8):
        for k in (1, 2, 4):
            for seed in (10, 11, 12):
                out.append(make(n, k, seed, 32, True))
    for n in (2, 4, 8):
        for k in (1, 2, 4):
            for seed in (10, 11):
                out.append(make(n, k, seed, 32, False))
    for n in (2, 4, 8):
        for k in (1, 2):
            for used in (24, 28):
                out.append(make(n, k, 10, used, True))
    return out


def colour(r, c):
    return (r + c) % 2


def decide(n, k, used, ring, cables):
    if used != 32:
        return "open (not tight)"
    pairs = n if ring else n - 1
    by_pair = {}
    for cb in cables:
        by_pair.setdefault(cb[0], []).append(cb)
    for choice in product(*[by_pair[i] for i in range(pairs)]):
        ok = True
        for m in range(n):
            ends = []
            if m < pairs:
                ends.append((choice[m][1], 7))  # exit to m+1
            prev = (m - 1) % n
            if prev < pairs and (ring or m > 0):
                ends.append((choice[prev][4], 0))  # entry from m-1
            if len(ends) == 2 and (ends[0] == ends[1] or colour(*ends[0]) == colour(*ends[1])):
                ok = False
                break
        if ok:
            return "feasible (colour)"
    return "INFEASIBLE (colour proof)" if k == 1 else "open (no single-segment choice)"


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "multi2"
    for name, n, k, used, ring, cables in multi3() if group == "multi3" else multi2():
        print(f"{name:<62} {decide(n, k, used, ring, cables)}")
