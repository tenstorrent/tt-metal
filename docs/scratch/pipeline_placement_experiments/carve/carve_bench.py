# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ring-of-rectangles carve on one slice grid: DFS with memo vs CaDiCaL.

Problem: stage k takes a rectangle of its shape (in slices), rectangles are disjoint, stage k touches
stage k+1, and the last stage touches stage 0.

Each solve runs in a child process that is killed after CARVE_TIMEOUT seconds (default 20):
SIGALRM cannot interrupt CaDiCaL inside C code. Usage: carve_bench.py [quick|big].
"""
import multiprocessing as mp
import os
import sys
import time

from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

TIMEOUT = float(os.environ.get("CARVE_TIMEOUT", "20"))


def positions(R, C, h, w):
    out = []
    for r in range(R - h + 1):
        for c in range(C - w + 1):
            m = 0
            for i in range(r, r + h):
                for j in range(c, c + w):
                    m |= 1 << (i * C + j)
            out.append(m)
    return out


def dilate(R, C, m):
    d = 0
    for b in range(R * C):
        if m >> b & 1:
            i, j = divmod(b, C)
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, bb = i + di, j + dj
                if 0 <= a < R and 0 <= bb < C:
                    d |= 1 << (a * C + bb)
    return d & ~m


def dfs_solve(R, C, shapes):
    n = len(shapes)
    pos = [positions(R, C, *s) for s in shapes]
    halo = {}
    for ps in pos:
        for m in ps:
            if m not in halo:
                halo[m] = dilate(R, C, m)
    area = [h * w for h, w in shapes]
    suffix = [sum(area[k:]) for k in range(n + 1)]
    full = (1 << (R * C)) - 1
    failed = set()
    nodes = 0

    def free_reach(free, src_halo):
        seen = src_halo & free
        frontier = seen
        while frontier:
            nxt = 0
            f = frontier
            while f:
                b = f & -f
                f ^= b
                nxt |= halo_bit[b.bit_length() - 1]
            nxt &= free & ~seen
            seen |= nxt
            frontier = nxt
        return seen

    halo_bit = [dilate(R, C, 1 << b) for b in range(R * C)]

    def go(k, used, last, first):
        nonlocal nodes
        nodes += 1
        if k == n:
            return bool(halo[last] & first)
        key = (k, used, last, first)
        if key in failed:
            return False
        free = full & ~used
        if bin(free).count("1") < suffix[k]:
            failed.add(key)
            return False
        if k > 0:
            reach = free_reach(free, halo[last])
            if bin(reach).count("1") < suffix[k] or not (dilate_cache(reach) & first):
                failed.add(key)
                return False
        for m in pos[k]:
            if m & used:
                continue
            if k > 0 and not (halo[last] & m):
                continue
            if go(k + 1, used | m, m, first if k else m):
                return True
        failed.add(key)
        return False

    dcache = {}

    def dilate_cache(m):
        if m not in dcache:
            d = 0
            f = m
            while f:
                b = f & -f
                f ^= b
                d |= halo_bit[b.bit_length() - 1]
            dcache[m] = d
        return dcache[m]

    ok = go(0, 0, 0, 0)
    return ok, nodes


def sat_solve(R, C, shapes):
    n = len(shapes)
    pool = IDPool()
    pos = [positions(R, C, *s) for s in shapes]
    x = {(k, i): pool.id((k, i)) for k in range(n) for i in range(len(pos[k]))}
    clauses = []
    for k in range(n):
        lits = [x[k, i] for i in range(len(pos[k]))]
        clauses += CardEnc.equals(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses
    for b in range(R * C):
        lits = [x[k, i] for k in range(n) for i, m in enumerate(pos[k]) if m >> b & 1]
        if len(lits) > 1:
            clauses += CardEnc.atmost(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses
    for k in range(n):
        k2 = (k + 1) % n
        for i, m in enumerate(pos[k]):
            h = dilate(R, C, m)
            clauses.append([-x[k, i]] + [x[k2, j] for j, m2 in enumerate(pos[k2]) if h & m2])
    with Cadical153(bootstrap_with=clauses) as s:
        ok = s.solve()
        return ok, s.accum_stats().get("conflicts", 0) if hasattr(s, "accum_stats") else 0


def _child(fn, R, C, shapes, q):
    t0 = time.perf_counter()
    ok, _ = fn(R, C, shapes)
    q.put((ok, time.perf_counter() - t0))


def timed(fn, R, C, shapes):
    """(ok, seconds), or (None, TIMEOUT) when the child is killed."""
    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(target=_child, args=(fn, R, C, shapes, q))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join()
        return None, TIMEOUT
    return q.get()


def run(name, R, C, shapes):
    res = []
    for label, fn in (("dfs", dfs_solve), ("sat", sat_solve)):
        ok, dt = timed(fn, R, C, shapes)
        if ok is None:
            res.append(f"{label} ??? >{TIMEOUT:.0f}s   ")
        else:
            res.append(f"{label} {'yes' if ok else 'no '} {dt:8.3f}s")
    area = sum(h * w for h, w in shapes)
    print(f"{name:<44} area {area:>2}/{R*C:<2} | " + " | ".join(res), flush=True)


def lattice_check(R, C, shapes):
    """Each h x w rectangle holds exactly one cell with i % h == h-1 and j % w == w-1, and an
    H x W stage holds at least (H // h) * (W // w) of them. Exact necessary condition, O(shapes^2)."""
    for h, w in set(shapes):
        marks = (R // h) * (C // w)
        need = sum((H // h) * (W // w) for H, W in shapes)
        if need > marks:
            return f"lattice {h}x{w}: stages need {need} marked cells, grid has {marks}"
    return None


S1, B, W = (1, 1), (2, 2), (2, 4)
QUICK = [
    ("4x6: 1x1, 4x 2x2, 1x1", 4, 6, [S1] + [B] * 4 + [S1]),
    ("4x8: 1x1, 6x 2x2, 1x1", 4, 8, [S1] + [B] * 6 + [S1]),
    ("4x8: 1x1, 2x4, 2x4, 2x2, 1x1", 4, 8, [S1, W, W, B, S1]),
    ("8x8: 1x1, 14x 2x2, 1x1", 8, 8, [S1] + [B] * 14 + [S1]),
    ("3x8: 1x1, 5x 2x2 (only 4 fit)", 3, 8, [S1] + [B] * 5),
    ("5x5: 1x1, 5x 2x2 (only 4 fit)", 5, 5, [S1] + [B] * 5),
    ("7x7: 1x1, 10x 2x2 (only 9 fit)", 7, 7, [S1] + [B] * 10),
    ("6x6: 1x1, 8x 2x2, 1x1", 6, 6, [S1] + [B] * 8 + [S1]),
    ("6x6: 1x1, 9x 2x2 (area 37 > 36)", 6, 6, [S1] + [B] * 9),
    ("8x8: 1x1, 15x 2x2, 3x 1x1 (fills 64)", 8, 8, [S1] + [B] * 15 + [S1] * 3),
]
# Slow: the first case had not finished when the run was stopped after 15 minutes on one core.
# Run them on a spare machine.
BIG = [
    ("9x9: 1x1, 17x 2x2 (only 16 fit)", 9, 9, [S1] + [B] * 17),
    ("8x10: 1x1, 2x4 x9 (only 8 fit)", 8, 10, [S1] + [W] * 9),
    ("8x8: 1x1, 2x4 x4, 2x2 x7", 8, 8, [S1] + [W] * 4 + [B] * 7),
]


def main(cases):
    for c in cases:
        t0 = time.perf_counter()
        why = lattice_check(*c[1:])
        dt = time.perf_counter() - t0
        if why:
            print(f"{c[0]:<44} lattice pre-check rejects in {dt*1e6:.0f} us: {why}", flush=True)
        run(*c)


if __name__ == "__main__":
    main(BIG if sys.argv[1:] == ["big"] else QUICK)
