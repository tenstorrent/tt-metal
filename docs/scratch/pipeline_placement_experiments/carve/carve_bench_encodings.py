# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Carve SAT encodings against the DFS: does a slice-level grouping encoding beat positions?

Same problem as carve_bench.py (stage k is a rectangle of its shape in slices, rectangles are
disjoint, k touches k+1, last touches first). Four encodings:

  pos       one variable per stage per rectangle position; AMO per slice over all covering
            positions; "k here -> k+1 in one of the halo positions" (the carve_bench.py baseline)
  pos+sym   pos, plus stage 0 pinned to one quadrant (grid reflections), and to the lower
            triangle when the grid is square and every stage is square (transpose)
  cell      broken down to slices: y[k,b] = slice b belongs to stage k, channelled to an anchor
            variable per rectangle position; AMO per slice over stages (n literals, not
            positions); touching via e[k,b] = "slice b of k has a k+1 neighbour", one of which
            must hold
  cell+sym  cell with the same stage-0 pinning

Each solve runs in a child process killed after CARVE_TIMEOUT seconds (default 30).
Usage: carve_bench_encodings.py [quick|big|open|file|sizes]; sizes prints variable/clause counts only;
file reads the cases from CARVE_CASES (the format topo/carve_encoding_cases.txt uses).
"""
import os
import random
import sys
import multiprocessing as mp
import time
from functools import partial

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical153

from carve_bench import TIMEOUT, dfs_solve, dilate, lattice_check, positions


def _neighbours(R, C, b):
    i, j = divmod(b, C)
    for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        a, bb = i + di, j + dj
        if 0 <= a < R and 0 <= bb < C:
            yield a * C + bb


def _anchor_rc(R, C, h, w, i):
    """positions() enumerates row-major over (R-h+1) x (C-w+1)."""
    return divmod(i, C - w + 1)


def _sym_units(R, C, shapes, x0, pool_unused=None):
    """Unit clauses pinning stage 0 to a canonical position under the grid's symmetries."""
    h0, w0 = shapes[0]
    square_all = R == C and all(h == w for h, w in shapes)
    units = []
    for i, _ in enumerate(positions(R, C, h0, w0)):
        r, c = _anchor_rc(R, C, h0, w0, i)
        bad = r > (R - h0) // 2 or c > (C - w0) // 2
        if square_all and r > c:
            bad = True
        if bad:
            units.append([-x0[i]])
    return units


def _amo(lits, pool):
    if len(lits) <= 1:
        return []
    if len(lits) <= 6:
        return CardEnc.atmost(lits, 1, vpool=pool, encoding=EncType.pairwise).clauses
    return CardEnc.atmost(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses


def build_pos(R, C, shapes, sym=False):
    n = len(shapes)
    pool = IDPool()
    pos = [positions(R, C, *s) for s in shapes]
    x = {(k, i): pool.id(("x", k, i)) for k in range(n) for i in range(len(pos[k]))}
    clauses = []
    for k in range(n):
        lits = [x[k, i] for i in range(len(pos[k]))]
        clauses += CardEnc.equals(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses
    for b in range(R * C):
        lits = [x[k, i] for k in range(n) for i, m in enumerate(pos[k]) if m >> b & 1]
        clauses += CardEnc.atmost(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses if len(lits) > 1 else []
    for k in range(n):
        k2 = (k + 1) % n
        for i, m in enumerate(pos[k]):
            h = dilate(R, C, m)
            clauses.append([-x[k, i]] + [x[k2, j] for j, m2 in enumerate(pos[k2]) if h & m2])
    if sym:
        clauses += _sym_units(R, C, shapes, {i: x[0, i] for i in range(len(pos[0]))})
    return pool, clauses


def build_cell(R, C, shapes, sym=False):
    n = len(shapes)
    pool = IDPool()
    pos = [positions(R, C, *s) for s in shapes]
    a = {(k, i): pool.id(("a", k, i)) for k in range(n) for i in range(len(pos[k]))}
    y = {(k, b): pool.id(("y", k, b)) for k in range(n) for b in range(R * C)}
    e = {(k, b): pool.id(("e", k, b)) for k in range(n) for b in range(R * C)}
    clauses = []
    for k in range(n):
        # exactly one anchor per stage
        clauses += CardEnc.equals(
            [a[k, i] for i in range(len(pos[k]))], 1, vpool=pool, encoding=EncType.seqcounter
        ).clauses
        covering = {b: [] for b in range(R * C)}
        for i, m in enumerate(pos[k]):
            for b in range(R * C):
                if m >> b & 1:
                    covering[b].append(i)
                    clauses.append([-a[k, i], y[k, b]])  # anchor -> its slices
        for b in range(R * C):
            clauses.append([-y[k, b]] + [a[k, i] for i in covering[b]])  # slice -> some anchor covers it
    for b in range(R * C):
        clauses += _amo([y[k, b] for k in range(n)], pool)  # at most one stage per slice
    for k in range(n):
        k2 = (k + 1) % n
        for b in range(R * C):
            clauses.append([-e[k, b], y[k, b]])
            clauses.append([-e[k, b]] + [y[k2, nb] for nb in _neighbours(R, C, b)])
        clauses.append([e[k, b] for b in range(R * C)])  # some slice of k touches k+1
    if sym:
        clauses += _sym_units(R, C, shapes, {i: a[0, i] for i in range(len(pos[0]))})
    return pool, clauses


def build_sub(R, C, shapes, sym=False, bidir=True, touch="port"):
    """Graph-to-graph mapping. Logical graph: one sub-node per slice of every stage, with the
    stage's rectangle fixed by two-literal clauses (sub-node (i,j) at slice (r,c) implies sub-node
    (i,j+1) at (r,c+1), and (i+1,j) at (r+1,c)). Physical graph: one node per slice, edges from
    the adjacency set, so a multi-mesh slice graph with cable edges plugs in unchanged. x[v,b]:
    sub-node v sits on slice b. Exactly one slice per sub-node, at most one sub-node per slice.
    Pipeline edge k -> k+1: pick an exit-port sub-node of k; it must sit next to some slice held
    by k+1. For single-slice stages this collapses to "v at b implies u on a neighbour of b"."""
    n = len(shapes)
    N = R * C
    pool = IDPool()
    adj = {b: list(_neighbours(R, C, b)) for b in range(N)}
    # sub-nodes and the slices each one can occupy (the rectangle must fit in the grid)
    subs = []  # (k, i, j)
    allowed = {}  # (k,i,j) -> list of b
    for k, (h, w) in enumerate(shapes):
        for i in range(h):
            for j in range(w):
                v = (k, i, j)
                subs.append(v)
                allowed[v] = [
                    r * C + c for r in range(R) for c in range(C) if i <= r < R - h + i + 1 and j <= c < C - w + j + 1
                ]
    x = {(v, b): pool.id(("x", v, b)) for v in subs for b in allowed[v]}
    clauses = []
    for v in subs:
        lits = [x[v, b] for b in allowed[v]]
        clauses += CardEnc.equals(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses
    for b in range(N):
        clauses += _amo([x[v, b] for v in subs if (v, b) in x], pool)
    # rigidity: (k,i,j)@(r,c) -> (k,i,j+1)@(r,c+1) and (k,i+1,j)@(r+1,c)
    for k, i, j in subs:
        h, w = shapes[k]
        for b in allowed[(k, i, j)]:
            r, c = divmod(b, C)
            if j + 1 < w:
                clauses.append([-x[(k, i, j), b], x[(k, i, j + 1), r * C + c + 1]])
                if bidir:
                    clauses.append([x[(k, i, j), b], -x[(k, i, j + 1), r * C + c + 1]])
            if i + 1 < h:
                clauses.append([-x[(k, i, j), b], x[(k, i + 1, j), (r + 1) * C + c]])
                if bidir:
                    clauses.append([x[(k, i, j), b], -x[(k, i + 1, j), (r + 1) * C + c]])
    # pipeline edges: exit port of k sits next to a slice of k+1
    for k in range(n):
        k2 = (k + 1) % n
        mine = [v for v in subs if v[0] == k]
        theirs = [u for u in subs if u[0] == k2]
        if touch == "slice" and len(mine) > 1:
            # t[k,b]: slice b, held by k, has a neighbour held by k+1; one such slice must exist
            t = {b: pool.id(("t", k, b)) for b in range(N)}
            for b in range(N):
                clauses.append([-t[b]] + [x[v, b] for v in mine if (v, b) in x])
                clauses.append([-t[b]] + [x[u, nb] for nb in adj[b] for u in theirs if (u, nb) in x])
            clauses.append(list(t.values()))
            continue
        if len(mine) == 1:
            port = {mine[0]: None}
        else:
            port = {v: pool.id(("port", v)) for v in mine}
            clauses += CardEnc.equals(
                list(port.values()), 1, vpool=pool, encoding=EncType.pairwise if len(mine) <= 6 else EncType.seqcounter
            ).clauses
        for v in mine:
            for b in allowed[v]:
                head = [-x[v, b]] if port[v] is None else [-port[v], -x[v, b]]
                tail = [x[u, nb] for nb in adj[b] for u in theirs if (u, nb) in x]
                clauses.append(head + tail)
    if sym:
        h0, w0 = shapes[0]
        anchor0 = {}
        for b in allowed[(0, 0, 0)]:
            r, c = divmod(b, C)
            anchor0[r * (C - w0 + 1) + c] = x[(0, 0, 0), b]
        clauses += _sym_units(R, C, shapes, anchor0)
    return pool, clauses


def solve_with(builder, R, C, shapes):
    """Returns (ok, build_seconds): the harness subtracts build time, which is pure-Python
    clause construction a C++ encoder would not pay."""
    t0 = time.perf_counter()
    pool, clauses = builder(R, C, shapes)
    build = time.perf_counter() - t0
    with Cadical153(bootstrap_with=clauses) as s:
        return s.solve(), build


def _child(fn, R, C, shapes, q):
    t0 = time.perf_counter()
    ok, build = fn(R, C, shapes)
    q.put((ok, time.perf_counter() - t0, build if isinstance(build, float) else 0.0))


def timed3(fn, R, C, shapes):
    """(ok, total_seconds, build_seconds), or (None, TIMEOUT, 0) when the child is killed."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, R, C, shapes, q))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join()
        return None, TIMEOUT, 0.0
    return q.get()


from carve_dfs2 import dfs2_solve  # noqa: E402  (symmetry, dynamic lattice, two-ended, compact-first)

ENCODINGS = [
    ("dfs", dfs_solve),
    ("dfs2", dfs2_solve),
    ("pos", partial(solve_with, partial(build_pos, sym=False))),
    ("pos+sym", partial(solve_with, partial(build_pos, sym=True))),
    ("cell", partial(solve_with, partial(build_cell, sym=False))),
    ("cell+sym", partial(solve_with, partial(build_cell, sym=True))),
    ("sub", partial(solve_with, partial(build_sub, sym=False))),
    ("sub+sym", partial(solve_with, partial(build_sub, sym=True))),
]


def run(name, R, C, shapes):
    res = []
    for label, fn in ENCODINGS:
        ok, dt, build = timed3(fn, R, C, shapes)
        if ok is None:
            res.append(f"{label} ???  >{TIMEOUT:.0f}s      ")
        else:
            # solve time excludes clause construction; (+b) is the Python build time
            res.append(
                f"{label} {'yes' if ok else 'no '} {dt - build:7.3f}s" + (f" (+{build:.1f}b)" if build else "       ")
            )
    area = sum(h * w for h, w in shapes)
    print(f"{name:<40} {area:>3}/{R*C:<3} | " + " | ".join(res), flush=True)


def sizes(name, R, C, shapes):
    out = []
    for label, b in (("pos", build_pos), ("cell", build_cell), ("sub", build_sub)):
        t0 = time.perf_counter()
        pool, clauses = b(R, C, shapes)
        lits = sum(len(c) for c in clauses)
        out.append(
            f"{label} vars {pool.top:>6} clauses {len(clauses):>7} lits {lits:>8} build {time.perf_counter()-t0:5.2f}s"
        )
    print(f"{name:<40} | " + " | ".join(out), flush=True)


S1, H, V, B, W = (1, 1), (1, 2), (2, 1), (2, 2), (2, 4)


def repo_random_cases():
    """The six random fills from carve_bench_repo.py, regenerated with the same seed."""
    random.seed(0)
    cases = []
    for _ in range(6):
        R, C = random.choice([(4, 8), (8, 4), (2, 16), (8, 8)])
        shapes, area = [S1], 1
        while True:
            s = random.choice([S1, S1, H, V, B])
            if s[0] > R or s[1] > C or area + s[0] * s[1] > R * C:
                break
            shapes.append(s)
            area += s[0] * s[1]
        cases.append((f"random {R}x{C}: {len(shapes)} stages", R, C, shapes))
    return cases


def wide_random_cases(R, C, seed, count, target_fill=0.95):
    random.seed(seed)
    cases = []
    for idx in range(count):
        shapes, area = [S1], 1
        while area < target_fill * R * C:
            s = random.choice([S1, H, V, B, B, W])
            if area + s[0] * s[1] > R * C:
                break
            shapes.append(s)
            area += s[0] * s[1]
        cases.append((f"random {R}x{C} s{seed}.{idx}: {len(shapes)} stages", R, C, shapes))
    return cases


QUICK = (
    [
        ("8x8: 1x1, 14x 2x2, 1x1", 8, 8, [S1] + [B] * 14 + [S1]),
        ("8x8: 1x1, 15x 2x2, 3x 1x1 (fills)", 8, 8, [S1] + [B] * 15 + [S1] * 3),
        ("8x8 ring of 64 x 1x1 (fills)", 8, 8, [S1] * 64),
        ("8x16: 1x1, 30x 2x2, 1x1", 8, 16, [S1] + [B] * 30 + [S1]),
        ("8x16: 1x1, 2x4 x14, 2x2 x3, 1x1 (fills)", 8, 16, [S1] + [W] * 14 + [B] * 3 + [S1] * 2),
        ("8x16 ring of 128 x 1x1 (fills)", 8, 16, [S1] * 128),
        ("16x16: 1x1, 62x 2x2, 1x1", 16, 16, [S1] + [B] * 62 + [S1]),
    ]
    + [c for c in repo_random_cases() if c[1] == 8 and c[2] == 8]
    + wide_random_cases(8, 16, 1, 3)
    + [
        # infeasible references
        ("7x7: 1x1, 10x 2x2 (only 9 fit)", 7, 7, [S1] + [B] * 10),
        ("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, [S1] * 25),
    ]
)

BIG = [
    ("8x8: 1x1, 2x4 x4, 2x2 x7", 8, 8, [S1] + [W] * 4 + [B] * 7),
    ("9x9: 1x1, 17x 2x2 (only 16 fit)", 9, 9, [S1] + [B] * 17),
] + wide_random_cases(16, 16, 2, 2)


# The quick cases every solver capped at 60 s: rerun with a long cap to learn whether they are
# feasible at all (CARVE_TIMEOUT=600 CARVE_ENCODINGS=dfs,pos carve_bench_encodings.py open).
OPEN = [
    c for c in QUICK if c[0] in ("random 8x8: 30 stages", "random 8x8: 34 stages") or c[0].startswith("random 8x16")
]


def main(argv):
    global ENCODINGS
    mode = argv[0] if argv else "quick"
    cases = {"big": BIG, "open": OPEN}.get(mode, QUICK)
    if mode == "file":  # CARVE_CASES: name<TAB>R<TAB>C<TAB>h w;h w;... (shared with topo/carve_footprint_bench.cpp)
        cases = []
        for line in open(os.environ["CARVE_CASES"]):
            if line.strip():
                name, R, C, sh = line.rstrip("\n").split("\t")
                cases.append((name, int(R), int(C), [tuple(map(int, p.split())) for p in sh.split(";")]))
    only = os.environ.get("CARVE_ENCODINGS")
    if only:
        ENCODINGS = [(l, f) for l, f in ENCODINGS if l in only.split(",")]
    print(f"encodings: {[l for l, _ in ENCODINGS]}  timeout {TIMEOUT:.0f}s", flush=True)
    for c in cases:
        why = lattice_check(*c[1:])
        if why:
            print(f"{c[0]:<40} lattice pre-check rejects: {why}", flush=True)
        if mode == "sizes":
            sizes(*c)
        else:
            run(*c)


if __name__ == "__main__":
    main(sys.argv[1:])
