# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""An improved carve DFS: the carve_bench.py search plus what the SAT encodings got for free.

Same problem (stage k is a rectangle of its shape in slices, rectangles disjoint, k touches k+1,
last touches first). Four changes over dfs_solve:

  sym      stage 0 is placed only in one quadrant (the grid's reflections map carves to carves);
           below the diagonal too when the grid and every stage are square
  lattice  after every placement, each rectangle class (h, w) still has enough free marked cells
           (i % h == h-1 and j % w == w-1) for the stages not yet placed: the lattice pre-check,
           applied dynamically
  middle   stages are placed from both ends of the ring, 0, 1, n-1, 2, n-2, ..., so a ring that
           cannot close fails after a few placements, not after n
  order    candidate positions are tried compact-first: fewest free slices around the new
           rectangle, so the walk hugs what is already placed instead of wandering row-major

The prune from dfs_solve is kept in its two-ended form: the free region touching the forward end
must hold the remaining area and touch the backward end. Memo on (step, used, forward, backward).
dfs2_solve(R, C, shapes) -> (ok, nodes); DFS2_BUDGET (env) caps nodes, exceeding it returns None.
"""
import os

from carve_bench import dilate, positions

BUDGET = int(os.environ.get("DFS2_BUDGET", "0")) or None


def _marks(R, C, h, w):
    m = 0
    for i in range(R):
        for j in range(C):
            if i % h == h - 1 and j % w == w - 1:
                m |= 1 << (i * C + j)
    return m


def dfs2_solve(R, C, shapes, budget=BUDGET, check=True):
    n = len(shapes)
    N = R * C
    full = (1 << N) - 1
    pos = [positions(R, C, *s) for s in shapes]
    halo_bit = [dilate(R, C, 1 << b) for b in range(N)]
    halo = {}

    def halo_of(m):
        h = halo.get(m)
        if h is None:
            h = 0
            f = m
            while f:
                b = f & -f
                f ^= b
                h |= halo_bit[b.bit_length() - 1]
            h &= ~m
            halo[m] = h
        return h

    # symmetry: stage 0 in one quadrant
    h0, w0 = shapes[0]
    square_all = R == C and all(h == w for h, w in shapes)
    pos0 = []
    for i, m in enumerate(pos[0]):
        r, c = divmod(i, C - w0 + 1)
        if r > (R - h0) // 2 or c > (C - w0) // 2 or (square_all and r > c):
            continue
        pos0.append(m)
    pos[0] = pos0

    # placement order: 0, 1, n-1, 2, n-2, ...
    order, is_fwd = [0], [True]
    lo, hi = 1, n - 1
    while lo <= hi:
        order.append(lo)
        is_fwd.append(True)
        if hi != lo:
            order.append(hi)
            is_fwd.append(False)
        lo, hi = lo + 1, hi - 1

    # dynamic lattice: per class, marks mask and, per step, how many marks the unplaced stages need
    classes = sorted(set(shapes))
    marks = {cls: _marks(R, C, *cls) for cls in classes}
    need_at = []  # need_at[step][cls] = marks of class cls needed by stages order[step:]
    remaining = {cls: 0 for cls in classes}
    area_at = [0] * (n + 1)
    for step in range(n - 1, -1, -1):
        H, W = shapes[order[step]]
        for h, w in classes:
            remaining[(h, w)] += (H // h) * (W // w)
        need_at.insert(0, dict(remaining))
        area_at[step] = area_at[step + 1] + H * W
    need_at.append({cls: 0 for cls in classes})

    placed = [0] * n
    failed = set()
    nodes = 0
    out_of_budget = False

    def reach_from(free, src):
        seen = src & free
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

    def go(step, used, fwd, bwd):
        """fwd: mask of the last stage placed going forward (stage order[step-1] when it was a
        low index), bwd: the last placed going backward (high index); 0 when none yet."""
        nonlocal nodes, out_of_budget
        nodes += 1
        if budget and nodes > budget:
            out_of_budget = True
            return False
        if step == n:
            return bool(halo_of(fwd) & bwd)
        key = (step, used, fwd, bwd)
        if key in failed:
            return False
        free = full & ~used
        if bin(free).count("1") < area_at[step]:
            failed.add(key)
            return False
        need = need_at[step]
        for cls in classes:
            if need[cls] and bin(free & marks[cls]).count("1") < need[cls]:
                failed.add(key)
                return False
        k = order[step]
        prev_k, next_k = (k - 1) % n, (k + 1) % n
        touch_masks = [placed[nb] for nb in (prev_k, next_k) if nb != k and placed[nb]]
        if step > 0:
            # the unplaced stages form one chain from fwd to bwd through free slices
            src = halo_of(fwd) if fwd else halo_of(bwd)
            reach = reach_from(free, src)
            if bin(reach).count("1") < area_at[step]:
                failed.add(key)
                return False
            if fwd and bwd and not (halo_of(reach) & bwd):
                failed.add(key)
                return False
        cands = []
        for m in pos[k]:
            if m & used:
                continue
            ok = True
            for t in touch_masks:
                if not (halo_of(m) & t):
                    ok = False
                    break
            if not ok:
                continue
            free_after = free & ~m
            cands.append((bin(halo_of(m) & free_after).count("1"), m))
        cands.sort()
        for _, m in cands:
            placed[k] = m
            nf, nb = (m, bwd) if is_fwd[step] else (fwd, m)
            if go(step + 1, used | m, nf, nb):
                return True
            placed[k] = 0
        failed.add(key)
        return False

    ok = go(0, 0, 0, 0)
    if out_of_budget:
        return None, nodes
    if ok and check:
        _verify(R, C, shapes, placed, halo_of)
    return ok, nodes


def _verify(R, C, shapes, placed, halo_of):
    n = len(shapes)
    acc = 0
    for k, m in enumerate(placed):
        assert m and not (m & acc), f"stage {k} overlaps or is missing"
        acc |= m
        rows = {b // C for b in range(R * C) if m >> b & 1}
        cols = {b % C for b in range(R * C) if m >> b & 1}
        assert (len(rows), len(cols)) == tuple(shapes[k]) and bin(m).count("1") == shapes[k][0] * shapes[k][1]
        assert halo_of(m) & placed[(k + 1) % n], f"stage {k} does not touch {(k + 1) % n}"


if __name__ == "__main__":
    import sys
    import time

    from carve_bench import QUICK, dfs_solve

    for name, R, C, shapes in QUICK:
        t0 = time.perf_counter()
        ok1, n1 = dfs_solve(R, C, shapes)
        t1 = time.perf_counter()
        ok2, n2 = dfs2_solve(R, C, shapes)
        t2 = time.perf_counter()
        print(f"{name:<44} dfs {ok1} {n1:>8} nodes {t1-t0:7.3f}s | dfs2 {ok2} {n2:>8} nodes {t2-t1:7.3f}s", flush=True)
