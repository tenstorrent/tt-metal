# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routing of sparse_kv_dispatch (host model of the rule its kernels implement), for test_sparse_kv_dispatch.py.

Ported from sparse_dispatch_2d (ttnn/cpp/ttnn/operations/experimental/moe_block/common/mesh_routing.cpp span_entry,
sparse_dispatch_2d_common.hpp): a row's tree is two legs, first along one axis (its order: 0 X first, 1 Y first) over
every line holding a destination, then from each chip of that leg along the other axis over the destinations in its
line. A leg's span on a ring takes the fewest edges, then the shallowest (smallest max(forward, backward)), and of the
at most two left, the one the row's tie bit picks; on a line it is forced. The order takes the fewer total edges, ties
by another bit of the row hash. Row hash: row index x Knuth's constant (2^32 / golden ratio), bit 30 for every ring tie,
bit 31 for the order.
"""

KNUTH = 2654435761
TIE_BIT, ORDER_BIT = 30, 31


def hbit(g, b):
    return ((g * KNUTH) & 0xFFFFFFFF) >> b & 1


def span(n, wrap, pos, line_mask, tie):
    """(forward hops, backward hops) from pos covering the set positions of line_mask (pos itself ignored)."""
    m = line_mask & ~(1 << pos) & ((1 << n) - 1)
    if not m:
        return 0, 0
    ts = [t for t in range(n) if m >> t & 1]
    if not wrap:
        return max(max(ts) - pos, 0), max(pos - min(ts), 0)
    assert n >= 3, "a wrapped axis needs at least 3 chips (a 2-chip axis is a line)"
    offs = [(t - pos) % n for t in ts]
    cands = [(a, b) for a in range(n) for b in range(n - a) if all(o <= a or n - o <= b for o in offs)]
    m1 = min(a + b for a, b in cands)
    cands = [c for c in cands if sum(c) == m1]
    m2 = min(max(c) for c in cands)
    cands = sorted(c for c in cands if max(c) == m2)
    return cands[tie if len(cands) > 1 else 0]


class Routing:
    def __init__(self, R, C, wrap_x, wrap_y):
        assert not (wrap_x and C == 2) and not (wrap_y and R == 2), "2-chip axes are lines"
        self.R, self.C, self.N = R, C, R * C
        self.wrap = (wrap_x, wrap_y)

    def coord(self, i):
        return divmod(i, self.C)

    def cols_of(self, mask):
        cm = 0
        for r in range(self.R):
            cm |= (mask >> (r * self.C)) & ((1 << self.C) - 1)
        return cm

    def rows_of(self, mask):
        return sum(1 << r for r in range(self.R) if (mask >> (r * self.C)) & ((1 << self.C) - 1))

    def rows_in_col(self, mask, c):
        return sum(1 << r for r in range(self.R) if mask >> (r * self.C + c) & 1)

    def cols_in_row(self, mask, r):
        return (mask >> (r * self.C)) & ((1 << self.C) - 1)

    def first_leg(self, axis_x, r, c, mask, g):
        if axis_x:
            return span(self.C, self.wrap[0], c, self.cols_of(mask), hbit(g, TIE_BIT))
        return span(self.R, self.wrap[1], r, self.rows_of(mask), hbit(g, TIE_BIT))

    def turn(self, into_x, r, c, mask, g):
        if into_x:
            return span(self.C, self.wrap[0], c, self.cols_in_row(mask, r), hbit(g, TIE_BIT))
        return span(self.R, self.wrap[1], r, self.rows_in_col(mask, c), hbit(g, TIE_BIT))

    def tree(self, owner, mask, order, g):
        """Directed edges ((r, c) -> (r, c)) and the deepest leaf of the row's tree in this order."""
        r0, c0 = self.coord(owner)
        f, b = self.first_leg(order == 0, r0, c0, mask, g)
        edges, depth = [], 0
        for k in range(-b, f + 1):
            if order == 0:
                c = (c0 + k) % self.C
                pos_first = (r0, c)
                tf, tb = self.turn(False, r0, c, mask, g)
            else:
                r = (r0 + k) % self.R
                pos_first = (r, c0)
                tf, tb = self.turn(True, r, c0, mask, g)
            depth = max(depth, abs(k) + max(tf, tb))
            if k != 0:  # the first-leg edge into this chip
                step = 1 if k > 0 else -1
                prev_k = k - step
                if order == 0:
                    edges.append(((r0, (c0 + prev_k) % self.C), pos_first))
                else:
                    edges.append((((r0 + prev_k) % self.R, c0), pos_first))
            rr, cc = pos_first
            for s in range(1, tf + 1):
                if order == 0:
                    edges.append((((rr + s - 1) % self.R, cc), ((rr + s) % self.R, cc)))
                else:
                    edges.append(((rr, (cc + s - 1) % self.C), (rr, (cc + s) % self.C)))
            for s in range(1, tb + 1):
                if order == 0:
                    edges.append((((rr - s + 1) % self.R, cc), ((rr - s) % self.R, cc)))
                else:
                    edges.append(((rr, (cc - s + 1) % self.C), (rr, (cc - s) % self.C)))
        return edges, depth

    def order_of(self, owner, mask, g):
        e0 = len(self.tree(owner, mask, 0, g)[0])
        e1 = len(self.tree(owner, mask, 1, g)[0])
        return 0 if e0 < e1 else 1 if e1 < e0 else hbit(g, ORDER_BIT)
