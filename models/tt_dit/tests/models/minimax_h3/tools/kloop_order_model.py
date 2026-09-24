# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Cycle model of the 2x2 K loop for ranking MVMUL schedules (plan §1). Not a predictor: fitted to the measured no-pack rows."""

ORDERS = {
    0: [[0, 1, 2, 3]],
    1: [[0, 1, 2, 3], [0, 2, 1, 3]],
    2: [[0, 2, 1, 3]],
    3: [[0, 1, 2, 3], [3, 2, 1, 0]],
    4: [[0, 1, 3, 2]],
    5: [[0, 1, 3, 2], [0, 2, 3, 1]],
    6: [[0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 1, 0], [3, 1, 2, 0]],
    7: [[0, 3, 1, 2]],
    8: [[0, 2, 1, 3], [0, 1, 2, 3]],
}


def simulate(orders, steps=700, mvmul=32, o=16, Lc=14, Lv=14, W=32):
    """o: per-MVMUL fixed overhead (dst write, MOP start, bank op); Lc: release->unpacker may write; W: write; Lv: write->valid."""
    P = len(orders)
    t = 0.0
    valid = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 0,
    }  # time each bank's current tile becomes valid (initially preloaded)
    unp_free = {"A": 0, "B": 0}
    releases = []  # (time, src, tile-slot)
    total = 0
    for k in range(steps):
        ord_ = orders[k % P]
        pairs = [(p >> 1, p & 1) for p in ord_]
        # last use index per (src,tile) in this step
        last = {}
        for i, (a, b) in enumerate(pairs):
            last[("A", a)] = i
            last[("B", b)] = i
        step_start = t
        for i, (a, b) in enumerate(pairs):
            start = max(t + o, valid[("A", a)], valid[("B", b)])
            t = start + mvmul
            for src, tile in (("A", a), ("B", b)):
                if last[(src, tile)] == i:
                    # released now; the NEXT step's tile in this bank can be written
                    ws = max(t + Lc, unp_free[src])
                    we = ws + W
                    unp_free[src] = we
                    valid[(src, tile)] = we + Lv
        total = t
    return total / steps


def fit():
    # fit (o, Lc+Lv) to measured no-pack rows: legacy row-major 198, mirror 193 (floor); floor = 4*(32+o) -> o ~ 16
    best = None
    for o in range(10, 22):
        for L in range(0, 40, 2):
            r0 = simulate(ORDERS[0], o=o, Lc=L // 2, Lv=L - L // 2)
            r1 = simulate(ORDERS[1], o=o, Lc=L // 2, Lv=L - L // 2)
            err = (r0 - 198) ** 2 + (r1 - 193) ** 2
            if best is None or err < best[0]:
                best = (err, o, L, r0, r1)
    return best


err, o, L, r0, r1 = fit()
print(
    f"fit: per-MVMUL overhead o={o}, refill latency Lc+Lv={L}  -> row-major {r0:.1f} (meas 198), mirror {r1:.1f} (meas 193)"
)
print(f"{'sched':>5s} {'period':>6s} {'model cyc/step (no pack)':>26s}  orders")
for s, ordn in ORDERS.items():
    print(f"{s:>5d} {len(ordn):>6d} {simulate(ordn,o=o,Lc=L//2,Lv=L-L//2):>26.1f}  {ordn}")
