"""Gather slots per query chunk when the K/V brick grid is offset (phase) from the query grid.

Stride 1, window clamped inside the volume (NA rule). Per axis the slot count is the worst case over
chunk positions, like neighborhood_plan's gather_bricks. Total = product over axes.
"""
import itertools, sys


def axis_slots(n, win, b, chunk_b, phase):
    worst = 1
    c = b * chunk_b
    for q0 in range(0, n, c):
        q1 = min(q0 + c, n) - 1
        lo = min(max(q0 - win // 2, 0), n - win)
        hi = min(max(q1 - win // 2, 0), n - win) + win  # exclusive
        k_lo = (lo - phase) // b
        k_hi = (hi - 1 - phase) // b
        worst = max(worst, k_hi - k_lo + 1)
    return worst


def slots(vol, win, brick, chunk, phase):
    r = 1
    for a in range(3):
        r *= axis_slots(vol[a], win[a], brick[a], chunk[a], phase[a])
    return r


vol, win = (145, 272, 480), (11, 11, 11)
rows = []
for brick in [(2, 4, 4), (2, 8, 2), (2, 2, 8), (4, 4, 2), (4, 2, 4), (8, 2, 2), (1, 4, 8), (1, 8, 4)]:
    for chunk in [(1, 1, 1), (2, 1, 1)]:
        base = slots(vol, win, brick, chunk, (0, 0, 0))
        best = min(((slots(vol, win, brick, chunk, p), p) for p in itertools.product(*[range(x) for x in brick])))
        q = chunk[0] * chunk[1] * chunk[2]
        rows.append((brick, chunk, base, best[0], best[1], base / q, best[0] / q))
print("| brick | chunk | slots today | slots shifted | K phase | per Q brick today | shifted | gain |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f} | {r[6]:.1f} | {r[5]/r[6]:.2f}x |")
