"""Repo-shaped carve cases for carve_bench.py (DFS vs CaDiCaL), without editing it.

Grids are the largest slice grids the repo's MGDs declare (host_topology): 4x8, 8x4, 2x16, 8x8.
Shapes are in slices. Each ring: stage k touches stage k+1, last touches first.
"""
import os
import random

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "carve_bench.py")).read()
exec(src.split("S1, B, W = ")[0])  # dfs_solve, sat_solve, run, positions, dilate, TIMEOUT
exec("def lattice_check" + src.split("def lattice_check")[1].split("\nif len(sys.argv)")[0])

TIMEOUT = 30
S1, H, V, B, W = (1, 1), (1, 2), (2, 1), (2, 2), (2, 4)

cases = [
    # one-slice stages filling the grid: a Hamiltonian ring of slices
    ("4x8 ring of 32 x 1x1 (fills)", 4, 8, [S1] * 32),
    ("2x16 ring of 32 x 1x1 (fills)", 2, 16, [S1] * 32),
    ("8x8 ring of 64 x 1x1 (fills)", 8, 8, [S1] * 64),
    ("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, [S1] * 25),
    ("7x7 ring of 49 x 1x1 (odd: none)", 7, 7, [S1] * 49),
    ("1x8 ring of 3 x 1x1 (line: none)", 1, 8, [S1] * 3),
    # blitz-like mixes
    ("4x8: 1x1, 1x2 x6, 2x2 x4, 1x1", 4, 8, [S1] + [H] * 6 + [B] * 4 + [S1]),
    ("8x4: 1x1, 2x1 x6, 2x2 x4, 1x1", 8, 4, [S1] + [V] * 6 + [B] * 4 + [S1]),
    ("2x16: 1x1, 1x2 x15, 1x1", 2, 16, [S1] + [H] * 15 + [S1]),
    ("2x16: 1x1, 2x2 x7, 1x1 (fills 30/32)", 2, 16, [S1] + [B] * 7 + [S1]),
    ("4x8: 2x4 x4 (fills, ring of 4)", 4, 8, [W] * 4),
]
random.seed(0)
for i in range(6):
    R, C = random.choice([(4, 8), (8, 4), (2, 16), (8, 8)])
    shapes, area = [S1], 1
    while True:
        s = random.choice([S1, S1, H, V, B])
        if s[0] > R or s[1] > C or area + s[0] * s[1] > R * C:
            break
        shapes.append(s)
        area += s[0] * s[1]
    cases.append((f"random {R}x{C}: {len(shapes)} stages", R, C, shapes))

for c in cases:
    why = lattice_check(*c[1:])
    if why:
        print(f"{c[0]:<44} lattice rejects: {why}", flush=True)
        continue
    run(*c)
