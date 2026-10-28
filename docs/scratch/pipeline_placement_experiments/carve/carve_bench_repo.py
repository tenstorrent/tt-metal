# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Repo-shaped carve cases for carve_bench.py (DFS vs CaDiCaL).

Grids are the largest slice grids the repo's MGDs declare (host_topology): 4x8, 8x4, 2x16, 8x8.
Shapes are in slices. Each ring: stage k touches stage k+1, last touches first.
Each solve is killed after CARVE_TIMEOUT seconds (default 30).
Usage: carve_bench_repo.py [quick|big]; big adds the full-grid rings of 1x1 stages on 7x7 and 8x8.
"""
import os
import random
import sys

os.environ.setdefault("CARVE_TIMEOUT", "30")
from carve_bench import lattice_check, run  # noqa: E402

S1, H, V, B, W = (1, 1), (1, 2), (2, 1), (2, 2), (2, 4)

cases = [
    # one-slice stages filling the grid: a Hamiltonian ring of slices
    ("4x8 ring of 32 x 1x1 (fills)", 4, 8, [S1] * 32),
    ("2x16 ring of 32 x 1x1 (fills)", 2, 16, [S1] * 32),
    ("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, [S1] * 25),
    ("1x8 ring of 3 x 1x1 (line: none)", 1, 8, [S1] * 3),
    # blitz-like mixes
    ("4x8: 1x1, 1x2 x6, 2x2 x4, 1x1", 4, 8, [S1] + [H] * 6 + [B] * 4 + [S1]),
    ("8x4: 1x1, 2x1 x6, 2x2 x4, 1x1", 8, 4, [S1] + [V] * 6 + [B] * 4 + [S1]),
    ("2x16: 1x1, 1x2 x15, 1x1", 2, 16, [S1] + [H] * 15 + [S1]),
    ("2x16: 1x1, 2x2 x7, 1x1 (fills 30/32)", 2, 16, [S1] + [B] * 7 + [S1]),
    ("4x8: 2x4 x4 (fills, ring of 4)", 4, 8, [W] * 4),
]
if sys.argv[1:] == ["big"]:
    cases += [
        ("8x8 ring of 64 x 1x1 (fills)", 8, 8, [S1] * 64),
        ("7x7 ring of 49 x 1x1 (odd: none)", 7, 7, [S1] * 49),
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
