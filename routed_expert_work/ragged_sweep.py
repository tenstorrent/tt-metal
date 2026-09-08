# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Emit the ragged-count patterns for the per-expert sweep, with their predicted block counts.

The op quantises each expert to ceil(count/32) tiles and then to ceil(tiles/M_BLOCK) blocks with
M_BLOCK = 8, i.e. 256 tokens. Cost should track BLOCKS and NON-ZERO EXPERTS (each of which pays a
full weight read), not token count -- these patterns are chosen to separate those three.
"""
import sys

TILE, M_BLOCK = 32, 8


def blocks(c):
    return 0 if c == 0 else -(-(-(-c // TILE)) // M_BLOCK)


def describe(name, counts):
    nz = sum(1 for c in counts if c > 0)
    return {
        "name": name,
        "counts": counts,
        "total": sum(counts),
        "nonzero": nz,
        "blocks": sum(blocks(c) for c in counts),
        "tiles": sum(-(-c // TILE) for c in counts),
    }


def zipf(total, n):
    w = [1.0 / (i + 1) for i in range(n)]
    s = sum(w)
    out = [int(total * x / s) for x in w]
    out[0] += total - sum(out)
    return out


# Family A: TOTAL fixed at 2048 across 8 experts, distribution varied.
FAMILY_A = [
    describe("balanced", [256] * 8),
    describe("mild_25pct", [192, 224, 240, 256, 272, 288, 304, 272]),
    describe("moderate_3x", [128, 160, 192, 224, 288, 320, 352, 384]),
    describe("heavy_hot", [1024, 256, 256, 256, 64, 64, 64, 64]),
    describe("zipf", zipf(2048, 8)),
    describe("sparse4", [512, 512, 512, 512, 0, 0, 0, 0]),
    describe("sparse2", [1024, 1024, 0, 0, 0, 0, 0, 0]),
    describe("sparse1", [2048, 0, 0, 0, 0, 0, 0, 0]),
]

# Family B: uniform counts stepped across the 256-token block boundary.
FAMILY_B = [describe(f"unif_{c}", [c] * 8) for c in (224, 251, 256, 257, 288, 512)]

if __name__ == "__main__":
    fam = FAMILY_A if sys.argv[1:] == ["A"] else FAMILY_B if sys.argv[1:] == ["B"] else FAMILY_A + FAMILY_B
    if "--emit" in sys.argv:
        for p in fam:
            print(f"{p['name']} {','.join(str(c) for c in p['counts'])}")
    else:
        print(f"{'pattern':>14}{'total':>7}{'nz':>4}{'blocks':>8}{'tiles':>7}  counts")
        for p in fam:
            print(f"{p['name']:>14}{p['total']:>7}{p['nonzero']:>4}{p['blocks']:>8}{p['tiles']:>7}  {p['counts']}")
