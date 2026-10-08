#!/usr/bin/env python3
"""tilean.py <run.cyc> [ntiles]: per-tile packer DEST refusals (tile = 32 packer-0 grants), tile length, and other L1 traffic per tile."""
import sys


def n(v):
    try:
        return int(v, 2)
    except:
        return 0


f = open(sys.argv[1])
keys = f.readline().split()
ix = {k: i for i, k in enumerate(keys)}
tiles = []
cur = None
g0 = 0
cyc = 0


def new(c):
    return {
        "start": c,
        "ref": [0, 0, 0, 0],
        "l1wait": [0, 0, 0, 0],
        "t0ic": 0,
        "t1ic": 0,
        "t2ic": 0,
        "t0l1": 0,
        "t1l1": 0,
        "noc": 0,
    }


for line in f:
    v = line.split()
    cyc += 1
    rd = n(v[ix["xb_rden"]])
    ry = n(v[ix["xb_ready"]])
    if rd and cur is None:
        cur = new(cyc)
        tiles.append(cur)
    if cur is None:
        continue
    for p in range(4):
        if (rd >> p) & 1 and not (ry >> p) & 1:
            cur["ref"][p] += 1
        if v[ix[f"p{p}_l1_wren"]] == "1" and v[ix[f"p{p}_l1_ready"]] != "1":
            cur["l1wait"][p] += 1
    for t in range(3):
        if v[ix[f"t{t}_ic_rden"]] == "1":
            cur[f"t{t}ic"] += 1
    for t in range(2):
        if v[ix[f"t{t}_l1_rden"]] == "1" or v[ix[f"t{t}_l1_wren"]] == "1":
            cur[f"t{t}l1"] += 1
    if n(v[ix["noc_l1_rden"]]):
        cur["noc"] += 1
    if rd & 1 and ry & 1:
        g0 += 1
        if g0 % 32 == 0:
            cur = new(cyc + 1)
            tiles.append(cur)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print("tiles", len(tiles))
for i, t in enumerate(tiles):
    if i < N or i >= len(tiles) - 3:
        ln = (tiles[i + 1]["start"] - t["start"]) if i + 1 < len(tiles) else 0
        print(
            i,
            t["start"],
            ln,
            "ref",
            t["ref"],
            "l1wait",
            t["l1wait"],
            "ic",
            t["t0ic"],
            t["t1ic"],
            t["t2ic"],
            "l1",
            t["t0l1"],
            t["t1l1"],
            "noc",
            t["noc"],
        )
# summary: tile lengths histogram
from collections import Counter

print(
    "lengths",
    Counter(
        (tiles[i + 1]["start"] - tiles[i]["start"]) for i in range(len(tiles) - 1)
    ).most_common(6),
)
