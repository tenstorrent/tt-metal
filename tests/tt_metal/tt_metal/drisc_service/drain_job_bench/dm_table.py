# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Usage: python dm_table.py <outdir>. Data movement bandwidth per test (rows) and load (columns) from dm_reparse.py's
# .bw files and the host sweeps in the gtest logs: the mean of the reps and the change against L_none. Device-timed
# rows are bytes per cycle (multi-core tests: the slowest core, the harness's figure; packet-size sweeps: the best
# size); host rows are GB/s.
import re, glob, os, sys, collections

O = sys.argv[1]
conds = ["L_none", "L_local", "L_d2h1", "L_d2h2", "L_d2h7", "S_none", "S_local", "S_d2h1", "S_d2h7"]
vals = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{O}/r*_*.bw"):
    cond = os.path.basename(f).split("_", 1)[1][:-3]
    for line in open(f):
        tid, name, riscv, lo, hi, n = re.match(
            r"(\d+) \((.*?)\) (riscv_\d) min ([0-9.]+) max ([0-9.]+) n (\d+)", line
        ).groups()
        sweep = "Sweep" in name
        label = f"{tid} {name[:30]} {riscv}" + (" (best size)" if sweep else " (slowest core)" if int(n) > 1 else "")
        vals[label][cond].append(float(hi) if sweep else float(lo))
for f in glob.glob(f"{O}/r*_*.log"):
    cond = os.path.basename(f).split("_", 1)[1][:-4]
    for d, buf, bw in re.findall(r"PCIe Host (Read|Write) \(\w+\): buf=(\d+) bytes.*BW=([0-9.]+) GB/s", open(f).read()):
        vals[f"host {d.lower()} {int(buf) // 1024} KiB GB/s"][cond].append(float(bw))


def order(k):
    return (1, k.split()[1], int(k.split()[2])) if k.startswith("host") else (0, int(k.split()[0]), "")


print(f"{'test':58s}" + "".join(f"{c:>14s}" for c in conds))
for k in sorted(vals, key=order):
    base = vals[k]["L_none"]
    b = sum(base) / len(base) if base else None
    cells = []
    for c in conds:
        v = vals[k][c]
        if not v:
            cells.append("-")
            continue
        m = sum(v) / len(v)
        cells.append(f"{m:6.2f}" + (f" {100 * (m / b - 1):+4.0f}%" if b and c != "L_none" else ""))
    print(f"{k:58s}" + "".join(f"{x:>14s}" for x in cells))
