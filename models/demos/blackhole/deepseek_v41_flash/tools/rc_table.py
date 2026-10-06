#!/usr/bin/env python3
"""usage: rc_table.py <session.log>  -- markdown table of the RECONFIGURE lines (free DRAM per bank before / weights only / after, timings, L1) and the first build."""
import re
import sys

txt = open(sys.argv[1], errors="replace").read()
m = re.search(
    r"MEMLOG model built\s+allocated\s+([\d.]+) MiB/bank\s+free\s+([\d.]+)\s+largest free block\s+([\d.]+)", txt
)
if m:
    print(
        f"first build ('model built', U=1): alloc {m.group(1)} MiB/bank, free {m.group(2)}, largest block {m.group(3)}\n"
    )
print(
    "| reconfigure | release s | rebuild s | free MiB/bank before | weights only: alloc / free | after: alloc / free / largest block | L1 B/bank released / after |"
)
print("|---|---|---|---|---|---|---|")
pat = re.compile(
    r"RECONFIGURE B (\d+) \(U=(\d+), ctx (\d+)\) -> B (\d+) \(U=(\d+), ctx (\d+)\): release ([\d.]+) s, rebuild ([\d.]+) s, total ([\d.]+) s; "
    r"DRAM MiB/bank before \[alloc ([\d.]+) / free ([\d.]+) / largest block ([\d.]+)\] \| weights only \[alloc ([\d.]+) / free ([\d.]+) / largest block ([\d.]+)\] \| "
    r"after \[alloc ([\d.]+) / free ([\d.]+) / largest block ([\d.]+)\] \| L1 B/bank allocated / largest free: before (\d+) / \d+, released (\d+) / \d+ .*?after (\d+) /"
)
for g in pat.findall(txt):
    print(
        f"| B {g[0]} -> {g[3]} (ctx {g[2]} -> {g[5]}) | {g[6]} | {g[7]} | {g[10]} | {g[12]} / {g[13]} | {g[15]} / {g[16]} / {g[17]} | {g[19]} / {g[20]} |"
    )
