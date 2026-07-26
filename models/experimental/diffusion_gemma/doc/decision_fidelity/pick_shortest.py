#!/usr/bin/env python3
"""Pick the cheapest reproducers from the 60 TT-specific failures.

Cheapest = fewest reference tokens, since those collapse without needing a long context, so each
device run is short. Also reports how early TT collapsed, because a block-0 collapse reproduces in
one block.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])")

gpu = {}
for path, label in ((f"{HERE}/gpu_8192.jsonl", "8192"), (f"{HERE}/gpu_16k_retry.jsonl", "16k")):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["index"] not in gpu or label == "16k":
            gpu[row["index"]] = row

tt = {}
for line in open(f"{HERE}/tt_logs/hifi2_summary.txt", encoding="utf-8"):
    parts = line.split()
    if len(parts) < 3 or not parts[0].startswith("q"):
        continue
    tt[int(parts[0][1:])] = {"guard": int(parts[1].split("=")[1]), "blocks": int(parts[2].split("=")[1])}

rows = []
for index, t in tt.items():
    g = gpu.get(index)
    if g is None or not t["guard"] or not g.get("parsed"):
        continue
    rows.append((g.get("completion_tokens") or 0, index, t["blocks"], g.get("correct")))

rows.sort()
print(f"{len(rows)} TT-specific failures (reference answered, TT collapsed)\n")
print(f"{'ref tokens':>11} {'q':>5} {'TT blocks before collapse':>26} {'ref correct':>12}")
for tokens, index, blocks, correct in rows[:15]:
    print(f"{tokens:>11} {index:>5} {blocks:>26} {str(bool(correct)):>12}")

block0 = [r for r in rows if r[2] == 0]
print(f"\nof those, TT collapsed on BLOCK 0 (cheapest possible reproducer): {len(block0)}")
print(f"  {[r[1] for r in block0]}")
print(f"\nsuggested reproducer set (shortest 5): {[r[1] for r in rows[:5]]}")
