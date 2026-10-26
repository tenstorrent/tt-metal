#!/usr/bin/env python3
"""stall.py WALK_LISTING [LAT]: from a walk.py -v listing (one instruction per line), estimate load-use stalls:
for each load, the distance (instructions) to the first instruction that reads its destination register; stall =
max(0, LAT - distance). Prints loads, total stall, and per-load details with -v."""
import re, sys
lines = [l for l in open(sys.argv[1]) if re.match(r"^\s*[0-9a-f]+ \S+", l) and not l.startswith("instr")]
LAT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
ins = []
for l in lines:
    p = l.split()
    if len(p) < 2 or p[0] == "instr": continue
    op = p[1]; args = p[2] if len(p) > 2 and ":" not in p[2] else ""
    ins.append((p[0], op, args))
LOADS = {"lw", "lh", "lhu", "lb", "lbu"}
total = 0; n = 0; det = []
for i, (ad, op, args) in enumerate(ins):
    if op not in LOADS: continue
    rd = args.split(",")[0]
    n += 1
    d = None
    for j in range(i + 1, len(ins)):
        a2 = ins[j][2]
        parts = re.split(r"[,()]", a2)
        srcs = parts[1:] if ins[j][1] not in ("sw", "sh", "sb") and not ins[j][1].startswith("b") else parts
        if ins[j][1] in ("sw", "sh", "sb"): srcs = parts
        if ins[j][1].startswith("b") or ins[j][1] in ("j",): srcs = parts
        if rd in [x.strip() for x in srcs if x.strip()]:
            d = j - i; break
        if parts and parts[0].strip() == rd and ins[j][1] not in ("sw", "sh", "sb") and not ins[j][1].startswith("b"):
            break  # overwritten before use
    if d is not None:
        st = max(0, LAT - d); total += st
        det.append((ad, op, args, d, st))
print(f"loads {n} stall {total} (lat {LAT})")
if "-v" in sys.argv:
    for x in det: print("  ", *x)
