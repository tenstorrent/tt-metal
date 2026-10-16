import re
import sys

d, cur, risc = {}, None, None
for l in open(sys.argv[1]):
    m = re.match(r"=== (\S+) (\S+) flag=(\d) ===", l)
    if m:
        cur = (m.group(2), m.group(3))
        risc = None
        continue
    m = re.match(r"(TRISC_\d) (\w+):", l)
    if m:
        risc = m.group(2)
        continue
    m = re.match(r"\s+(.+?)\s+n=\s*(\d+)\s+mean=\s*(\d+)\s+median=\s*(\d+)", l)
    if m and cur:
        if risc is None:
            risc = "math"  # first block of a run has no header line in the tail-trimmed log
        d[(cur[0], cur[1], risc, m.group(1).strip())] = (int(m.group(2)), int(m.group(3)))
arms = sorted({k[0] for k in d})
want = [
    ("math", "SUB"),
    ("math", "Q@KT MM+Pack"),
    ("pack", "EXP"),
    ("pack", "PACK SUB_EXP"),
    ("math", "ROW_NORM"),
    ("pack", "ROW_NORM"),
    ("math", "NORM_MATMUL_RECIP"),
    ("math", "TRISC-KERNEL"),
    ("pack", "TRISC-KERNEL"),
]
for a in arms:
    print(f"\n### {a}  (mean cycles per zone occurrence)")
    print(f"{'thread / zone':34s} {'off':>9s} {'on':>9s} {'delta':>9s}   n(off)")
    for r, z in want:
        o, n = d.get((a, "0", r, z)), d.get((a, "1", r, z))
        if not o or not n:
            continue
        print(f"{r+' '+z:34s} {o[1]:9d} {n[1]:9d} {n[1]-o[1]:+9d}   {o[0]}")
