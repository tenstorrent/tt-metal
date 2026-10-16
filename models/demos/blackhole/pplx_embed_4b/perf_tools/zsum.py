import re
import sys

blocks, cur, risc = {}, None, None
for l in open(sys.argv[1]):
    m = re.match(r"=== (\S+) (\S+) flag=(\d) ===", l)
    if m:
        cur = (m.group(2), m.group(3))
        continue
    m = re.match(r"(TRISC_\d) (\w+):", l)
    if m:
        risc = m.group(2)
        continue
    m = re.match(r"\s+(.+?)\s+n=\s*(\d+)\s+mean=\s*(\d+)", l)
    if m and cur:
        blocks[(cur[0], cur[1], risc, m.group(1).strip())] = int(m.group(3))
arms = sorted({k[0] for k in blocks})
zones = [
    "P1 QK+exp rowgroup #1",
    "P1 QK+exp rowgroup #2",
    "P1 reduce max #1",
    "P1 reduce max #2",
    "P2 exp rg1 + PV rg0 #1",
    "P2 PV rg1 #1",
    "P2 normalize #1",
    "P2 normalize #2",
    "unit",
]
zones = [
    z for z in zones if any((a, f, r, z) in blocks for a in arms for f in "01" for r in ("math", "pack", "unpack"))
]
for a in arms:
    print(f"\n### {a}   (mean cycles per work unit)")
    print(
        f"{'zone':26s} "
        + "".join(f"{r+' '+f:>13s}" for r in ("math", "pack") for f in ("off", "on"))
        + "   math Δ   pack Δ"
    )
    for z in zones:
        row = [blocks.get((a, f, r, z)) for r in ("math", "pack") for f in ("0", "1")]
        if any(v is None for v in row):
            continue
        mo, mn, po, pn = row
        print(f"{z:26s} " + "".join(f"{v:13d}" for v in row) + f" {mn-mo:+8d} {pn-po:+8d}")
