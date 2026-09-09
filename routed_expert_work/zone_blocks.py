"""Per-block sums of a zone's entries for a few cores. usage: zone_blocks.py csv risc zone entries_per_block"""
import csv, sys
from collections import defaultdict

path, risc, zone, per = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(path) as f:
    head = f.readline()
    freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    next(rd)
    ev = defaultdict(list)
    for r in rd:
        if len(r) < 12:
            continue
        if r[3].strip() == risc and r[10].strip() == zone:
            ev[(int(r[1]), int(r[2]))].append((int(r[5]), r[11].strip()))
cores = sorted(ev)
for core in [cores[0], cores[len(cores) // 3], cores[len(cores) // 2], cores[-1]]:
    e = sorted(ev[core])
    d = []
    st = None
    for t, ty in e:
        if ty == "ZONE_START":
            st = t
        elif st is not None:
            d.append((t - st) / freq)
            st = None
    print(
        f"{zone} core {core}: n={len(d)} total={sum(d):.1f}us per-block={[round(sum(d[i:i+per]),1) for i in range(0,len(d),per)]}"
    )
