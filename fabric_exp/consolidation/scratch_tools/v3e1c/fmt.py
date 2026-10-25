import re, sys
rows = {}
vs = sys.argv[2:]
for l in open(sys.argv[1]):
    m = re.match(r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+instr (\d+)\s+taken (\d+)\+1(\s+call/ret (\d+))?", l)
    if m: rows[(m.group(1), m.group(2), m.group(3), m.group(4))] = (int(m.group(5)), int(m.group(6)), int(m.group(8) or 0))
for c in ["W", "NS", "1D"]:
    print(f"== {c}")
    print("lvl  scen     " + "".join(f"{v:>12s}" for v in vs))
    for l in ["O3", "O2", "Os"]:
        for s in ["write", "scatter", "atomic", "fused", "fusedfl"]:
            cells = []
            for v in vs:
                r = rows.get((c, l, v, s)); cells.append((f"{r[0]}/{r[1]}" + (f"c{r[2]}" if r[2] else "")) if r else "-")
            print(f"{l:4s} {s:8s} " + "".join(f"{x:>12s}" for x in cells))
