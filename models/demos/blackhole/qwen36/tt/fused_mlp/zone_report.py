"""Per-zone timing for each program in profile_log_device.csv (device 0): for every zone name, per-core
duration median/max and start offset (from the program's earliest kernel start), in us at the CSV's CHIP_FREQ."""
import csv, sys, re, collections, statistics as st
p = sys.argv[1]
hdr = open(p).readline(); mhz = float(re.search(r'CHIP_FREQ\[MHz\]:\s*(\d+)', hdr).group(1)) or 1350
rows = list(csv.reader(open(p)))[1:]
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
prog = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows[1:]:
    if len(r) < len(cols) - 1 or int(r[ix["PCIe slot"]]) != int(sys.argv[2] if len(sys.argv) > 2 else r[ix["PCIe slot"]]): continue
    rid = r[ix["run host ID"]].strip(); z = r[ix["zone name"]].strip(); typ = r[ix["type"]].strip()
    core = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]].strip()); t = int(r[ix["time[cycles since reset]"]])
    d = prog[rid][z].setdefault(core, [None, None])
    if typ == "ZONE_START": d[0] = t if d[0] is None else min(d[0], t)
    elif typ == "ZONE_END": d[1] = t if d[1] is None else max(d[1], t)
for rid, zs in prog.items():
    ks = [v[0] for z, c in zs.items() if z.endswith("-KERNEL") for v in c.values() if v[0]]
    ke = [v[1] for z, c in zs.items() if z.endswith("-KERNEL") for v in c.values() if v[1]]
    if not ks: continue
    t0 = min(ks); print(f"== run {rid} kernel span {(max(ke)-t0)/mhz:.1f} us ({len(set(k[:2] for c in zs.values() for k in c))} cores) @ {mhz} MHz")
    for z, c in sorted(zs.items(), key=lambda kv: min((v[0] or 0) for v in kv[1].values())):
        if z.endswith("-FW"): continue
        du = [(v[1]-v[0])/mhz for v in c.values() if v[0] and v[1]]; so = [(v[0]-t0)/mhz for v in c.values() if v[0]]
        eo = [(v[1]-t0)/mhz for v in c.values() if v[1]]
        if du: print(f"  {z:22} n={len(du):3} dur med {st.median(du):6.1f} max {max(du):6.1f} | start med {st.median(so):6.1f} | end med {st.median(eo):6.1f} max {max(eo):6.1f}")
