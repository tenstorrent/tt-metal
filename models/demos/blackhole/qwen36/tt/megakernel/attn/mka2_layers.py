"""Per-layer timing of an MKA2 run from profile_log_device.csv (last run in the file).
Layer L ends when the last STREAM core finishes its L-th MLP all-reduce stall (W-ARSTALL #L); layer L's duration is
end(L) - end(L-1) (layer 0 from the kernel start). Also per-layer spans of the main phases (k-th occurrence per core).
usage: mka2_layers.py csv [MHZ]"""
import csv, sys, collections, re
path = sys.argv[1]
with open(path) as f:
    hdr = f.readline(); rows = list(csv.reader(f))
m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", hdr); MHZ = float(sys.argv[2]) if len(sys.argv) > 2 else (float(m.group(1)) if m and int(m.group(1)) else 1350.0)
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
last = rows[-1][ix["run host ID"]].strip()
occ = collections.defaultdict(list)   # (zone, core) -> [(start, end)]
pend = {}
t0 = None
for r in rows[1:]:
    if len(r) < len(cols) - 1 or r[ix["run host ID"]].strip() != last: continue
    z = r[ix["zone name"]].strip(); core = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]].strip())
    t = int(r[ix["time[cycles since reset]"]]); typ = r[ix["type"]].strip()
    if z.endswith("-KERNEL") and typ == "ZONE_START": t0 = t if t0 is None else min(t0, t)
    k = (z, core)
    if typ == "ZONE_START": pend[k] = t
    elif k in pend: occ[k].append((pend.pop(k), t))
def layer_spans(zone):
    per = collections.defaultdict(lambda: [None, None])
    for (z, core), lst in occ.items():
        if z != zone: continue
        for L, (s, e) in enumerate(sorted(lst)):
            p = per[L]; p[0] = s if p[0] is None else min(p[0], s); p[1] = e if p[1] is None else max(p[1], e)
    return per
ends = layer_spans("W-ARSTALL")
NLr = len(ends)
print(f"freq {MHZ} MHz, run {last}, layers {NLr}")
prev = t0
for L in range(NLr):
    e = ends[L][1]
    print(f"  layer {L}: {(e - prev) / MHZ:8.1f} us  (ends at {(e - t0) / MHZ:8.1f} us)")
    prev = e
for z in ["W-MIX", "C-MIX", "W-SDPA", "C-SDPA", "R-KV", "R-RED", "W-WOAR", "C-GU", "C-DN", "W-HWAIT", "W-ARSTALL"]:
    sp = layer_spans(z)
    if sp: print(f"  {z:<10} " + "  ".join(f"L{L} {(s - t0) / MHZ:7.1f}-{(e - t0) / MHZ:7.1f} ({(e - s) / MHZ:6.1f})" for L, (s, e) in sorted(sp.items())))
