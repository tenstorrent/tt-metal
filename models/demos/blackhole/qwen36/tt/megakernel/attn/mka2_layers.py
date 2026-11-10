"""Per-layer timing of an MKA2 run from profile_log_device.csv (last run in the file).
Layer L ends when the last STREAM core finishes its L-th MLP all-reduce stall (W-ARSTALL #L; MK_REAL_AR: W-ARWAIT #L); layer L's duration is
end(L) - end(L-1) (layer 0 from the kernel start). Also per-layer spans of the main phases (k-th occurrence per core).
usage: mka2_layers.py csv [MHZ]"""
import csv, sys, collections, re
path = sys.argv[1]
with open(path) as f:
    hdr = f.readline(); rows = list(csv.reader(f))
m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", hdr); MHZ = float(sys.argv[2]) if len(sys.argv) > 2 else (float(m.group(1)) if m and int(m.group(1)) else 1350.0)
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
last = rows[-1][ix["run host ID"]].strip()
# mesh runs: one device at a time (MKA_DEV = PCIe slot / device id; default: the lowest in the file)
import os
devs = sorted({r[ix["PCIe slot"]].strip() for r in rows[1:] if len(r) >= len(cols) - 1}, key=lambda d: int(d) if d.isdigit() else 0)
DEVSEL = os.environ.get("MKA_DEV", devs[0] if devs else "0")
# the last run of the selected device (mesh runs carry a different runtime id per device)
_ids = [r[ix["run host ID"]].strip() for r in rows[1:] if len(r) >= len(cols) - 1 and r[ix["PCIe slot"]].strip() == DEVSEL]
last = _ids[-1] if _ids else last
occ = collections.defaultdict(list)   # (zone, core) -> [(start, end)]
pend = {}
t0 = None
for r in rows[1:]:
    if len(r) < len(cols) - 1 or r[ix["run host ID"]].strip() != last: continue
    if r[ix["PCIe slot"]].strip() != DEVSEL: continue
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
if not ends: ends = layer_spans("W-ARWAIT")   # MK_REAL_AR: layer L ends when the last STREAM core has its row's x of L + 1
NLr = len(ends)
print(f"freq {MHZ} MHz, run {last}, device {DEVSEL} of {devs}, layers {NLr}")
prev = t0
for L in range(NLr):
    e = ends[L][1]
    print(f"  layer {L}: {(e - prev) / MHZ:8.1f} us  (ends at {(e - t0) / MHZ:8.1f} us)")
    prev = e
for z in ["W-MIX", "C-MIX", "W-SDPA", "C-SDPA", "R-KV", "R-RED", "W-WOAR", "C-GU", "C-DN", "W-HWAIT", "W-VX", "W-VMD", "W-OREL", "C-OSUM", "W-ARSTALL", "W-AR", "C-AR", "W-ARWAIT", "R-ARDLVW", "R-ARDLV", "PS-O-RSWAIT", "PS-O-REDUCE", "PS-G-AGWAIT", "PS-DELIVER"]:
    sp = layer_spans(z)
    if sp: print(f"  {z:<10} " + "  ".join(f"L{L} {(s - t0) / MHZ:7.1f}-{(e - t0) / MHZ:7.1f} ({(e - s) / MHZ:6.1f})" for L, (s, e) in sorted(sp.items())))
