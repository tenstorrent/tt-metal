#!/usr/bin/env python3
"""Generalized per-op extractor (prefix-configurable) for any blaze fused stage.
Same method as gen_csv.py but works for embedding / lm_head (not just GPTOSS decoder):
  pair START/END per (chip,core,RISC,zone) -> median per (chip,core,zone) across RISC/iters,
  busy core = median>0.3us, per-op = MAX over all busy cores (any chip).
Usage: gen_perop.py <profile_log_device.csv> <PREFIX> [clamp_us]
       PREFIX e.g. EMBEDDING_LAYER__  or  LM_HEAD_SAMPLING_LAYER__
"""
import sys, collections, statistics
FREQ=1350.0; BUSY=0.3
CSV=sys.argv[1]; PFX=sys.argv[2]; CLAMP=float(sys.argv[3]) if len(sys.argv)>3 else 40.0

raw=collections.defaultdict(list)
with open(CSV,errors="replace") as f:
    f.readline(); f.readline()
    for ln in f:
        p=ln.split(",")
        if len(p)<12: continue
        z=p[10].strip()
        if not (z.startswith(PFX) or z.startswith("INJECTED_FABRIC_BARRIER")): continue
        try: raw[(p[0],int(p[1]),int(p[2]),p[3].strip(),z)].append((int(p[5]),p[11].strip()))
        except ValueError: continue

pool=collections.defaultdict(list)
for (c,x,y,ri,z),evs in raw.items():
    evs.sort(); st=[a for a,t in evs if t=="ZONE_START"]; en=[a for a,t in evs if t=="ZONE_END"]
    for s,e in zip(st,en):
        d=(e-s)/FREQ
        if 0<d<=CLAMP: pool[(c,x,y,z)].append(d)
core={k:statistics.median(v) for k,v in pool.items() if v}

# per-op = max over all busy cores (any chip) of median; also record #busy cores, chips
byop=collections.defaultdict(lambda:{"vals":[],"chips":set(),"cores":0})
for (c,x,y,z),d in core.items():
    if d<=BUSY: continue
    o=byop[z]; o["vals"].append(d); o["chips"].add(c); o["cores"]+=1

def clean(z):
    return z.replace(PFX,"").replace("INJECTED_FABRIC_BARRIERBARRIER_","FABRIC_")
rows=[]
for z,o in byop.items():
    rows.append((clean(z), round(max(o["vals"]),3), o["cores"], len(o["chips"])))
rows.sort(key=lambda r:-r[1])
print(f"prefix={PFX}  distinct busy ops={len(rows)}")
print(f"{'op':56} {'time_us':>8} {'busy_cores':>10} {'chips':>5}")
for op,t,nc,ch in rows:
    print(f"  {op:54} {t:8.3f} {nc:10d} {ch:5d}")
# total (overlapping)
print(f"\nSigma(top compute op only): use the biggest as the stage compute cost")
