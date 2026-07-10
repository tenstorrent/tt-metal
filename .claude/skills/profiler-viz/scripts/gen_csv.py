#!/usr/bin/env python3
"""Per-op CSV tables from a blaze device-profiler report (profile_log_device.csv).

OUTPUT  ops.csv       op,phase,time_us,pct_wall,busy_cores,n_chips,prev,next
        coregrid.csv  op,chip,core_x,core_y,dur_us  (busy cores only)

METHOD (how per-op time is produced)
  1. Rows are DeviceZoneScopedN markers: PCIe slot, core_x, core_y, RISC,
     timer_id, time[cycles], ..., zone_name, type(ZONE_START|END).
  2. Per (chip,core,RISC,zone): pair each START with the next END ->
     duration = (end-start)/FREQ_MHZ us. Drop durations > CLAMP_US (mis-pairs).
  3. Per (chip,core,zone): POOL all instances across RISCs, take the MEDIAN
     (robust to the DM/other RISCs that sit and wait inside the zone).
  4. BUSY core = median > BUSY_US.
  5. Reduce-root chip = chip with the largest MOE__REDUCE_TO_ONE core.
  6. Per-op time = MAX over the reduce-root chip busy cores (the op critical
     core, where the real compute lands; peer chips wait).
  7. WALL = median STAGE_CHECKPOINT duration (per-iteration wall marker); the
     per-core median across the iterations drops the layer-0 cold path. If no
     STAGE_CHECKPOINT zones are present, WALL falls back to 67.5 (nominal) or
     --wall; pct_wall then becomes attribution-only.
  8. pct_wall = time / wall.
Layer prefix (GPTOSS_GLOBAL_LAYER__ / GPTOSS_WINDOWED_LAYER__) is auto-detected.
"""
import argparse, collections, statistics, csv, os

FREQ=1350.0; BUSY_US=0.3; CLAMP_US=40.0; CHECKPOINT="STAGE_CHECKPOINT"
PFX=""  # auto-detected in parse()

def phase_of(z):
    zz=z.replace(PFX,"")
    if "REDUCE_TO_ONE" in zz: return "comm"
    if "CB_RECONFIG" in zz:   return "infra"
    if zz.startswith("ATTN"): return "attn"
    if zz.startswith("MOE"):  return "moe"
    return "comm"

def parse(path, clamp=40.0, cpcap=500.0):
    global PFX
    raw=collections.defaultdict(list)
    for ln in open(path).readlines()[2:]:
        p=[x.strip() for x in ln.split(",")]
        if len(p)<12: continue
        z=p[10]
        if not (z.startswith("GPTOSS") or z.startswith("INJECTED") or z==CHECKPOINT): continue
        raw[(p[0],int(p[1]),int(p[2]),p[3],z)].append((int(p[5]),p[11]))
    # auto-detect layer prefix from the reduce-to-one zone
    for (_,_,_,_,z) in raw:
        i=z.find("MOE__REDUCE_TO_ONE")
        if i>0 and z.startswith("GPTOSS"): PFX=z[:i]; break
    pool=collections.defaultdict(list); firststart=collections.defaultdict(list)
    for (c,x,y,risc,z),evs in raw.items():
        evs.sort(); st=[a for a,t in evs if t=="ZONE_START"]; en=[a for a,t in evs if t=="ZONE_END"]
        if st: firststart[z].append(min(st))
        cap=cpcap if z==CHECKPOINT else clamp
        for s,e in zip(st,en):
            d=(e-s)/FREQ
            if 0<d<=cap: pool[(c,x,y,z)].append(d)
    core={k:statistics.median(v) for k,v in pool.items() if v}
    return core, firststart

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv")
    ap.add_argument("--wall",type=float,default=0.0,help="override; 0 = derive from STAGE_CHECKPOINT")
    ap.add_argument("--clamp",type=float,default=40.0,help="per-op mis-pair clamp (us)")
    ap.add_argument("--cpcap",type=float,default=500.0,help="STAGE_CHECKPOINT mis-pair clamp (us)")
    ap.add_argument("--out",default=".")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    core,firststart=parse(a.csv, a.clamp, a.cpcap)
    # reduce-root chip = chip with the largest MOE__REDUCE_TO_ONE core
    rr=collections.defaultdict(float)
    for (c,x,y,z),d in core.items():
        if z==PFX+"MOE__REDUCE_TO_ONE": rr[c]=max(rr[c],d)
    root=max(rr,key=rr.get) if rr else None
    # wall = STAGE_CHECKPOINT on the reduce-root chip (lock-stepped critical path):
    # per-core median drops the layer-0 cold iteration; median over root-chip cores
    # is robust to buffer-wrap corruption in the idle-core tail.
    rootcps=[d for (c,x,y,z),d in core.items() if z==CHECKPOINT and c==root]
    allcps=[d for (c,x,y,z),d in core.items() if z==CHECKPOINT]
    cps=rootcps or allcps
    wall=a.wall if a.wall>0 else (round(statistics.median(cps),3) if cps else 67.5)
    byop=collections.defaultdict(lambda:{"root":[],"chips":set(),"cores":0}); grid=collections.defaultdict(list)
    for (c,x,y,z),d in core.items():
        if z==CHECKPOINT or d<=BUSY_US: continue
        byop[z]["chips"].add(c); byop[z]["cores"]+=1; grid[z].append((c,x,y,round(d,2)))
        if c==root: byop[z]["root"].append(d)
    order=sorted(byop, key=lambda z: min(firststart[z]) if firststart[z] else 0)
    seq=[z.replace(PFX,"") for z in order]
    with open(os.path.join(a.out,"ops.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["op","phase","time_us","pct_wall","busy_cores","n_chips","prev","next"])
        for i,z in enumerate(order):
            rb=byop[z]["root"]; t=round(max(rb),3) if rb else 0.0
            w.writerow([z.replace(PFX,""),phase_of(z),t,round(t/wall*100,2),byop[z]["cores"],len(byop[z]["chips"]),
                        seq[i-1] if i>0 else "", seq[i+1] if i<len(seq)-1 else ""])
    with open(os.path.join(a.out,"coregrid.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["op","chip","core_x","core_y","dur_us"])
        for z,cells in grid.items():
            for c,x,y,d in sorted(cells): w.writerow([z.replace(PFX,""),c,x,y,d])
    print("PFX=%s | reduce-root=PCIe %s | ops=%d | WALL=%.3f | checkpoint_samples=%d"%(PFX,root,len(order),wall,len(cps)))

if __name__=="__main__": main()
