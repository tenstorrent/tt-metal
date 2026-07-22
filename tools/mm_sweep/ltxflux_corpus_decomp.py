#!/usr/bin/env python3
# Corpus-wide causal decomposition for the LTX/FLUX Mt<=8 deep investigation.
# Per shape (config=None auto cfg): baseline wall + DRAM-ideal + delivered GB/s; TRISC compute-side split
# (matmul residual / in0-wait / in1-wait via DIAG_ZONES=16); reduction realizable ceiling (NO_REDUCE=8);
# and a bound-classification. All from the REAL production kernel (diag entry, mask 0 for baseline).
import json, os, sys, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import regime_a_diag_suite as ds, regime_a_bench as rb, zone_parse as zp

CORPUS = [(32,256,6144),(32,2048,512),(32,2048,1536),(32,2048,2048),(32,6144,1536),
          (32,6144,2304),(32,6144,4608),(32,6144,6144),(256,2048,1024)]


def trisc_split(csv):
    raw = zp.parse_raw(csv); us = lambda c: c/rb.FREQ*1e6
    niter = {(x,y): len(v) for (x,y,ri,z),v in raw.items() if z=="TRISC-KERNEL"}
    def totpc(x,y,zone):
        for (a,b,ri,zz),v in raw.items():
            if a==x and b==y and zz==zone: return sum(e-s for s,e in v)/max(niter.get((x,y),1),1)
        return 0.0
    tc = [(x,y) for (x,y,ri,z),v in raw.items() if z=="TRISC-KERNEL"]
    rows = []
    for (x,y) in set(tc):
        kern = st.median([e-s for (a,b,ri,z),v in raw.items() if a==x and b==y and z=="TRISC-KERNEL" for (s,e) in v][1:] or [0])
        i0, i1 = totpc(x,y,"Z_C_IN0WAIT"), totpc(x,y,"Z_C_IN1WAIT")
        rows.append((us(kern), us(i0), us(i1), us(kern)-us(i0)-us(i1)))
    return {"kern": st.median([r[0] for r in rows]), "in0wait": st.median([r[1] for r in rows]),
            "in1wait": st.median([r[2] for r in rows]), "matmul": st.median([r[3] for r in rows])}


def one(M,K,N):
    cfg = tuple(rb.auto_config(M,K,N))
    b = [ds.run_one(M,K,N,cfg,0)["wall_us"] for _ in range(3)]; b=[x for x in b if x]
    if not b: return {"shape":[M,K,N],"cfg":list(cfg),"error":"baseline fail"}
    base = st.median(b)
    ideal = rb.logical_bytes(M,K,N)/512e9*1e6
    gbps = rb.logical_bytes(M,K,N)/(base/1e6)/1e9
    z = ds.run_one(M,K,N,cfg,16); split = trisc_split(rb.BIN_CSV) if z.get("ok") else {}
    nr = [ds.run_one(M,K,N,cfg,8)["wall_us"] for _ in range(3)]; nr=[x for x in nr if x]
    red_ceiling = (1 - st.median(nr)/base)*100 if nr else None
    return {"shape":[M,K,N],"cfg":list(cfg),"Pk":cfg[1],"wall_us":round(base,2),"ideal_us":round(ideal,2),
            "wall_over_ideal":round(base/ideal,2),"gbps":round(gbps),"trisc":{k:round(v,2) for k,v in split.items()},
            "reduction_ceiling_pct":round(red_ceiling,1) if red_ceiling is not None else None}


if __name__ == "__main__":
    out = []
    for s in CORPUS:
        r = one(*s); out.append(r)
        t = r.get("trisc",{})
        print(f"{r['shape'][0]}x{r['shape'][1]}x{r['shape'][2]:<6} cfg={r['cfg']} Pk={r.get('Pk')} "
              f"wall={r.get('wall_us')} w/id={r.get('wall_over_ideal')} {r.get('gbps')}GB/s | "
              f"mm={t.get('matmul')} in0w={t.get('in0wait')} in1w={t.get('in1wait')} | redceil={r.get('reduction_ceiling_pct')}%", flush=True)
        json.dump(out, open("ltxflux_corpus_decomp.json","w"), indent=2)
    print("CORPUS DECOMP DONE")
