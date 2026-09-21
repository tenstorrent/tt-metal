# Fit-based vs witness-based row-scale readout on the same device outputs, real k (gamma) and v (no gamma).
import torch, ttnn, os
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    SD=os.environ["SD"]; eps=1e-6
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    def rne(x): return x.to(torch.bfloat16).double()
    for f in ("phn_001_k.pt","phn_002_v.pt"):
        d=torch.load(f"{SD}/phndump/{f}"); x0=d["x"].double().squeeze(); w_t=d["w"]; R,W=x0.shape
        gam=w_t.double().reshape(1,-1) if w_t is not None else torch.ones(1,W,dtype=torch.float64)
        print(f"CK ==== {f} gamma={w_t is not None}")
        print(f"CK {'variant':22s} | {'gen: fit bias':>13s} {'witness bias':>12s} {'|fit-wit| med':>13s} | {'nat: fit bias':>13s} {'witness bias':>12s} {'|fit-wit| med':>13s} | {'fit snapped rows +1 gen/nat':>26s} {'witness rows +1 gen/nat':>24s}")
        for label,pos in (("untouched (fit only)",None),("witness at pos 0",0),("witness at pos 100",100),("witness at pos 255",255)):
            x=x0.clone()
            if pos is not None: x[:,pos]=1.0
            xb=x.to(torch.bfloat16); xd=xb.double(); base=xd*gam
            s_ex=1/torch.sqrt(xd.pow(2).mean(-1)+eps); u=ulp(s_ex)
            X=ttnn.from_torch(xb.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            Wt=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row=f"CK {label:22s} |"; snap=[]; wit=[]
            for fn in (ttnn.rms_norm, ttnn._native_rms_norm):
                kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
                if Wt is not None: kw["weight"]=Wt
                o=ttnn.to_torch(fn(X,**kw)).double().reshape(R,W)
                fit=(o*base).sum(-1)/(base*base).sum(-1); e_fit=(fit-s_ex)/u; e_snap=(rne(fit)-s_ex)/u
                snap.append((e_snap.round()>=1).double().mean().item())
                if pos is not None:
                    gw=gam[0,pos].item(); s_w=o[:,pos]/gw          # witness output = 1*gamma*s -> divide by gamma (exact only if gamma is a power of two)
                    e_w=(s_w-s_ex)/u; wit.append((e_w.round()>=1).double().mean().item())
                    row+=f" {e_fit.mean().item():+13.3f} {e_w.mean().item():+12.3f} {(e_fit-e_w).abs().median().item():13.3f} |"
                else:
                    row+=f" {e_fit.mean().item():+13.3f} {'—':>12s} {'—':>13s} |"
            row+=f" {snap[0]:.2f} / {snap[1]:.2f}{'':16s} " + (f"{wit[0]:.2f} / {wit[1]:.2f}" if wit else "—")
            print(row, flush=True)
        if w_t is not None: print(f"CK gamma at pos 0/100/255: {gam[0,0].item():.4f} {gam[0,100].item():.4f} {gam[0,255].item():.4f} (witness exact only where gamma is a power of two)")
    print("CK PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
