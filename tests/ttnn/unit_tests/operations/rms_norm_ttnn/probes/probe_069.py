# Same k rows, scale read exactly through a witness (x=1 at pos 0), with gamma absent, gamma present,
# and gamma present but equal to 1.0 everywhere (so the witness stays exact).
import torch, ttnn, os
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    SD=os.environ["SD"]; eps=1e-6
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    d=torch.load(f"{SD}/phndump/phn_001_k.pt"); x=d["x"].double().squeeze().clone(); w_t=d["w"]; R,W=x.shape
    x[:,0]=1.0; xb=x.to(torch.bfloat16); xd=xb.double(); s_ex=1/torch.sqrt(xd.pow(2).mean(-1)+eps); u=ulp(s_ex)
    X=ttnn.from_torch(xb.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    Wreal=ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    Wones=ttnn.from_torch(torch.ones_like(w_t),ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    Wpow2=ttnn.from_torch(torch.full_like(w_t,0.125),ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gam0=w_t.double().reshape(-1)[0].item()
    print(f"GM {'config':28s} | {'gen bias':>9s} {'gen mean|e|':>11s} | {'nat bias':>9s} {'nat mean|e|':>11s}   (witness exact except 'real gamma', where gamma[0]={gam0:.4f} adds one rounding)")
    for label,Wt,g0 in (("no gamma",None,1.0),("gamma = 1.0 everywhere",Wones,1.0),("gamma = 0.125 everywhere",Wpow2,0.125),("real gamma",Wreal,gam0)):
        row=f"GM {label:28s} |"
        for fn in (ttnn.rms_norm, ttnn._native_rms_norm):
            kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
            if Wt is not None: kw["weight"]=Wt
            print(f"GM -- {label} {'gen' if fn is ttnn.rms_norm else 'nat'}", flush=True)
            o=ttnn.to_torch(fn(X,**kw)).double().reshape(R,W); e=(o[:,0]/g0-s_ex)/u
            row+=f" {e.mean().item():+9.3f} {e.abs().mean().item():11.3f} |"
        print(row, flush=True)
    print("GM PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
