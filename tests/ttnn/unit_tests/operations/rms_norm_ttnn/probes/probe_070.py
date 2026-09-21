# Is the per-row scale error coherent across the tokens of a head?  Rows are head-major (row = head*S + token).
# For each grouping (G groups of rows), report the fraction of error variance explained by the group (ICC-like R^2),
# and how sign-consistent the error is within a group.
import torch, ttnn, os, glob
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    SD=os.environ["SD"]; eps=1e-6
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    print(f"CO {'file':9s} {'rows':>4s} {'groups×tokens':>13s} | {'gen: R² by group':>16s} {'sign-agree':>10s} {'mean|e|':>7s} | {'nat: R² by group':>16s} {'sign-agree':>10s} {'mean|e|':>7s} | {'shuffled-rows R² gen/nat (null)':>30s}")
    for f in sorted(p for p in glob.glob(f"{SD}/phndump/phn_00?_?.pt")):
        x=torch.load(f)["x"].double().squeeze().clone(); R,W=x.shape; x[:,0]=1.0
        xb=x.to(torch.bfloat16); xd=xb.double(); s_ex=1/torch.sqrt(xd.pow(2).mean(-1)+eps); u=ulp(s_ex)
        X=ttnn.from_torch(xb.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        e={}
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            o=ttnn.to_torch(fn(X,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R,W)[:,0]; e[side]=(o-s_ex)/u
        def r2(v,G):
            g=v.reshape(G,-1); return (1 - ((g-g.mean(1,keepdim=True))**2).sum()/((v-v.mean())**2).sum()).item()
        def sign_agree(v,G):
            g=v.reshape(G,-1).sign(); return (g.mean(1).abs()).mean().item()   # 1 = all tokens of a head share the sign
        for G in (R//8, R//32) if R>=64 else (R//8,):
            if G<2: continue
            perm=torch.randperm(R, generator=torch.Generator().manual_seed(0))
            print(f"CO {os.path.basename(f)[:9]:9s} {R:4d} {G:>6d}×{R//G:<6d} | {r2(e['gen'],G):16.2f} {sign_agree(e['gen'],G):10.2f} {e['gen'].abs().mean().item():7.2f} | {r2(e['nat'],G):16.2f} {sign_agree(e['nat'],G):10.2f} {e['nat'].abs().mean().item():7.2f} | {r2(e['gen'][perm],G):.2f} / {r2(e['nat'][perm],G):.2f}", flush=True)
    print("CO PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
