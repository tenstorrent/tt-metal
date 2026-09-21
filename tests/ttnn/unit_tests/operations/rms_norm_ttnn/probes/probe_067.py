# Real per-head rows under different ARRANGEMENTS of the same values (exact scale unchanged per row):
# as-is, tile order reversed, tiles rotated by 4, tile order shuffled, positions shuffled within each tile.
import torch, ttnn, os, glob, math
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    SD=os.environ["SD"]; eps=1e-6
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    g=torch.Generator().manual_seed(0)
    def arrangements(x):
        R,W=x.shape; t=x.reshape(R,8,32)
        out={"as-is":x, "tiles reversed":t.flip(1).reshape(R,W), "tiles rotated 4":t.roll(4,1).reshape(R,W)}
        perm=torch.randperm(8,generator=g); out["tiles shuffled"]=t[:,perm].reshape(R,W)
        p=torch.rand(R,8,32,generator=g).argsort(-1); out["positions shuffled within tile"]=torch.gather(t,2,p).reshape(R,W)
        p2=torch.rand(R,W,generator=g).argsort(-1); out["fully permuted"]=torch.gather(x,1,p2)
        return out
    print(f"AR {'file':>9s} {'arrangement':32s} | {'bias gen':>9s} {'bias nat':>9s} | {'mean|e| gen':>11s} {'nat':>6s} | {'rows gen +1':>11s} {'gen closer':>10s} {'nat closer':>10s}")
    for f in sorted(p for p in glob.glob(f"{SD}/phndump/phn_00?_?.pt")):
        x0=torch.load(f)["x"].double().squeeze(); R,W=x0.shape
        for name,x in arrangements(x0).items():
            x=x.clone(); x[:,0]=1.0; xb=x.to(torch.bfloat16); xd=xb.double()
            s_ex=1/torch.sqrt(xd.pow(2).mean(-1)+eps); u=ulp(s_ex)
            X=ttnn.from_torch(xb.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            e={}
            for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
                o=ttnn.to_torch(fn(X,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R,W)[:,0]
                e[side]=(o-s_ex)/u
            d=e["gen"].abs()-e["nat"].abs()
            print(f"AR {os.path.basename(f)[:9]:>9s} {name:32s} | {e['gen'].mean().item():+9.3f} {e['nat'].mean().item():+9.3f} | {e['gen'].abs().mean().item():11.3f} {e['nat'].abs().mean().item():6.3f} | {(e['gen'].round()>=1).double().mean().item():11.2f} {(d<0).double().mean().item():10.2f} {(d>0).double().mean().item():10.2f}", flush=True)
    print("AR PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
