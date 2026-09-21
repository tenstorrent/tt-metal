# (b) F3': S exact in bf16, generic m -> isolates the finalize.  (c) F2 under fp32 acc.  (d) pass B rounding via
# witness columns.  (e) fit candidate 16-bit narrowing rules (rne / trunc / ties-away) to F2's observed scales
# for both pipelines: gen = fold x^2 across tiles then one column sum; nat = per-tile column sums accumulated.
import torch, ttnn, os
OUT=os.environ["OUT"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    torch.manual_seed(0); R,W,eps=64,256,1e-6
    def rne(x): return x.to(torch.bfloat16).double()
    def trunc(x):
        f=x.float().contiguous(); return (f.view(torch.int32)&0xFFFF0000).view(torch.float32).double()
    def bits(x): return x.float().contiguous().view(torch.int32)>>16
    def fb(b): return ((b.to(torch.int32)<<16).view(torch.float32)).double()
    def away(x):
        t=trunc(x); n=fb(bits(t.abs())+1)*torch.sign(x); n=torch.where(t==0,t,n)
        return torch.where((x-t).abs()*2>=(n-t).abs(), n, t)
    def ulp(x): return torch.pow(2.0,torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    def stats(tag, dev_val, exact):
        e=((dev_val-exact)/ulp(exact)).flatten()
        print(f"{tag}: err vs exact (ulp) mean {e.mean().item():+.3f} min {e.min().item():+.2f} max {e.max().item():+.2f} frac<0 {(e<0).double().mean().item():.2f} | match rne {(dev_val==rne(exact)).double().mean().item():.3f} trunc {(dev_val==trunc(exact)).double().mean().item():.3f} away {(dev_val==away(exact)).double().mean().item():.3f}", flush=True)
    def run(fn,X,ck=None):
        kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
        if ck is not None: kw["compute_kernel_config"]=ck
        return ttnn.to_torch(fn(X,**kw)).double().reshape(R,W)
    CK=lambda f32,approx=True: ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=f32,math_approx_mode=approx,packer_l1_acc=False)
    save={}
    # (b)
    ks=torch.arange(R)%5-2; ns=1+4*torch.arange(R)
    x=torch.zeros(R,W,dtype=torch.float64)
    for r in range(R): x[r,:ns[r]]=(2.0**float(ks[r]))*torch.where(torch.rand(int(ns[r]))<0.5,-1.0,1.0).double()
    X=ttnn.from_torch(x.to(torch.bfloat16).reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s_ex=1/torch.sqrt(x.pow(2).sum(-1)/W+eps); sc={}
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        for approx in (True,False):
            o=run(fn,X,CK(False,approx)); s=o[:,0]/x[:,0]
            ok=all(torch.equal(o[r,:ns[r]]/x[r,:ns[r]], s[r].repeat(int(ns[r]))) for r in range(R))
            sc[(side,approx)]=s; stats(f"(b) F3' exact-S {side} approx={approx} row-consistent={ok}", s, s_ex)
    for approx in (True,False):
        d=(sc[("gen",approx)]-sc[("nat",approx)])/ulp(s_ex); print(f"(b) F3' gen-nat approx={approx}: mean {d.mean().item():+.3f} ulp, rows equal {(d==0).sum().item()}/{R}")
    # (c)+(e)
    k2=torch.randint(-3,4,(R,W)).double(); x2=torch.pow(2.0,k2)*torch.where(torch.rand(R,W)<0.5,-1.0,1.0).double()
    X2=ttnn.from_torch(x2.to(torch.bfloat16).reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    S_ex=x2.pow(2).sum(-1); s2=1/torch.sqrt(S_ex/W+eps); obs={}
    for f32 in (True,False):
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            o=run(fn,X2,CK(f32)); obs[(side,f32)]=(o/x2)[:,0]; stats(f"(c) F2 fp32acc={f32} {side}", obs[(side,f32)], s2)
        d=(obs[("gen",f32)]-obs[("nat",f32)])/ulp(s2); print(f"(c) F2 fp32acc={f32} gen-nat: mean {d.mean().item():+.3f} ulp, rows equal {(d==0).sum().item()}/{R}")
    save.update(x2=x2, obs_gen=obs[("gen",False)], obs_nat=obs[("nat",False)], obs_gen32=obs[("gen",True)], obs_nat32=obs[("nat",True)])
    xt=x2.reshape(R,8,32)
    for nm,N in (("rne",rne),("trunc",trunc),("away",away)):
        acc=N(xt[:,0]**2)
        for t in range(1,8): acc=N(acc+xt[:,t]**2)
        Sg=N(acc.sum(-1)); sg=rne(1/torch.sqrt(Sg/W+eps))
        q=N(xt**2); accn=N(q[:,0].sum(-1)/W)
        for t in range(1,8): accn=N(accn+q[:,t].sum(-1)/W)
        sn=rne(1/torch.sqrt(N(accn+eps)))
        print(f"(e) narrowing={nm:5s}: gen pipeline predicts observed gen scale on {(sg==obs[('gen',False)]).sum().item()}/{R} rows (S bias {((Sg-S_ex)/ulp(S_ex)).mean().item():+.2f} ulp) | nat pipeline predicts nat on {(sn==obs[('nat',False)]).sum().item()}/{R} (S bias {((accn*W-S_ex)/ulp(S_ex)).mean().item():+.2f})", flush=True)
    # (d)
    x3=torch.randn(R,W)*3; x3[:,0:4]=torch.tensor([1.,2.,4.,8.]); x3=x3.to(torch.bfloat16); xd=x3.double()
    X3=ttnn.from_torch(x3.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        o=run(fn,X3); wit=o[:,0:4]/xd[:,0:4]; agree=bool((wit.max(-1).values==wit.min(-1).values).all()); s=wit[:,0:1]
        print(f"(d) {side}: witnesses agree {agree}, scale bf16-valued {torch.equal(rne(s),s)}")
        stats(f"(d) {side} pass B output vs exact x*s", o[:,4:], xd[:,4:]*s)
        Sc=ttnn.from_torch(s.to(torch.bfloat16).reshape(1,1,R,1),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
        o2=ttnn.to_torch(ttnn.multiply(X3,Sc)).double().reshape(R,W)
        stats(f"(d) ttnn.multiply(x, s_col) vs exact      ", o2[:,4:], xd[:,4:]*s)
        print(f"(d) {side} pass B == ttnn.multiply: {(o2[:,4:]==o[:,4:]).double().mean().item():.3f}", flush=True)
        save[f"passB_{side}"]=o; save["x3"]=xd
    a=torch.randn(1,1,R,W).to(torch.bfloat16); b=torch.randn(1,1,R,W).to(torch.bfloat16)
    o=ttnn.to_torch(ttnn.multiply(ttnn.from_torch(a,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev),ttnn.from_torch(b,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev))).double()
    stats("(d) ttnn.multiply elementwise bf16 default", o, a.double()*b.double())
    torch.save(save, f"{OUT}/probe_rounding.pt")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
