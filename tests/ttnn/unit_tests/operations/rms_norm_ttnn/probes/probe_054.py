# (a) ttnn.rsqrt on bf16: the SFPU rsqrt's approximation + store rounding, measured directly.
# (b) F3': rows of n copies of +-2^k and zeros -> S = n*4^k is exact in bf16 with no partial-sum rounding,
#     but m = S/W is generic -> isolates the two ops' FINALIZE chains at 16-bit DEST.
# (c) F2 (random powers of two) under fp32_dest_acc -> do the ops agree when nothing is narrowed?
# (d) pass B rounding: witness columns x=1,2,4,8 give the row scale exactly; check other elements.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    torch.manual_seed(0); R,W,eps=64,256,1e-6
    def rne(x): return x.to(torch.bfloat16).double()
    def trunc(x):
        f=x.float().contiguous(); return (f.view(torch.int32)&0xFFFF0000).view(torch.float32).double()
    def ulp(x): return torch.pow(2.0,torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    def stats(tag, dev_val, exact):
        e=((dev_val-exact)/ulp(exact)).flatten()
        print(f"{tag}: err vs exact (ulp) mean {e.mean().item():+.3f} min {e.min().item():+.2f} max {e.max().item():+.2f} frac<0 {(e<0).double().mean().item():.2f} | match rne {(dev_val==rne(exact)).double().mean().item():.3f} trunc {(dev_val==trunc(exact)).double().mean().item():.3f}", flush=True)
    # (a)
    m=(torch.rand(1,1,R,W)*4000+0.5).to(torch.bfloat16); M=ttnn.from_torch(m,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
    for fam in (True,False):
        o=ttnn.to_torch(ttnn.rsqrt(M,fast_and_approximate_mode=fam)).double(); stats(f"(a) ttnn.rsqrt bf16 approx={fam}", o, 1/torch.sqrt(m.double()))
    # (b)
    ks=torch.arange(R)%5-2; ns=1+4*torch.arange(R)
    x=torch.zeros(R,W,dtype=torch.float64)
    for r in range(R): x[r,:ns[r]]=torch.pow(2.0,float(ks[r]))*torch.where(torch.rand(ns[r])<0.5,-1.0,1.0)
    X=ttnn.from_torch(x.to(torch.bfloat16).reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s_ex=1/torch.sqrt(x.pow(2).sum(-1)/W+eps); sc={}
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        for approx in (True,False):
            ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=False,math_approx_mode=approx,packer_l1_acc=False)
            o=ttnn.to_torch(fn(X,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=ck)).double().reshape(R,W)
            s=(o[:,0]/x[:,0]); ok=all(torch.equal((o[r,:ns[r]]/x[r,:ns[r]]), s[r].repeat(ns[r])) for r in range(R))
            sc[(side,approx)]=s; stats(f"(b) F3' exact-S {side} approx={approx} (row-consistent {ok})", s, s_ex)
    for approx in (True,False):
        d=((sc[("gen",approx)]-sc[("nat",approx)])/ulp(s_ex)); print(f"(b) F3' gen-nat approx={approx}: mean {d.mean().item():+.3f} ulp, rows equal {(d==0).sum().item()}/{R}, hist {dict((int(k),int((d.round()==k).sum())) for k in d.round().unique())}")
    # (c)
    k2=torch.randint(-3,4,(R,W)).double(); x2=torch.pow(2.0,k2)*torch.where(torch.rand(R,W)<0.5,-1.0,1.0).double()
    X2=ttnn.from_torch(x2.to(torch.bfloat16).reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s2=1/torch.sqrt(x2.pow(2).sum(-1)/W+eps); r2={}
    for f32 in (True,False):
        ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=f32,math_approx_mode=True,packer_l1_acc=False)
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            o=ttnn.to_torch(fn(X2,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=ck)).double().reshape(R,W)
            r2[side]=(o/x2)[:,0]; stats(f"(c) F2 generic-S fp32acc={f32} {side}", r2[side], s2)
        d=(r2["gen"]-r2["nat"])/ulp(s2); print(f"(c) F2 fp32acc={f32} gen-nat: mean {d.mean().item():+.3f} ulp, rows equal {(d==0).sum().item()}/{R}")
    # (d)
    x3=torch.randn(R,W)*3; x3[:,0:4]=torch.tensor([1.,2.,4.,8.]); x3=x3.to(torch.bfloat16); xd=x3.double()
    X3=ttnn.from_torch(x3.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        o=ttnn.to_torch(fn(X3,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R,W)
        wit=o[:,0:4]/xd[:,0:4]; agree=bool((wit.max(-1).values==wit.min(-1).values).all()); s=wit[:,0:1]
        print(f"(d) {side}: witnesses agree {agree}, scale bf16-valued {torch.equal(rne(s),s)}")
        stats(f"(d) {side} pass B output vs exact x*s", o[:,4:], xd[:,4:]*s)
        Sc=ttnn.from_torch(s.to(torch.bfloat16).reshape(1,1,R,1),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
        o2=ttnn.to_torch(ttnn.multiply(X3,Sc)).double().reshape(R,W)
        stats(f"(d) ttnn.multiply(x, s) vs exact x*s      ", o2[:,4:], xd[:,4:]*s)
        print(f"(d) {side} pass B == ttnn.multiply: {(o2[:,4:]==o[:,4:]).double().mean().item():.3f}", flush=True)
    a=torch.randn(1,1,R,W).to(torch.bfloat16); b=torch.randn(1,1,R,W).to(torch.bfloat16)
    o=ttnn.to_torch(ttnn.multiply(ttnn.from_torch(a,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev),ttnn.from_torch(b,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev))).double()
    stats("(d) ttnn.multiply elementwise bf16 default", o, a.double()*b.double())
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
