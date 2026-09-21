# How does the FPU round a bf16*bf16 product into a 16-bit DEST?  (1) plain ttnn.multiply, elementwise and
# column-broadcast; (2) each norm op's pass B, using power-of-two "witness" columns (x=1,2,4,8) that reveal the
# row's scale exactly, then checking every other element against candidate roundings of x*s.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    torch.manual_seed(0); R,W,eps=64,256,1e-6
    def rne(x): return x.to(torch.bfloat16).double()
    def trunc(x):
        f=x.float().contiguous(); return (f.view(torch.int32)&0xFFFF0000).view(torch.float32).double()
    def bits(x): return x.float().contiguous().view(torch.int32)>>16
    def fb(b): return ((b.to(torch.int32)<<16).view(torch.float32)).double()
    def away(x):  # round half away from zero: trunc, then bump if the dropped part >= half
        t=trunc(x); u=fb(bits(t.abs())+1)*torch.sign(x); return torch.where((x-t).abs()>=(u-t).abs()/2, u, t)
    def up(x):    # toward +inf
        t=trunc(x); n=fb(bits(t.abs())+1)*torch.sign(x); return torch.where((x>t)&(x>0), n, torch.where((x<t)&(x<0)&False, t, t)) if False else torch.where(x>t, n if True else t, t)
    def report(tag, o, exact):
        c={"rne":rne(exact),"trunc":trunc(exact),"away":away(exact)}
        # generic: signed error of the device value relative to the exact product, in ulps of the result
        e=(o-exact)/torch.pow(2.0,torch.floor(torch.log2(exact.abs().clamp_min(1e-30)))-7)
        print(f"{tag}: match rne {(o==c['rne']).double().mean().item():.3f} trunc {(o==c['trunc']).double().mean().item():.3f} away {(o==c['away']).double().mean().item():.3f} | (dev-exact)/ulp: min {e.min().item():+.2f} max {e.max().item():+.2f} mean {e.mean().item():+.3f} frac|e|>0.5: {((e.abs()>0.5+1e-9)).double().mean().item():.3f}", flush=True)
    # (1) plain multiply
    a=torch.randn(1,1,R,W).to(torch.bfloat16); b=torch.randn(1,1,R,W).to(torch.bfloat16); col=(torch.rand(1,1,R,1)*0.1).to(torch.bfloat16)
    A=ttnn.from_torch(a,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev); B=ttnn.from_torch(b,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev); C=ttnn.from_torch(col,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi):
        for f32 in (False, True):
            ck=ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=f32, math_approx_mode=True, packer_l1_acc=False)
            o=ttnn.to_torch(ttnn.multiply(A,B,compute_kernel_config=ck)).double(); report(f"multiply elementwise {str(fid).split('.')[-1]:5s} fp32acc={f32}", o, a.double()*b.double())
            o=ttnn.to_torch(ttnn.multiply(A,C,compute_kernel_config=ck)).double(); report(f"multiply col-bcast   {str(fid).split('.')[-1]:5s} fp32acc={f32}", o, a.double()*col.double())
    # (2) the two norm ops' pass B
    x=torch.randn(R,W)*3; x[:,0:4]=torch.tensor([1.,2.,4.,8.]); x=x.to(torch.bfloat16); xd=x.double()
    X=ttnn.from_torch(x.reshape(1,1,R,W),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s_exact=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        for f32 in (False,True):
            ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=f32, math_approx_mode=True, packer_l1_acc=False)
            o=ttnn.to_torch(fn(X,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=ck)).double().reshape(R,W)
            wit=o[:,0:4]/xd[:,0:4]; agree=(wit.max(-1).values==wit.min(-1).values).all().item()
            s=wit[:,0:1]; isbf=torch.equal(rne(s),s)
            e_s=((s-s_exact)/torch.pow(2.0,torch.floor(torch.log2(s_exact))-7)).squeeze()
            print(f"{side} fp32acc={f32}: witnesses agree {agree}, scale is bf16-valued {isbf}; scale err ulps mean {e_s.mean().item():+.3f} min {e_s.min().item():+.2f} max {e_s.max().item():+.2f} frac<0 {(e_s<0).double().mean().item():.2f}")
            report(f"   {side} pass B vs x*s_witness", o[:,4:], xd[:,4:]*s)
            # same multiply through plain ttnn.multiply with the witness scale as a column
            Sc=ttnn.from_torch(s.to(torch.bfloat16).reshape(1,1,R,1) if isbf else s.float().reshape(1,1,R,1), ttnn.bfloat16 if isbf else ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            o2=ttnn.to_torch(ttnn.multiply(X,Sc,compute_kernel_config=ck)).double().reshape(R,W)
            print(f"   {side} pass B == ttnn.multiply(x, s_witness): {(o2[:,4:]==o[:,4:]).double().mean().item():.3f}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
