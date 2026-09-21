# Power-of-two inputs make pass B exact, so the output reveals each row's 1/rms bit for bit.
# F3: every row is +-2^k with one k -> S exact in bf16, no partial-sum rounding anywhere -> isolates the finalize.
# F2: random powers of two -> S has many mantissa bits -> exercises the reduce's rounding.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    torch.manual_seed(0)
    W, R, eps = 256, 64, 1e-6
    def rne(x): return x.to(torch.bfloat16).double()
    def trunc(x):
        f=x.float().contiguous(); return (f.view(torch.int32)&0xFFFF0000).view(torch.float32).double()
    def ulp(x):
        e=torch.floor(torch.log2(x.abs().clamp_min(1e-30))); return torch.pow(2.0,e-7)
    ks3 = torch.arange(R) % 12 - 6                      # F3: k in [-6,5], same k across a row
    x3 = torch.pow(2.0, ks3.double())[:,None].repeat(1,W) * torch.where(torch.rand(R,W)<0.5,-1.0,1.0).double()
    k2 = torch.randint(-3,4,(R,W)).double()             # F2
    x2 = torch.pow(2.0,k2) * torch.where(torch.rand(R,W)<0.5,-1.0,1.0).double()
    for fam,xd in (("F3 uniform-k",x3),("F2 random-k",x2)):
        assert torch.equal(rne(xd), xd)
        S_exact = xd.pow(2).sum(-1); s_exact = 1/torch.sqrt(S_exact/W+eps)
        x = ttnn.from_torch(xd.to(torch.bfloat16).reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        obs={}
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            o = ttnn.to_torch(fn(x, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R,W)
            ratio = o/xd
            spread = (ratio.max(-1).values - ratio.min(-1).values)
            obs[side]=ratio[:,0]
            e = (ratio[:,0]-s_exact)/ulp(s_exact)
            print(f"{fam} {side}: rows where the whole row agrees on one scale: {(spread==0).sum().item()}/{R} | scale err vs exact (ulp): mean {e.mean().item():+.3f} min {e.min().item():+.2f} max {e.max().item():+.2f} | frac<0 {(e<0).double().mean().item():.2f}")
        d=(obs["gen"]-obs["nat"])/ulp(s_exact)
        print(f"{fam} gen-nat scale (ulp): mean {d.mean().item():+.3f}; rows equal {(d==0).sum().item()}/{R}; hist " + str({int(k):int((d.round()==k).sum()) for k in d.round().unique()}))
        if fam.startswith("F2"):
            # emulate the two reduce pipelines under RNE and TRUNC at every 16-bit DEST write; exact rsqrt after.
            xt = xd.reshape(R,8,32)
            def emu_gen(Rd):
                acc = Rd(xt[:,0]**2)
                for t in range(1,8): acc = Rd(acc + xt[:,t]**2)
                return Rd(acc.sum(-1))
            def emu_nat(Rd):
                q = Rd(xt**2); acc = Rd(q[:,0].sum(-1)/W)
                for t in range(1,8): acc = Rd(acc + q[:,t].sum(-1)/W)
                return acc*W
            for nm,Rd in (("RNE",rne),("TRUNC",trunc)):
                Sg, Sn = emu_gen(Rd), emu_nat(Rd)
                dS = ((Sg-Sn)/ulp(S_exact))
                sg, sn = 1/torch.sqrt(Sg/W+eps), 1/torch.sqrt(Sn/W+eps)
                pred = (sg-sn)/ulp(s_exact)
                print(f"   emulated {nm:5s}: S_gen-S_exact mean {((Sg-S_exact)/ulp(S_exact)).mean().item():+.3f} ulp, S_nat-S_exact {((Sn-S_exact)/ulp(S_exact)).mean().item():+.3f} | predicted gen-nat scale mean {pred.mean().item():+.3f} ulp, corr with observed {torch.corrcoef(torch.stack([pred,d]))[0,1].item():+.2f}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
