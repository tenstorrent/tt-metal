import torch, ttnn, glob, os
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x):
        e=torch.floor(torch.log2(x.abs().clamp_min(1e-30))); return torch.pow(2.0,e-7)
    cfgs = {"default": None,
            "fp32acc": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True, packer_l1_acc=False),
            "noapprox": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False, packer_l1_acc=False)}
    cases=[]
    for f in sorted([p for p in glob.glob(f"{SD}/phndump/phn_00[012]_*.pt") if not p.endswith("_out.pt")]):
        d=torch.load(f); cases.append((os.path.basename(f)[:9], d["x"], d["w"], d["eps"]))
    torch.manual_seed(1)
    cases.append(("synth64x3840", torch.randn(1,1,64,3840,dtype=torch.bfloat16)*4, torch.randn(1,1,120,32,dtype=torch.bfloat16), 1e-6))
    cases.append(("synth64x256", torch.randn(1,1,64,256,dtype=torch.bfloat16)*4, torch.randn(1,1,8,32,dtype=torch.bfloat16), 1e-6))
    print(f"{'case':13s} {'cfg':9s} | {'gen bias(ulp)':>13s} {'gen frac<0':>10s} {'gen |err|':>9s} | {'nat bias(ulp)':>13s} {'nat frac<0':>10s} {'nat |err|':>9s}")
    for name,x_t,w_t,eps in cases:
        x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xd=x_t.double().squeeze(); gam=w_t.double().reshape(1,-1) if w_t is not None else torch.ones(1,xd.shape[-1],dtype=torch.float64)
        base=xd*gam; true=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
        for cn,ckc in cfgs.items():
            row=f"{name:13s} {cn:9s} |"
            for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
                kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
                if w is not None: kw["weight"]=w
                if ckc is not None: kw["compute_kernel_config"]=ckc
                o=ttnn.to_torch(fn(x,**kw)).double().squeeze()
                sc=(o*base).sum(-1,keepdim=True)/(base*base).sum(-1,keepdim=True)
                e=((sc-true)/ulp(true)).squeeze()
                row+=f" {e.mean().item():+13.3f} {(e<0).double().mean().item():10.2f} {e.abs().mean().item():9.3f} |"
            print(row, flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
