# Kernel-level bisection of the generated op's row-scale bias on the real Gemma per-head inputs,
# plus a determinism check. Variants flip the descriptor's precision knobs; nothing in the tree changes.
import torch, ttnn, glob, os
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x):
        e=torch.floor(torch.log2(x.abs().clamp_min(1e-30))); return torch.pow(2.0,e-7)
    fp32acc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True, packer_l1_acc=False)
    cases=[]
    for f in sorted(p for p in glob.glob(f"{SD}/phndump/phn_00[0-5]_*.pt") if not p.endswith("_out.pt")):
        d=torch.load(f); cases.append((os.path.basename(f)[:9], d["x"], d["w"], d["eps"]))
    D0=(desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT)
    variants=[("gen default",       ttnn.rms_norm,          None,    D0),
              ("gen ReduceTile",    ttnn.rms_norm,          None,    (10**9, D0[1])),
              ("gen no-fold",       ttnn.rms_norm,          None,    (D0[0], 0)),
              ("gen RT+no-fold",    ttnn.rms_norm,          None,    (10**9, 0)),
              ("gen fp32acc",       ttnn.rms_norm,          fp32acc, D0),
              ("nat default",       ttnn._native_rms_norm,  None,    D0),
              ("nat fp32acc",       ttnn._native_rms_norm,  fp32acc, D0)]
    print(f"{'variant':16s} | " + " | ".join(f"{c[0]:>9s}" for c in cases) + "   (row-scale bias in bf16 ulps, mean over rows; frac rows<0)")
    for vname,fn,ckc,(a,b) in variants:
        desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT = a,b
        dev.disable_and_clear_program_cache()
        row=f"{vname:16s} |"
        for name,x_t,w_t,eps in cases:
            x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
            if w is not None: kw["weight"]=w
            if ckc is not None: kw["compute_kernel_config"]=ckc
            o=ttnn.to_torch(fn(x,**kw)).double().squeeze()
            xd=x_t.double().squeeze(); gam=w_t.double().reshape(1,-1) if w_t is not None else torch.ones(1,xd.shape[-1],dtype=torch.float64)
            base=xd*gam; true=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
            sc=(o*base).sum(-1,keepdim=True)/(base*base).sum(-1,keepdim=True)
            e=((sc-true)/ulp(true)).squeeze()
            row+=f" {e.mean().item():+5.2f}/{(e<0).double().mean().item():.2f} |"
        print(row, flush=True)
    desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT = D0
    # determinism: same input, 12 dispatches each, bitwise compare (fresh input tensor each time too)
    dev.disable_and_clear_program_cache()
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        for name,x_t,w_t,eps in cases[:3]:
            outs=[]
            for i in range(12):
                x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
                w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
                kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
                if w is not None: kw["weight"]=w
                outs.append(ttnn.to_torch(fn(x,**kw)))
            same=all(torch.equal(outs[0],o) for o in outs[1:])
            print(f"determinism {side} {name}: 12 dispatches bitwise identical = {same}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
