# Row-scale bias on the real per-head inputs with the square fold actually disabled (both knobs), and with the
# reduce forced onto the FPU matmul path.  The earlier attempt left the fold on via the grouped fallback.
import torch, ttnn, glob, os
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x): return torch.pow(2.0,torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    cases=[]
    for f in sorted(p for p in glob.glob(f"{SD}/phndump/phn_00[0-5]_*.pt") if not p.endswith("_out.pt")):
        d=torch.load(f); cases.append((os.path.basename(f)[:9], d["x"], d["w"], d["eps"]))
    D0=(desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT, desc.SQ_FOLD_GROUP)
    print("defaults (REDUCE_ACC_VIA_ADD_MIN_WT, DEST_ACC_SQUARE_MAX_WT, SQ_FOLD_GROUP) =", D0)
    variants=[("gen default (fold+ReduceTile)", ttnn.rms_norm, D0),
              ("gen no-fold, AccViaAdd",        ttnn.rms_norm, (D0[0], 0, 1)),
              ("gen no-fold, ReduceTile",       ttnn.rms_norm, (10**9, 0, 1)),
              ("gen fold group 2",              ttnn.rms_norm, (D0[0], 0, 2)),
              ("nat default",                   ttnn._native_rms_norm, D0)]
    print(f"{'variant':30s} | " + " | ".join(f"{c[0]:>10s}" for c in cases) + "   (row-scale bias, bf16 ulps: mean / frac<0)")
    outs={}
    for vname,fn,(a,b,g) in variants:
        desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT, desc.SQ_FOLD_GROUP = a,b,g
        row=f"{vname:30s} |"
        for name,x_t,w_t,eps in cases:
            x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
            if w is not None: kw["weight"]=w
            o=ttnn.to_torch(fn(x,**kw)).double().squeeze(); outs[(vname,name)]=o
            xd=x_t.double().squeeze(); gam=w_t.double().reshape(1,-1) if w_t is not None else torch.ones(1,xd.shape[-1],dtype=torch.float64)
            base=xd*gam; true=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
            sc=(o*base).sum(-1,keepdim=True)/(base*base).sum(-1,keepdim=True); e=((sc-true)/ulp(true)).squeeze()
            row+=f" {e.mean().item():+5.2f}/{(e<0).double().mean().item():.2f} |"
        print(row, flush=True)
    desc.REDUCE_ACC_VIA_ADD_MIN_WT, desc.DEST_ACC_SQUARE_MAX_WT, desc.SQ_FOLD_GROUP = D0
    for v in ("gen no-fold, AccViaAdd","gen no-fold, ReduceTile"):
        print(f"{v}: elements identical to native: " + " ".join(f"{(outs[(v,c[0])]==outs[('nat default',c[0])]).double().mean().item():.3f}" for c in cases))
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
