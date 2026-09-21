# Verify the RMS_REDUCE_ORDER ablation switch on the six captured per-head inputs.
import torch, ttnn, glob, os
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x): return torch.pow(2.0,torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    print("RMS_REDUCE_ORDER =", repr(os.environ.get("RMS_REDUCE_ORDER","")), "->", desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT)
    row_b=f"{'gen ('+desc.REDUCE_ORDER+')':22s} |"; row_i=f"{'identical to native':22s} |"; hdr=f"{'':22s} |"
    for f in sorted(p for p in glob.glob(f"{SD}/phndump/phn_00[0-5]_*.pt") if not p.endswith("_out.pt")):
        d=torch.load(f); name=os.path.basename(f)[:9]; x_t,w_t,eps=d["x"],d["w"],d["eps"]
        x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kw={"epsilon":eps,"memory_config":ttnn.DRAM_MEMORY_CONFIG}
        if w is not None: kw["weight"]=w
        og=ttnn.to_torch(ttnn.rms_norm(x,**kw)).double().squeeze(); on=ttnn.to_torch(ttnn._native_rms_norm(x,**kw)).double().squeeze()
        xd=x_t.double().squeeze(); gam=w_t.double().reshape(1,-1) if w_t is not None else torch.ones(1,xd.shape[-1],dtype=torch.float64)
        base=xd*gam; true=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
        e=(((og*base).sum(-1,keepdim=True)/(base*base).sum(-1,keepdim=True)-true)/ulp(true)).squeeze()
        hdr+=f" {name:>10s} |"; row_b+=f" {e.mean().item():+5.2f}/{(e<0).double().mean().item():.2f} |"; row_i+=f" {(og==on).double().mean().item():10.3f} |"
    print(hdr); print(row_b); print(row_i); print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
