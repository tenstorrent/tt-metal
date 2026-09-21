# Larger outliers: the small squares fall well below half a 16-bit step of the outlier's running sum.
# Nearest-type landing must drop them; round-up landing inflates each by a full step.  Scale is read
# off the outlier position (x_out is a power of two, so o = x_out*s exactly).
import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    R, W, eps = 32, 256, 1e-6
    cfgs = [(64,1),(64,3),(16,0.5),(16,1.5),(128,1),(128,5)]; tiles=[0,3,7]
    x = torch.ones(R, W, dtype=torch.float64); meta=[]
    r=0
    for xo,xs in cfgs:
        for t in tiles:
            x[r,:]=xs; x[r,32*t+12]=xo; meta.append((xo,xs,t)); r+=1
    assert torch.equal(x.to(torch.bfloat16).double(), x)
    def rne(v): return torch.tensor(float(v)).to(torch.bfloat16).double().item()
    def ceil16(v):  # round toward +inf to a bf16 grid point
        f=torch.tensor(float(v),dtype=torch.float32); i=(f.view(torch.int32)&0xFFFF0000).view(torch.float32).item()
        return i if i==v else (torch.tensor(i,dtype=torch.float32).view(torch.int32)+0x10000).view(torch.float32).item()
    def fold_pred(xo,xs,t,land):   # per-position fold, then exact sum over the 32 positions
        acc=0.0
        for k in range(8): acc=land(acc+(xo*xo if k==t else xs*xs))
        return acc + 31*land_seq(xs,land)
    def land_seq(xs,land):
        a=0.0
        for k in range(8): a=land(a+xs*xs)
        return a
    def implied_S(s):
        lo=hi=None
        for S in range(1, 30000):
            if rne(1/((S/W+eps)**0.5))==s: lo=S if lo is None else lo; hi=S
        return f"{lo}..{hi}" if lo is not None else "none"
    X = ttnn.from_torch(x.to(torch.bfloat16).reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res={}
    for name, fn, order in (("gen shipped", ttnn.rms_norm, "shipped"), ("native", ttnn._native_rms_norm, "shipped")):
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
        o = ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, W)
        res[name]=torch.tensor([o[i,32*meta[i][2]+12].item()/meta[i][0] for i in range(len(meta))])
    S_exact = x.pow(2).sum(-1)
    print(f"{'x_out':>5s} {'x_small':>7s} {'tile':>4s} | {'exact S':>8s} | {'gen believed S':>14s} | {'pred nearest':>12s} {'pred round-up':>13s} | {'native believed S':>17s}")
    for i,(xo,xs,t) in enumerate(meta):
        print(f"{xo:5g} {xs:7g} {t:4d} | {S_exact[i].item():8.2f} | {implied_S(res['gen shipped'][i].item()):>14s} | {fold_pred(xo,xs,t,rne):12.0f} {fold_pred(xo,xs,t,ceil16):13.0f} | {implied_S(res['native'][i].item()):>17s}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
