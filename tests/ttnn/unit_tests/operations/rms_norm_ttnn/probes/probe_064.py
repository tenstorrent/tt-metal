import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    R, W, eps = 32, 256, 1e-6
    cfgs = [(16,1),(16,0.5),(16,1.5),(64,1),(64,3),(128,1),(128,5)]; tiles=[0,3,7]
    x = torch.ones(R, W, dtype=torch.float64); meta=[]; r=0
    for xo,xs in cfgs:
        for t in tiles:
            x[r,:]=xs; x[r,32*t+12]=xo; meta.append((xo,xs,t)); r+=1
    assert torch.equal(x.to(torch.bfloat16).double(), x)
    def rne(v): return torch.tensor(float(v)).to(torch.bfloat16).double().item()
    def fold_pred(xo,xs,t):
        acc=0.0
        for k in range(8): acc=rne(acc+(xo*xo if k==t else xs*xs))
        a=0.0
        for k in range(8): a=rne(a+xs*xs)
        return acc+31*a
    def implied_S(s):
        lo=hi=None
        for S in range(1, 30000):
            if rne(1/((S/W+eps)**0.5))==s: lo=S if lo is None else lo; hi=S
        return f"{lo}..{hi}" if lo is not None else "none"
    X = ttnn.from_torch(x.to(torch.bfloat16).reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res={}
    for name, fn, order in (("gen", ttnn.rms_norm, "shipped"), ("gen-natorder", ttnn.rms_norm, "native"), ("native", ttnn._native_rms_norm, "shipped")):
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
        o = ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, W)
        res[name]=[o[i,32*meta[i][2]+12].item()/meta[i][0] for i in range(len(meta))]
    S_exact = x.pow(2).sum(-1)
    print(f"ROW {'x_out':>5s} {'x_small':>7s} {'tile':>4s} | {'exact S':>8s} | {'gen shipped':>13s} {'pred: drop':>10s} | {'gen native-order':>16s} | {'native':>13s}")
    for i,(xo,xs,t) in enumerate(meta):
        print(f"ROW {xo:5g} {xs:7g} {t:4d} | {S_exact[i].item():8.2f} | {implied_S(res['gen'][i]):>13s} {fold_pred(xo,xs,t):10.0f} | {implied_S(res['gen-natorder'][i]):>16s} | {implied_S(res['native'][i]):>13s}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
