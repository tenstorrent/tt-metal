# Width sweep on heavy-tailed rows: shipped generated order, native's order via RMS_REDUCE_ORDER, and native.
import torch, ttnn, os
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    torch.manual_seed(0); R, eps = 64, 1e-6
    def ulp(x): return torch.pow(2.0,torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    def path(wt):  # what the descriptor would choose for a resident row of wt tiles
        xs = desc._x_squared_wt(wt, 0)
        acc = desc.REDUCE_BULK==1 and wt>=desc.REDUCE_ACC_VIA_ADD_MIN_WT and wt>=desc.REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT and not (xs<desc.REDUCE_ACC_VIA_ADD_MIN_CALL_WT) and not (desc.REDUCE_ORDER=="native" and desc._reduce_order_applies(wt))
        fold = "no fold" if xs==wt else (f"fold {wt//xs} tiles/group -> {xs} tiles" if xs>1 else f"fold all {wt} tiles -> 1")
        return f"{fold}; then {'pairwise tile add + SFPU collapse' if acc else 'FPU per-tile reduce'}"
    for W in (256, 512, 1024, 2048, 3840, 11008):
        x = torch.randn(R, W); m = torch.rand(R, W) < 0.02; x[m] *= 16      # heavy tail: 2 % outliers x16
        x = x.to(torch.bfloat16); xd = x.double()
        true = 1/torch.sqrt(xd.pow(2).mean(-1, keepdim=True)+eps)
        X = ttnn.from_torch(x.reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"W={W} ({W//32} tiles)  |  {'variant':26s} | scale bias mean | std  | frac<0 | rel-RMS vs fp64")
        for vname, fn, order in (("gen shipped", ttnn.rms_norm, "shipped"), ("gen RMS_REDUCE_ORDER=native", ttnn.rms_norm, "native"), ("nat", ttnn._native_rms_norm, "shipped")):
            desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
            o = ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, W)
            sc = (o*xd).sum(-1,keepdim=True)/(xd*xd).sum(-1,keepdim=True); e = ((sc-true)/ulp(true)).squeeze()
            rel = ((o-xd*true).norm()/(xd*true).norm()).item()
            tag = "gen" if vname.startswith("gen") else "nat"
            print(f"{tag} {'':22s} {vname:26s} | {e.mean().item():+7.2f}         | {e.std().item():.2f} | {(e<0).double().mean().item():.2f}   | {rel:.2e}")
            if tag=="gen": print(f"  path: {path(W//32)}")
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = "shipped", 0
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
