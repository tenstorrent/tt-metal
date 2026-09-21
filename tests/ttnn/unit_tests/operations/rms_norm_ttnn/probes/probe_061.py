# The textbook row on hardware: one square of 256 (x=16) and 255 squares of 1 (x=1), W=256 (8 tiles).
# Ones positions output s*1 = s exactly, so each row's scale is read off the output; the sum the
# device "believed" is every integer S whose bf16 rsqrt matches that scale.
import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    R, W, eps = 32, 256, 1e-6
    x = torch.ones(R, W, dtype=torch.float64); desc_rows = []
    for r in range(8):  x[r, 32*r+12] = 16;                    desc_rows.append(f"outlier in tile {r}, pos 12")
    for r in range(8):  x[8+r, 32*r+31] = 16;                  desc_rows.append(f"outlier in tile {r}, pos 31")
    for r in range(8):  x[16+r, 12] = 16; x[16+r, 32*r+12+ (0 if r else 0)] = 16 if r else 16; desc_rows.append(f"outliers in tiles 0 and {r}, pos 12" if r else "one outlier, tile 0, pos 12 (dup)")
    for r in range(8):  desc_rows.append("no outlier (all ones)")
    def rne(v): return torch.tensor(v).to(torch.bfloat16).double().item()
    def implied_S(s):  # integers S whose bf16 scale equals the observed one
        ok=[S for S in range(200, 700) if rne(1/((S/W+eps)**0.5))==s]
        return f"{ok[0]}..{ok[-1]}" if ok else "none"
    X = ttnn.from_torch(x.to(torch.bfloat16).reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = {}
    for name, fn, order in (("gen shipped", ttnn.rms_norm, "shipped"), ("gen native-order", ttnn.rms_norm, "native"), ("native", ttnn._native_rms_norm, "shipped")):
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
        o = ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, W)
        ones = (x == 1)
        s = torch.stack([o[r][ones[r]] for r in range(R)])            # scale read at every ones position
        assert all((s[r] == s[r,0]).all() for r in range(R)), "ones positions disagree"
        res[name] = s[:,0]
    desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = "shipped", 0
    S_exact = x.pow(2).sum(-1)
    print(f"{'row':>3s} | {'row content':34s} | {'exact S':>7s} {'exact s':>9s} | " + " | ".join(f"{n:>17s} {'believed S':>10s}" for n in res))
    for r in [0,1,2,3,4,5,6,7,8,15,17,20,23,24]:
        se = 1/((S_exact[r].item()/W+eps)**0.5)
        cells = " | ".join(f"{res[n][r].item():17.6f} {implied_S(res[n][r].item()):>10s}" for n in res)
        print(f"{r:3d} | {desc_rows[r]:34s} | {S_exact[r].item():7.0f} {se:9.6f} | {cells}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
