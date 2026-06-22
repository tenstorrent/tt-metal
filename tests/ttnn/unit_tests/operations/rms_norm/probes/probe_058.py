import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from ttnn.operations.rms_norm import rms_norm

desc._FORCE_REGIME = "B"
desc._FORCE_TRANSPORT = 2

device = ttnn.open_device(device_id=0)
try:
    shp = (1, 1, 32, 8192)
    W = shp[-1]
    xo = torch.ones(*shp)
    to = ttnn.from_torch(xo.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    oo = ttnn.to_torch(rms_norm(to)).float()[0, 0]  # [32, W]
    # how many cores K? Wt=256; print number of distinct column-shard values
    row0 = oo[0]  # first stick
    print("row0[:8] =", row0[:8].tolist())
    print("row0 min/max/mean:", row0.min().item(), row0.max().item(), row0.mean().item())
    # Look for which columns are finite vs inf/nan
    finite = torch.isfinite(row0)
    print("num finite cols:", int(finite.sum()), "of", W)
    if int(finite.sum()) > 0:
        idx = finite.nonzero().flatten()
        print("finite col range:", idx.min().item(), idx.max().item(), "sample vals:", row0[idx[:4]].tolist())
    # show value at start of each potential shard (Wt_s tiles). Try K=32 -> Wt_s=8 tiles=256 cols
    for k in [32, 16]:
        Wts_cols = W // k
        vals = [row0[i * Wts_cols].item() for i in range(k)]
        print(f"K={k} per-shard-first-col:", vals[:8], "...")
finally:
    ttnn.close_device(device)
