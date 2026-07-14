import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


device = ttnn.open_device(device_id=0)
try:
    H, W = 17, 64
    # row r = constant (r+1). RMSNorm(row) = (r+1)/sqrt((r+1)^2+eps) ~ 1.0 for every col.
    x = torch.zeros(1, 1, H, W, dtype=torch.float32)
    for r in range(H):
        x[0, 0, r, :] = float(r + 1)
    for layout, nm in [(ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")]:
        ti = ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = rms_norm(ti, epsilon=1e-6, compute_kernel_config=cfg())
        act = ttnn.to_torch(out).reshape(1, 1, H, W).to(torch.float32)
        # per-row mean value (should be ~1.0)
        rowmean = act[0, 0].mean(dim=1)
        print(f"{nm}: per-row mean (expect ~1.0 each):")
        print("   ", [round(v, 3) for v in rowmean.tolist()])
finally:
    ttnn.close_device(device)
