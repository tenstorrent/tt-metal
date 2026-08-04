import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    W = 64
    x = torch.full((1, 1, 32, W), 1.0, dtype=torch.float32)
    g = torch.ones(W, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for approx in (False, True):
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = approx
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=c)).to(torch.float64)
        print(f"APPROX math_approx_mode={approx}: got={out[0,0,0,0].item():.10f}")
finally:
    ttnn.close_device(dev)
