import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    for W, val in ((64, 1.0), (64, 2.0), (256, 1.0)):
        x = torch.full((1, 1, 32, W), val, dtype=torch.float32)
        g = torch.ones(W, dtype=torch.float32)
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg())).to(torch.float64)
        exp = val / torch.sqrt(torch.tensor(val * val, dtype=torch.float64) + 1e-6)
        got = out[0, 0, 0, 0].item()
        print(f"W={W} val={val}: expected={exp.item():.10f} got={got:.10f} ratio={got/exp.item():.8f}")
        # also check the sum-of-squares path in isolation via a big-W case
finally:
    ttnn.close_device(dev)
