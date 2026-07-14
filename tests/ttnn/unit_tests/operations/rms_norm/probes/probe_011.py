import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def ref(x, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return xf / rms


torch.manual_seed(0)
print(f"{'dtype':>8} {'W':>7} {'ratio_med':>10} {'ratio_std':>10}")
for dt, tdt in [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)]:
    for W in [512, 1024, 2048, 4096, 8192, 16384]:
        shape = (1, 1, 32, W)
        x = torch.randn(shape, dtype=tdt)
        ti = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = rms_norm(ti, epsilon=1e-6, compute_kernel_config=cfg())
        act = ttnn.to_torch(out).reshape(shape).to(torch.float32)
        exp = ref(x)
        std = exp.std().item()
        mask = exp.abs() > 1e-3 * std
        r = act[mask] / exp[mask]
        print(f"{str(dt).split('.')[-1]:>8} {W:>7} {r.median().item():>10.5f} {r.std().item():>10.5f}")
ttnn.close_device(device)
