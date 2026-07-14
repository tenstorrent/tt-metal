import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(fp32=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c


def ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def metrics(act, exp):
    act = act.to(torch.float32)
    exp = exp.to(torch.float32)
    n = act.numel()
    diff = act - exp
    abs_rms = (diff.pow(2).sum(dtype=torch.float64) / n).item() ** 0.5
    std = exp.std().item()
    rms = abs_rms / std if std > 1e-12 else abs_rms
    a = act.flatten()
    e = exp.flatten()
    ac = a - a.mean()
    ec = e - e.mean()
    pcc = (
        (ac * ec).sum(dtype=torch.float64)
        / (ac.pow(2).sum(dtype=torch.float64) * ec.pow(2).sum(dtype=torch.float64)).sqrt()
    ).item()
    return pcc, rms


print("=== CASE 1: bf16 + fp32_dest_acc_en=True, randn, (1,1,32,W) ===")
print(f"{'W':>7} {'pcc':>10} {'rms(<=0.04)':>12} {'ratio_med':>10}")
torch.manual_seed(0)
for W in [8192, 16384, 32768]:
    shape = (1, 1, 32, W)
    x = torch.randn(shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(ti, epsilon=1e-6, compute_kernel_config=cfg(True))
    act = ttnn.to_torch(out).reshape(shape).to(torch.float32)
    exp = ref(x)
    pcc, rms = metrics(act, exp)
    std = exp.std().item()
    mask = exp.abs() > 1e-3 * std
    r = (act[mask] / exp[mask]).median().item()
    print(f"{W:>7} {pcc:>10.6f} {rms:>12.4f} {r:>10.5f}")

print("\n=== CASE 2: bf16 + fp32_dest_acc_en=False, uniform rand + gamma, (1,h,4096) ===")
print(f"{'shape':>14} {'pcc':>10} {'rms':>10} {'frob':>10}")
torch.manual_seed(0)
for h in [24, 128]:
    W = 4096
    x = torch.rand((1, h, W), dtype=torch.bfloat16)
    g = torch.rand((W,), dtype=torch.bfloat16)
    exp = ref(x, gamma=g)
    ti = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gt = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(ti, gamma=gt, compute_kernel_config=cfg(False))
    act = ttnn.to_torch(out).reshape((1, h, W)).to(torch.float32)
    pcc, rms = metrics(act, exp)
    frob = (torch.linalg.norm((act - exp).flatten()) / torch.linalg.norm(exp.flatten())).item()
    print(f"{str((1,h,W)):>14} {pcc:>10.6f} {rms:>10.5f} {frob:>10.5f}")

ttnn.close_device(device)
