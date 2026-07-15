import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g, eps):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(ms + eps)
    if g is not None:
        y = y * g.reshape(1, 1, 1, -1)
    return y


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


eps = 1e-6
torch.manual_seed(0)
fails = 0
for shape in [(1, 1, 32, 256), (1, 1, 32, 8192), (1, 1, 64, 8192)]:
    W = shape[-1]
    for layout, lname in [(ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")]:
        for wg in [False, True]:
            x = torch.randn(shape, dtype=torch.float32)
            g = torch.randn(W, dtype=torch.float32) if wg else None
            tx = ttnn.from_torch(
                x, dtype=ttnn.float32, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tg = (
                ttnn.from_torch(
                    g.reshape(1, 1, 1, W),
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if wg
                else None
            )
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi4
            cfg.fp32_dest_acc_en = True
            cfg.math_approx_mode = False
            out = rms_norm(tx, gamma=tg, epsilon=eps, compute_kernel_config=cfg)
            r = ttnn.to_torch(out).float()
            e = ref(x, g, eps)
            p = pcc(r, e)
            md = (r - e).abs().max().item()
            ok = p > 0.9999 and md < 0.02
            if not ok:
                fails += 1
            print(f"{shape} {lname} {'wg' if wg else 'nog'}: PCC={p:.6f} maxabs={md:.4f} {'OK' if ok else 'FAIL'}")
print("FAILS:", fails)
