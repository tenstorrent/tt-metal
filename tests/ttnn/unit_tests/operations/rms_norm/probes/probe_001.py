import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if g is not None:
        out = out * g.to(torch.float32).reshape(-1)
    return out


cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
cfg.math_approx_mode = False

torch.manual_seed(0)
shape = (1, 1, 32, 32)
x = torch.randn(shape, dtype=torch.bfloat16)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = rms_norm(tx, compute_kernel_config=cfg)
a = ttnn.to_torch(out).to(torch.float32)
e = ref(x)
print("max diff", (a - e).abs().max().item())
print("a[0,0,0,:6]", a[0, 0, 0, :6])
print("e[0,0,0,:6]", e[0, 0, 0, :6])
