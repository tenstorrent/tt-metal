import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def ref(x, g, eps=1e-6):
    xf = x.float()
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / r
    if g is not None:
        out = out * g.float().reshape(-1)
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


cases = [
    ((1, 1, 32, 64), True),
    ((1, 1, 32, 128), True),
    ((1, 1, 64, 256), True),
    ((1, 1, 32, 4096), True),
    ((1, 1, 32, 1024), False),
    ((1, 1, 32, 50), True),
    ((1, 1, 17, 64), True),
    ((2, 1, 64, 512), True),
    ((1, 1, 320, 128), True),
]
for shape, has_g in cases:
    t = torch.randn(*shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16) if has_g else None
    ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gi = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if has_g else None
    o = ttnn.to_torch(rms_norm(ti, gamma=gi)).float()
    e = ref(t, g)
    print(f"{shape} gamma={has_g}: pcc={pcc(o,e):.6f} maxdiff={(o-e).abs().max().item():.4f}")
ttnn.close_device(device)
