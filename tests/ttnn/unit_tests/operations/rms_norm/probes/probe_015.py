import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g, eps=1e-6):
    xf = x.float()
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / r
    if g is not None:
        out = out * g.float().reshape(-1)
    return out


shape = (1, 1, 32, 128)
t = torch.randn(*shape, dtype=torch.bfloat16)
g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
gi = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
o = ttnn.to_torch(rms_norm(ti, gamma=gi))
e = ref(t, g)
print("baseline max diff", (o.float() - e).abs().max().item())
