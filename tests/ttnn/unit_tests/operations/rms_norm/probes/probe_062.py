import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod


def rel_rms(o, r):
    return (torch.sqrt(torch.mean((o - r) ** 2)) / (torch.sqrt(torch.mean(r**2)) + 1e-30)).item()


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden(x, g, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()).to(torch.bfloat16)


eps = 1e-6
saved = pdmod.GRID_W
# One device session per GRID_W so the ttnn program cache cannot serve a stale plan.
for gw in (1, 8, 16, 28, 32, 56):
    device = ttnn.open_device(device_id=0)
    try:
        pdmod.GRID_W = gw
        shape = (1, 1, 32, 7168)
        torch.manual_seed(3)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn((7168,), dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(ttnn.rms_norm(xt, weight=gt, epsilon=eps))
        ref = golden(x, g, eps)
        tree = pdmod._combine_tree_arity(gw, 1) if gw > 1 else None
        print(f"GRID_W={gw:3d} tree={str(tree):8s} pcc={pcc(o, ref):.7f} rel_rms={rel_rms(o.float(), ref.float()):.6f}")
    finally:
        pdmod.GRID_W = saved
        ttnn.close_device(device)
