import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    o = x * torch.rsqrt(var + eps)
    return o * g if g is not None else o


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    # gamma cases (Regime B) + LOOSE large-W
    cases = [(1, 1, 32, 4096), (1, 1, 64, 12288), (1, 32, 8192), (2, 1, 64, 4096)]
    loose = [(1, 1, 32, 16384), (1, 1, 32, 32768), (1, 1, 64, 12288)]
    for shp in cases:
        W = shp[-1]
        xr = torch.randn(shp)
        g = torch.randn(W)
        tr = ttnn.from_torch(xr.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.to_torch(rms_norm(tr, gamma=tg)).float()
        gr = ref(xr.float(), g.float())
        pcc = torch.corrcoef(torch.stack([o.flatten(), gr.flatten()]))[0, 1].item()
        rms = ((o - gr).pow(2).mean().sqrt() / gr.pow(2).mean().sqrt()).item()
        print(f"GAMMA {str(shp):16} pcc={pcc:.5f} relRMS={rms:.4f}")
    for shp in loose:
        x1 = torch.ones(shp)
        t1 = ttnn.from_torch(x1.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o1 = ttnn.to_torch(rms_norm(t1)).float()
        print(f"LOOSE {str(shp):16} ones_mean={o1.mean().item():.4f}")
finally:
    ttnn.close_device(dev)
