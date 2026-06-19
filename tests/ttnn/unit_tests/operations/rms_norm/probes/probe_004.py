import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    cases = [
        (1, 1, 32, 4096),
        (1, 1, 32, 1280),
        (1, 1, 64, 8192),
        (1, 1, 128, 4096),
        (1, 1, 256, 2048),
        (1, 1, 64, 12288),
        (1, 32, 4096),
        (128, 512),
    ]
    for shp in cases:
        # all-ones
        x1 = torch.ones(shp)
        t1 = ttnn.from_torch(x1.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o1 = ttnn.to_torch(rms_norm(t1)).float()
        # random
        xr = torch.randn(shp)
        tr = ttnn.from_torch(xr.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        orr = ttnn.to_torch(rms_norm(tr)).float()
        gref = ref(xr.float())
        pcc = torch.corrcoef(torch.stack([orr.flatten(), gref.flatten()]))[0, 1].item()
        rms = ((orr - gref).pow(2).mean().sqrt() / gref.pow(2).mean().sqrt()).item()
        print(f"{str(shp):18} ones_mean={o1.mean().item():.4f}  rand pcc={pcc:.5f} relRMS={rms:.4f}")
finally:
    ttnn.close_device(dev)
