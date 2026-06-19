import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    o = x * torch.rsqrt(v + eps)
    return o * g if g is not None else o


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    # A multi-row/core gamma (gamma reuse + recip pop across rows), num_chunks>1, B gamma
    cases = [(1, 1, 8192, 64), (1, 1, 2048, 256), (2, 4, 128, 512), (1, 1, 32, 4096), (1, 1, 64, 12288)]
    for shp in cases:
        W = shp[-1]
        x = torch.randn(shp)
        g = torch.randn(W)
        ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.to_torch(rms_norm(ti, gamma=tg)).float()
        gr = ref(x.float(), g.float())
        pcc = torch.corrcoef(torch.stack([o.flatten(), gr.flatten()]))[0, 1].item()
        rms = ((o - gr).pow(2).mean().sqrt() / gr.pow(2).mean().sqrt()).item()
        print(f"{str(shp):16} GAMMA pcc={pcc:.5f} relRMS={rms:.4f}")
finally:
    ttnn.close_device(dev)
