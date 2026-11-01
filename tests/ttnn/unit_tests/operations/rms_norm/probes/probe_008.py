import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

d = ttnn.open_device(device_id=0)
for shape in [(1, 1, 544, 11008)]:
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
    try:
        y = ttnn.to_torch(rms_norm(tx, gamma=tg)).float()
        ref = x.float() / torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6) * g.float()
        print("PROBE", shape, "pcc", torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item())
    except Exception as e:
        print("PROBE", shape, "EXC", str(e)[:300])
ttnn.close_device(d)
