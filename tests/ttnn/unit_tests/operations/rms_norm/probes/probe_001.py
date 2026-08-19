import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 64)]:
        x = torch.ones(shape, dtype=torch.float32)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(tx)).float()
        print("ONES", shape, "min", out.min().item(), "max", out.max().item())
        # monotonic per row
        r = torch.arange(shape[-2], dtype=torch.float32).reshape(1, 1, shape[-2], 1).expand(shape).contiguous()
        tr = ttnn.from_torch(r + 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o2 = ttnn.to_torch(rms_norm(tr)).float()
        print("ROWCONST out[:,:, :4, :4]=", o2[0, 0, :4, :4])
        # column ramp
        c = torch.arange(shape[-1], dtype=torch.float32).reshape(1, 1, 1, shape[-1]).expand(shape).contiguous()
        tc = ttnn.from_torch(c + 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o3 = ttnn.to_torch(rms_norm(tc)).float()
        exp3 = (c + 1.0) / torch.sqrt(((c + 1.0) ** 2).mean(-1, keepdim=True) + 1e-6)
        print("COLRAMP got", o3[0, 0, 0, :8])
        print("COLRAMP exp", exp3[0, 0, 0, :8])
        print("COLRAMP row1 got", o3[0, 0, 1, :8])
finally:
    ttnn.close_device(dev)
