import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from ttnn.operations.rms_norm import rms_norm

desc._FORCE_TRANSPORT = 1   # root-relay gather-then-broadcast
desc._FORCE_REGIME = "B"    # force Regime B

def pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return (a*b).sum() / (a.norm()*b.norm() + 1e-12)

device = ttnn.open_device(device_id=0)
try:
    for shp in [(1,1,32,4096),(1,1,32,8192),(1,1,64,8192),(1,1,32,16384),(1,1,64,12288)]:
        # all-ones exact
        xo = torch.ones(*shp)
        to = ttnn.from_torch(xo.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        oo = ttnn.to_torch(rms_norm(to)).float()
        me = (oo - 1.0).abs().max().item()
        # random vs torch
        x = torch.randn(*shp)
        ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(rms_norm(ti)).float()
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        p = pcc(out, ref).item()
        mx = (out - ref).abs().max().item()
        print(f"RESULT shape={shp} ones_maxerr={me:.4f} PCC={p:.6f} maxerr={mx:.4f}")
finally:
    ttnn.close_device(device)
