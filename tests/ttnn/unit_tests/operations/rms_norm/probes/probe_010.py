import torch, ttnn
import ttnn.operations.rms_norm as M

M.SUPPORTED["layout"] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
M.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]
M.SUPPORTED["gamma_layout"] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
from ttnn.operations.rms_norm import rms_norm


def ref(x, g=None, eps=1e-6):
    xf = x.float()
    rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + eps)
    o = xf / rms
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


dev = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 64), (1, 1, 64, 128)]:
        x = torch.randn(shape)
        ri = ttnn.from_torch(
            x.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o = rms_norm(ri)
        out = ttnn.to_torch(o)
        print(
            f"RM nogamma {shape}: out_layout={o.layout} pcc={pcc(out,ref(x)):.5f} maxerr={(out.float()-ref(x)).abs().max():.4f}"
        )
finally:
    ttnn.close_device(dev)
