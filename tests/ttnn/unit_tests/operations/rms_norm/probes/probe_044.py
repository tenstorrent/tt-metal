import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
grid = dev.compute_with_storage_grid_size()
print("GRID:", grid.x, grid.y)


def cfg(acc=False, fid=ttnn.MathFidelity.HiFi2):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def ref(x, g, eps=1e-6):
    xf = x.float()
    rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + eps)
    o = xf / rms
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


CASES = [
    ((1, 1, 32, 7168), True),
    ((1, 1, 32, 1024), True),
    ((1, 1, 32, 4096), True),
    ((1, 1, 64, 12288), True),
    ((1, 1, 224, 3072), True),
    ((1, 1, 224, 1000), True),  # partial_w + split
    ((1, 1, 32, 16384), True),
    ((1024, 1024), True),
    ((1, 1, 8192, 1024), True),  # prefill -> no split
    ((1, 1, 32, 4064), True),  # prime Wt -> no split
    ((1, 1, 32, 7168), False),  # no gamma
]
for shape, hasg in CASES:
    torch.manual_seed(0)
    W = shape[-1]
    xt = torch.randn(shape, dtype=torch.bfloat16)
    gt = torch.randn(W, dtype=torch.bfloat16) if hasg else None
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = (
        ttnn.from_torch(
            gt.reshape(*([1] * (len(shape) - 1)), W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        if hasg
        else None
    )
    # what plan did we get?
    Rt = 1
    out = rms_norm(x, gamma=g, compute_kernel_config=cfg())
    got = ttnn.to_torch(out).float()
    exp = ref(xt, gt)
    d = got - exp
    pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
    rms = (d.norm() / exp.norm()).item()
    print(f"{str(shape):22s} gamma={hasg}  pcc={pcc:.6f} relrms={rms:.5f}")
ttnn.close_device(dev)
