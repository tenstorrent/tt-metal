import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod

device = ttnn.open_device(device_id=0)


def cfg(acc, fid):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def ref(x):
    xf = x.float()
    return xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)


FIDS = [
    ("LoFi", ttnn.MathFidelity.LoFi),
    ("HiFi2", ttnn.MathFidelity.HiFi2),
    ("HiFi3", ttnn.MathFidelity.HiFi3),
    ("HiFi4", ttnn.MathFidelity.HiFi4),
]

for shape in [(1, 1, 32, 7168), (1, 1, 160, 11008), (1, 1, 32, 1024)]:
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    e = ref(x)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for bulk in (1, 0):
        pdmod.REDUCE_BULK = bulk
        line = f"[{str(shape):18s} BULK={bulk}]"
        for fname, fid in FIDS:
            a = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg(False, fid))).float()
            r = ((a - e).abs().pow(2).mean().sqrt() / e.std()).item()
            line += f" {fname}={r:.5f}"
        # reference: acc=True HiFi4
        a = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg(True, ttnn.MathFidelity.HiFi4))).float()
        line += f" | accTrue={((a-e).abs().pow(2).mean().sqrt()/e.std()).item():.5f}"
        print(line)
    pdmod.REDUCE_BULK = 1
ttnn.close_device(device)
