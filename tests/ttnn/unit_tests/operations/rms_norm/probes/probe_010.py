import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def ref(x, g):
    xf = x.float()
    o = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    return o * g.float().reshape(-1)


# shapes that FAILED with severity=bug in the golden run
shapes = [(1, 1, 8192, 1024), (7136, 736), (3, 3104, 544), (1, 1, 32, 1024)]
for shape in shapes:
    W = shape[-1]
    for acc in (True, False):
        for lay in (ttnn.TILE_LAYOUT,):
            torch.manual_seed(0)
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(W, dtype=torch.bfloat16)
            e = ref(x, g)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device)
            tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg(acc))
            a = ttnn.to_torch(out).float()
            err = (a - e).abs()
            # per-row (last-dim) max error -> which rows are broken
            flat = err.reshape(-1, W).amax(dim=-1)
            bad = (flat > 0.5).nonzero().flatten()
            print(
                f"[{str(shape):20s} acc={acc}] pcc={pcc(a,e):.6f} rms={(err.pow(2).mean().sqrt()/e.std()).item():.5f} "
                f"max={err.max().item():.4g} bad_rows={len(bad)}/{flat.numel()} first_bad={bad[:8].tolist()}"
            )

ttnn.close_device(device)
