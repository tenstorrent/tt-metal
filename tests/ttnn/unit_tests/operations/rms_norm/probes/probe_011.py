import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

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
    return (xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)


# class B (was catastrophic) + class A (precision near-miss) shapes
shapes = [
    (1, 1, 8192, 1024),
    (7136, 736),
    (3, 3104, 544),
    (5, 3, 928, 544),
    (99991, 64),
    (13, 777, 1023),
    (1, 1, 160, 11008),
    (1, 224, 11008),
    (1, 1, 32, 7168),
    (1, 1, 96, 6144),
]
for shape in shapes:
    W = shape[-1]
    row = f"[{str(shape):18s}]"
    for acc in (True, False):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn(W, dtype=torch.bfloat16)
        e = ref(x, g)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg(acc))
        a = ttnn.to_torch(out).float()
        err = (a - e).abs()
        nbad = (err.reshape(-1, W).amax(dim=-1) > 0.5).sum().item()
        row += f"  acc={int(acc)}: pcc={pcc(a,e):.6f} rms={(err.pow(2).mean().sqrt()/e.std()).item():.5f} bad={nbad}"
    print(row)

ttnn.close_device(device)
