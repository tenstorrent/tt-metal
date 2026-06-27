import torch, ttnn
from ttnn.operations.matmul import matmul

device = ttnn.open_device(device_id=0)


def pcc(g, c):
    a = g.flatten().double()
    b = c.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def relrms(g, c):
    return float(((c - g).pow(2).mean().sqrt()) / (g.std() + 1e-12))


def run(ash, bsh, dt, wdt, acc, precise):
    torch.manual_seed(0)
    A = torch.randn(ash, dtype=torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.bfloat16)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False, bfp8_pack_precise=precise
    )
    out = ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()
    return pcc(exp, out), relrms(exp, out)


try:
    cases = [
        ("A512x4096 bf8b acc=F", (512, 4096), (4096, 4096), ttnn.bfloat8_b, ttnn.bfloat8_b, False, 0.15),
        ("A256x8192 bf8b acc=F", (256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, False, 0.15),
        ("A256x8192 bf16 acc=F", (256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, False, 0.10),
    ]
    for name, ash, bsh, dt, wdt, acc, tgt in cases:
        p0, r0 = run(ash, bsh, dt, wdt, acc, False)
        p1, r1 = run(ash, bsh, dt, wdt, acc, True)
        print(f"{name}: default PCC={p0:.5f} rms={r0:.4f} | precise PCC={p1:.5f} rms={r1:.4f} | target rms<={tgt}")
finally:
    ttnn.close_device(device)
