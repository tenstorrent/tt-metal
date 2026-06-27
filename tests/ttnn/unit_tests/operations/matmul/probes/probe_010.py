"""EXPERIMENT 3b — Lever B robustness: multi-seed, mixed-dtype-out, shallow-K.

Rules out a lucky-seed artifact and checks the dtype-routing of Lever B
(triggered iff acc=False AND OUTPUT dtype == bf8b; output dtype == ACT dtype).
  - multi-seed deep-K bf8b acc=False (should all PASS, tight RMS).
  - bf8b-act / bf16-weight, acc=False: OUTPUT=bf8b -> Lever B fires. Must work.
  - bf16-act / bf8b-weight, acc=False: OUTPUT=bf16 -> default path (bf16 interm),
    Lever B does NOT fire. Reported for contrast.
  - shallow K bf8b acc=False (num_k_blocks==1): Lever B still set; confirm no
    corruption when there is no spill.
"""
import torch
import ttnn
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


def run(ash, bsh, dt, wdt, acc, seed):
    torch.manual_seed(seed)
    A = torch.randn(ash, dtype=torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.bfloat16)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out_t = matmul(a, b, compute_kernel_config=cfg)
    out = ttnn.to_torch(out_t).float()
    return pcc(exp, out), relrms(exp, out), str(out_t.dtype)


try:
    print("=== multi-seed deep-K bf8b/bf8b acc=False, A256x8192 (target 0.15) ===")
    for s in [0, 1, 7, 42, 123]:
        p, r, od = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, False, s)
        print(f"  seed={s:4d}: out={od} PCC={p:.5f} relRMS={r:.4f} {'PASS' if r <= 0.15 else 'MISS'}")

    print("\n=== mixed-dtype outputs, deep-K acc=False ===")
    p, r, od = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat16, False, 0)
    print(f"  bf8b-act/bf16-wt out={od} (Lever B fires): PCC={p:.5f} relRMS={r:.4f} {'PASS' if r <= 0.15 else 'MISS'}")
    p, r, od = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat8_b, False, 0)
    print(f"  bf16-act/bf8b-wt out={od} (default path): PCC={p:.5f} relRMS={r:.4f}")

    print("\n=== shallow K bf8b acc=False (num_k_blocks may be 1) ===")
    for ash, bsh in [((64, 32), (32, 64)), ((128, 256), (256, 128)), ((256, 1024), (1024, 256))]:
        p, r, od = run(ash, bsh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, 0)
        print(f"  {ash}@{bsh}: out={od} PCC={p:.5f} relRMS={r:.4f}")
finally:
    ttnn.close_device(device)
