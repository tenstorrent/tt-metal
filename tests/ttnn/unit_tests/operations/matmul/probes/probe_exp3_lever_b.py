"""EXPERIMENT 3 — Lever B validation: packer_l1_acc=True + bf16 interm, bf8b out, acc=False.

The descriptor now auto-enables packer_l1_acc + Float16_b interm for the
(acc=False, bf8b output) corner. This probe:
  (1) sanity: small shape, bf8b acc=False — does L1-acc path produce correct output
      (no corruption / hang)? Compare PCC/RMS to the bf8b acc=True reference.
  (2) worst case: A256x8192 bf8b/bf8b acc=False (target rms<=0.15) — does Lever B
      pull RMS under tolerance?
  (3) A512x4096 bf8b/bf8b acc=False (target 0.15).
  (4) regression guard: bf16 acc=False, bf16 acc=True, bf8b acc=True still take the
      DEFAULT path (flag off) and are unchanged.
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


def run(ash, bsh, dt, wdt, acc):
    torch.manual_seed(0)
    A = torch.randn(ash, dtype=torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.bfloat16)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()
    return pcc(exp, out), relrms(exp, out)


try:
    print("=== (1) sanity: small bf8b acc=False (Lever B path now active) ===")
    for ash, bsh in [((64, 128), (128, 256)), ((256, 512), (512, 256))]:
        p, r = run(ash, bsh, ttnn.bfloat8_b, ttnn.bfloat8_b, False)
        print(f"  {ash}@{bsh}: PCC={p:.5f} relRMS={r:.4f}  (acc=True ref below)")
        pt, rt = run(ash, bsh, ttnn.bfloat8_b, ttnn.bfloat8_b, True)
        print(f"  {ash}@{bsh} acc=True ref: PCC={pt:.5f} relRMS={rt:.4f}")

    print("\n=== (2)(3) worst-case deep-K, bf8b/bf8b acc=False (target rms<=0.15) ===")
    for ash, bsh, name in [
        ((256, 8192), (8192, 2048), "A256x8192"),
        ((512, 4096), (4096, 4096), "A512x4096"),
        ((256, 8192), (8192, 2048), "A256x8192-rerun"),
    ]:
        p, r = run(ash, bsh, ttnn.bfloat8_b, ttnn.bfloat8_b, False)
        print(f"  {name} {ash}@{bsh}: PCC={p:.5f} relRMS={r:.4f}  {'PASS' if r <= 0.15 else 'MISS'}")

    print("\n=== (4) regression guard: these MUST be unchanged (default path) ===")
    p, r = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, False)
    print(f"  bf16/bf16 acc=False (target 0.10): PCC={p:.5f} relRMS={r:.4f}")
    p, r = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, True)
    print(f"  bf16/bf16 acc=True : PCC={p:.5f} relRMS={r:.4f}")
    p, r = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, True)
    print(f"  bf8b/bf8b acc=True : PCC={p:.5f} relRMS={r:.4f}")
finally:
    ttnn.close_device(device)
