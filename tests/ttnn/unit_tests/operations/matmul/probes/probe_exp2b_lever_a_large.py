"""EXPERIMENT 2b — Lever A at LARGER K-block widths (64, 128) + 2nd failing shape.

Continues Lever A: does a still-larger in0_block_w push bf8b acc=False under
0.15, and does the footprint loop keep L1 bounded (no hang/OOM)? Also checks
the A512x4096 failing cell. forced_w only narrows num_k_blocks; the descriptor's
footprint loop shrinks bM/bN to fit L1 automatically.
"""
import torch
import ttnn
from ttnn.operations.matmul import matmul
import ttnn.operations.matmul.matmul_program_descriptor as pdmod

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


_orig_fmd = pdmod._find_max_divisor


def run(ash, bsh, dt, wdt, acc, forced_w):
    torch.manual_seed(0)
    A = torch.randn(ash, dtype=torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.bfloat16)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )

    def patched(value, max_div):
        if max_div == 4:
            for d in range(min(forced_w, value), 0, -1):
                if value % d == 0:
                    return d
            return 1
        return _orig_fmd(value, max_div)

    pdmod._find_max_divisor = patched
    try:
        out = ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()
    finally:
        pdmod._find_max_divisor = _orig_fmd
    return pcc(exp, out), relrms(exp, out)


try:
    print("=== A256x8192 bf8b/bf8b acc=False, LARGE widths (target rms<=0.15) ===")
    for w in [32, 64, 128, 256]:
        try:
            p, r = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, False, w)
            print(f"  in0_block_w={w:3d}: PCC={p:.5f} relRMS={r:.4f} {'PASS' if r <= 0.15 else 'MISS'}")
        except Exception as e:
            print(f"  in0_block_w={w:3d}: EXCEPTION {type(e).__name__}: {e}")

    print("\n=== A512x4096 bf8b/bf8b acc=False (Kt=128), widths (target rms<=0.15) ===")
    for w in [4, 8, 16, 32, 64, 128]:
        try:
            p, r = run((512, 4096), (4096, 4096), ttnn.bfloat8_b, ttnn.bfloat8_b, False, w)
            print(f"  in0_block_w={w:3d}: PCC={p:.5f} relRMS={r:.4f} {'PASS' if r <= 0.15 else 'MISS'}")
        except Exception as e:
            print(f"  in0_block_w={w:3d}: EXCEPTION {type(e).__name__}: {e}")
finally:
    ttnn.close_device(device)
