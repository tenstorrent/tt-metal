"""EXPERIMENT 1 — confirm deep-K acc=False error is the 16-bit DEST accumulation.

(a) K-sweep at fixed M,N (bf8b/bf8b acc=False): show RMS grows ~O(sqrt(K)).
(b) Isolate the bf8b-interm spill as secondary contributor: at fixed deep K
    compare acc=False with bf8b output (interm forced bf8b) vs bf16 output
    (interm bf16). Both keep the 16-bit DEST; only the interm format differs.
(c) acc=True reference (interm fp32, DEST fp32) — the "fully fixed" floor.
Worst case: A256x8192 bf8b/bf8b acc=False.
"""
import math
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


def _bf8b_quant(t):
    """Round-trip a tensor through ttnn bf8b to mimic bf8b INPUT quantization,
    returning a bf16 tensor that carries the bf8b-quantized values. Lets us hold
    the input-quant level fixed while varying only the on-device dtype/interm."""
    q = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(q).to(torch.bfloat16)


def run(ash, bsh, dt, wdt, acc, prequant_bf8b=False):
    torch.manual_seed(0)
    A = torch.randn(ash, dtype=torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.bfloat16)
    if prequant_bf8b:
        A = _bf8b_quant(A)
        B = _bf8b_quant(B)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()
    return pcc(exp, out), relrms(exp, out)


try:
    print("=== (a) K-sweep, M=256 N=2048, bf8b/bf8b acc=False ===")
    for K in [512, 1024, 2048, 4096, 8192]:
        p, r = run((256, K), (K, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, False)
        print(f"  K={K:5d}: PCC={p:.5f} relRMS={r:.4f}  rms/sqrt(K)={r/math.sqrt(K):.6f}")

    print("\n=== (b) interm-format isolation @ deep K (M=256 N=2048 K=8192), acc=False ===")
    p8, r8 = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, False)
    print(f"  bf8b out (interm=bf8b), acc=False: PCC={p8:.5f} relRMS={r8:.4f}")
    p16, r16 = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, False)
    print(f"  bf16 out (interm=bf16), acc=False: PCC={p16:.5f} relRMS={r16:.4f}")
    # Control for input quant: bf16-DTYPE tensors carrying bf8b-quantized values.
    # Same input quant as the bf8b case, but interm=bf16 (no bf8b spill rounding).
    pc, rc = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, False, prequant_bf8b=True)
    print(f"  bf16 out, bf8b-quant inputs, interm=bf16, acc=False: PCC={pc:.5f} relRMS={rc:.4f}")

    print("\n=== (c) acc=True reference (interm fp32, DEST fp32) — fully fixed floor ===")
    pt, rt = run((256, 8192), (8192, 2048), ttnn.bfloat8_b, ttnn.bfloat8_b, True)
    print(f"  bf8b/bf8b acc=True : PCC={pt:.5f} relRMS={rt:.4f}")
    pt16, rt16 = run((256, 8192), (8192, 2048), ttnn.bfloat16, ttnn.bfloat16, True)
    print(f"  bf16/bf16 acc=True : PCC={pt16:.5f} relRMS={rt16:.4f}")
finally:
    ttnn.close_device(device)
