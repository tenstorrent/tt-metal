"""EXPERIMENT 2 — Lever A: K-block size (in0_block_w / num_k_blocks).

Default descriptor: in0_block_w = find_max_divisor(Kt, 4) -> Kt=256 (K=8192)
gives in0_block_w=4, num_k_blocks=64. Each K-block boundary forces a
DEST->interm->DEST round trip that rounds to 16-bit DEST (acc=False). FEWER,
LARGER K-blocks = fewer rounding boundaries.

Hypothesis: larger in0_block_w (fewer num_k_blocks) lowers deep-K acc=False RMS.
We monkeypatch _find_max_divisor to force in0_block_w in {1,2,4,8,16}, keeping
L1 bounded (the descriptor's footprint loop shrinks bM/bN to fit anyway).
Worst case: A256x8192 bf8b/bf8b acc=False (target rms<=0.15).
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
    if forced_w is None:
        pdmod._find_max_divisor = _orig_fmd
    else:
        # Force the K-block width: largest divisor of Kt that is <= forced_w.
        # (Only override the Kt-vs-4 call; leave any other use intact by checking max_div.)
        def patched(value, max_div):
            if max_div == 4:  # this is the in0_block_w selection call
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
    ash, bsh = (256, 8192), (8192, 2048)  # Kt=256
    print("=== Lever A: K-block width sweep, A256x8192 bf8b/bf8b acc=False (Kt=256) ===")
    print("  (forced_w=4 is the current default; lower=more blocks, higher=fewer blocks)")
    for w in [1, 2, 4, 8, 16, 32]:
        p, r = run(ash, bsh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, w)
        nkb = 256 // w if 256 % w == 0 else "n/a"
        flag = "  <-- DEFAULT" if w == 4 else ""
        print(f"  in0_block_w={w:2d} (num_k_blocks={nkb}): PCC={p:.5f} relRMS={r:.4f}{flag}")

    print("\n=== same sweep, bf16/bf16 acc=False (target rms<=0.10) ===")
    for w in [1, 2, 4, 8, 16, 32]:
        p, r = run(ash, bsh, ttnn.bfloat16, ttnn.bfloat16, False, w)
        nkb = 256 // w if 256 % w == 0 else "n/a"
        flag = "  <-- DEFAULT" if w == 4 else ""
        print(f"  in0_block_w={w:2d} (num_k_blocks={nkb}): PCC={p:.5f} relRMS={r:.4f}{flag}")
finally:
    ttnn.close_device(device)
