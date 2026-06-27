"""EXPERIMENT 3c — reproduce the golden deep-K cells with golden tolerances.

Mirrors eval/golden_tests/matmul/helpers.py (seed=0, HiFi4, golden TOLERANCES)
over the SUPPORTED deep-K shapes (tile-aligned, single weight) x all (act,weight)
dtype pairs (no fp32+acc=False, an EXCLUSION) x acc in {True,False}. Reports
PASS/FAIL against the per-(effective-dtype, acc) tolerance so we can see exactly
which of the 14-fail-signature cells Lever B now fixes and whether any regress.
"""
import torch
import ttnn
from ttnn.operations.matmul import matmul

device = ttnn.open_device(device_id=0)

F32, BF16, BF8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
_COARSE = {F32: 0, BF16: 1, BF8: 2}
TOL = {
    (F32, True): (0.999, 0.02),
    (BF16, True): (0.997, 0.04),
    (BF16, False): (0.99, 0.10),
    (BF8, True): (0.98, 0.12),
    (BF8, False): (0.98, 0.15),
}


def eff(dt, wdt):
    return dt if _COARSE[dt] >= _COARSE[wdt] else wdt


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
    A = torch.randn(ash, dtype=torch.float32 if dt == F32 else torch.bfloat16)
    B = torch.randn(bsh, dtype=torch.float32 if wdt == F32 else torch.bfloat16)
    exp = A.float() @ B.float()
    a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(B, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = ttnn.to_torch(matmul(a, b, compute_kernel_config=cfg)).float()
    return pcc(exp, out), relrms(exp, out)


# SUPPORTED deep-K shapes (tile-aligned, single 2D weight) with K>=4096.
DEEP = [
    ((512, 4096), (4096, 4096)),
    ((128, 4096), (4096, 11008)),
    ((256, 8192), (8192, 2048)),
    ((1, 512, 4096), (4096, 8192)),
]
# all (act, weight) dtype pairs except fp32+acc=False excluded combos
DTYPE_PAIRS = [(BF16, BF16), (BF8, BF8), (BF16, BF8), (BF8, BF16), (BF16, F32), (BF8, F32), (F32, BF16), (F32, BF8)]

try:
    n_fail = 0
    n_total = 0
    print("=== golden deep-K cells, acc=False (the 14-fail signature) ===")
    for ash, bsh in DEEP:
        for dt, wdt in DTYPE_PAIRS:
            if dt == F32:  # fp32 activation + acc=False is an op EXCLUSION (never runs)
                continue
            e = eff(dt, wdt)
            pcc_floor, rms_floor = TOL[(e, False)]
            p, r = run(ash, bsh, dt, wdt, False)
            ok = (p >= pcc_floor) and (r <= rms_floor)
            n_total += 1
            if not ok:
                n_fail += 1
            print(
                f"  {ash}@{bsh} act={str(dt)[9:]:9s} wt={str(wdt)[9:]:9s} eff={str(e)[9:]:9s}"
                f" PCC={p:.4f} rms={r:.4f} tol(pcc>={pcc_floor},rms<={rms_floor}) {'PASS' if ok else 'FAIL'}"
            )
    print(f"\nacc=False deep-K: {n_total - n_fail}/{n_total} PASS, {n_fail} FAIL")

    print("\n=== regression: same cells acc=True (must stay PASS) ===")
    n_fail_t = 0
    n_total_t = 0
    for ash, bsh in DEEP:
        for dt, wdt in DTYPE_PAIRS:
            e = eff(dt, wdt)
            pcc_floor, rms_floor = TOL[(e, True)]
            p, r = run(ash, bsh, dt, wdt, True)
            ok = (p >= pcc_floor) and (r <= rms_floor)
            n_total_t += 1
            if not ok:
                n_fail_t += 1
                print(f"  REGRESSION {ash}@{bsh} act={str(dt)[9:]} wt={str(wdt)[9:]} PCC={p:.4f} rms={r:.4f} FAIL")
    print(f"acc=True deep-K: {n_total_t - n_fail_t}/{n_total_t} PASS, {n_fail_t} FAIL")
finally:
    ttnn.close_device(device)
