# Aggregate the precision matrix into the per-config summary for
# precision_matrix_results.md. Same axes/shapes as
# test_rms_norm_precision_matrix.py; worst case over shapes x distributions.
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, EXCLUSIONS

device = ttnn.open_device(device_id=0)
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
SHAPES = [
    (32, 32),
    (1, 1, 64, 128),
    (1, 1, 128, 512),
    (1, 1, 32, 4096),
    (32, 48),
    (48, 64),
    (1, 1, 17, 50),
    (2, 1, 100, 47),
]
DT = [(ttnn.bfloat16, "bfloat16"), (ttnn.float32, "float32"), (ttnn.bfloat8_b, "bfloat8_b")]
FID = [
    (ttnn.MathFidelity.HiFi4, "HiFi4"),
    (ttnn.MathFidelity.HiFi3, "HiFi3"),
    (ttnn.MathFidelity.HiFi2, "HiFi2"),
    (ttnn.MathFidelity.LoFi, "LoFi"),
]


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


print("BEGIN_TABLE")
print("| dtype | fp32_dest_acc_en | math_fidelity | min PCC | max rel-RMS | max abs err |")
print("|---|---|---|---:|---:|---:|")
for dt, dname in DT:
    for acc in (True, False):
        if any(all({"dtype": dt, "fp32_dest_acc_en": acc}.get(k) == v for k, v in e.items()) for e in EXCLUSIONS):
            print(f"| {dname} | {acc} | — | *EXCLUDED (op refuses)* | | |")
            continue
        for fid, fname in FID:
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity = fid
            c.fp32_dest_acc_en = acc
            c.math_approx_mode = False
            worst_pcc, worst_rms, worst_abs = 1.0, 0.0, 0.0
            for shape in SHAPES:
                for gen in (torch.rand, torch.randn):
                    torch.manual_seed(0)
                    x = gen(*shape, dtype=TORCH_DTYPE[dt])
                    g = gen(shape[-1], dtype=TORCH_DTYPE[dt])
                    tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
                    tg = ttnn.from_torch(
                        g.reshape(1, 1, 1, shape[-1]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=c)).float()
                    xf = x.float()
                    exp = xf / torch.sqrt((xf**2).mean(-1, keepdim=True) + 1e-6) * g.float().reshape(-1)
                    e = (got - exp).abs()
                    worst_pcc = min(worst_pcc, pcc(got, exp))
                    worst_rms = max(worst_rms, (e.pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item())
                    worst_abs = max(worst_abs, e.max().item())
            print(f"| {dname} | {acc} | {fname} | {worst_pcc:.6f} | {worst_rms:.3e} | {worst_abs:.3e} |", flush=True)
print("END_TABLE")
ttnn.close_device(device)
