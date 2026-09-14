import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations._op_contract import ExcludedCell


def ref(x, g):
    xf = x.float()
    y = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    return y * g.float().reshape(-1) if g is not None else y


def metrics(exp, got):
    e, a = exp.flatten(), got.flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    rel_rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    return pcc, rel_rms


cells = [
    # (shape, dtype, gamma_dtype, gamma_layout, fp32_acc, fidelity, tag)
    (
        (1, 1, 32, 4096),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        False,
        ttnn.MathFidelity.HiFi2,
        "bf16/16bit R2 perf-config",
    ),
    (
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        False,
        ttnn.MathFidelity.HiFi2,
        "bf16/16bit decode 7168 (26 partials)",
    ),
    (
        (2, 4, 128, 512),
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        False,
        ttnn.MathFidelity.HiFi4,
        "bf16/16bit R1 fp32 RM gamma (aliased CB)",
    ),
    (
        (1, 1, 64, 128),
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        True,
        ttnn.MathFidelity.HiFi4,
        "bf8b/fp32 small",
    ),
    (
        (1, 1, 32, 4096),
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        False,
        ttnn.MathFidelity.HiFi4,
        "bf8b/16bit R2",
    ),
    (
        (1, 1, 32, 4096),
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        True,
        ttnn.MathFidelity.HiFi4,
        "bf16 x, bf8b gamma",
    ),
]
for shape, dt, gdt, gl, acc, fid, tag in cells:
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    g = torch.randn(shape[-1]).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=gdt, layout=gl, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=acc, math_approx_mode=False)
    out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg)).float()
    # reference from the dtype-rounded host inputs
    xr = ttnn.to_torch(tx)
    gr = ttnn.to_torch(tg)
    pcc, rr = metrics(ref(xr, gr), out)
    print(f"PROBE {tag}: pcc={pcc:.6f} rel_rms={rr:.5f}")

# excluded cell must still refuse
tx = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
try:
    rms_norm(tx, compute_kernel_config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False))
    print("PROBE EXCLUSION: NOT REFUSED (BUG)")
except ExcludedCell as e:
    print("PROBE EXCLUSION refused OK:", e)
