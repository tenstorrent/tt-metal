import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0)


def run(x, w, fid):
    tx = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tw = ttnn.from_torch(w.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    e = torch.matmul(x.bfloat16().double(), ttnn.to_torch(tw).double())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=False, math_approx_mode=False)
    g = ttnn.to_torch(wq_b_linear(tx, tw, compute_kernel_config=cfg)).double()
    slope = (g * e).sum() / (e * e).sum()
    eb = e.bfloat16().double()
    _, p = comp_pcc(e, g)
    _, pb = comp_pcc(e, eb)
    rms = (g - e).pow(2).mean().sqrt() / e.std()
    rmsb = (eb - e).pow(2).mean().sqrt() / e.std()
    nz = e.abs() > 1e-3 * e.abs().max()
    r = g[nz] / e[nz]
    rb = eb[nz] / e[nz]
    print(
        f"P: {fid} pcc={p:.6f} bf16floor_pcc={pb:.6f} rms={rms:.5f} bf16floor_rms={rmsb:.5f} slope={slope:.5f} rmed={r.median():.5f} rbmed={rb.median():.5f} mean_err={(g-e).mean():.5f}"
    )


torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536)
w = torch.randn(1536, 4096) * 0.02
for f in [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]:
    run(x, w, f)
torch.manual_seed(1)
x = torch.rand(1, 1, 160, 1536)
w = (torch.randn(1536, 4096) + 0.5) * 0.02
for f in [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]:
    run(x, w, f)
ttnn.close_device(dev)
