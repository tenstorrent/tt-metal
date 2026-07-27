"""R1: precision cost of fp32_dest_acc_en=False (informs the Refinement 1b follow-up)."""
import torch, ttnn
from ttnn.operations.onorm import onorm, default_compute_kernel_config

HV, V = 32, 128
FLAT = HV * V
EPS = 1e-5


def ref(o, g, w):
    f = o.to(torch.float32)
    n = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + EPS)
    n = n * w.to(torch.float32).reshape(1, 1, 1, V)
    return n.reshape(o.shape[0], o.shape[1], FLAT) * torch.sigmoid(g.to(torch.float32))


CFGS = {
    "fp32_dest_on ": default_compute_kernel_config(),
    "fp32_dest_off": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    ),
}

print(f"{'shape':>10} {'config':>14} {'PCC':>12} {'rel_RMS':>9} {'max_abs':>9} {'ratio_med':>10}")
for b, t in [(1, 32), (1, 128), (1, 640), (4, 256)]:
    torch.manual_seed(42)
    t_o = torch.randn(b, t, HV, V, dtype=torch.bfloat16)
    t_g = torch.randn(b, t, FLAT, dtype=torch.bfloat16)
    t_w = (1.0 + 0.02 * torch.randn(1, 1, 1, V)).to(torch.bfloat16)
    e = ref(t_o, t_g, t_w)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for name, cfg in CFGS.items():
        got = ttnn.to_torch(onorm(o, g, w, compute_kernel_config=cfg)).to(torch.float32)
        d = got - e
        pcc = torch.corrcoef(torch.stack([e.flatten().double(), got.flatten().double()]))[0, 1].item()
        rms = (d.pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
        m = e.abs() > 0.1 * e.abs().median()
        rmed = (got[m] / e[m]).double().median().item()
        print(f"{f'B{b}/T{t}':>10} {name:>14} {pcc:>12.6f} {rms:>9.4f} {d.abs().max().item():>9.4f} {rmed:>10.4f}")
