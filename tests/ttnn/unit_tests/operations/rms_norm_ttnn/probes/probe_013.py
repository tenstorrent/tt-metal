import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn, default_compute_kernel_config

device = ttnn.open_device(device_id=0)
shape = (32, 64)
torch.manual_seed(42)
x = (torch.rand(shape) + 0.5).to(torch.float32)
exp = torch_rms_norm_ttnn(x, epsilon=1e-12).float().flatten()
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
print(f"{'cfg':34s} {'pcc':>10s} {'rms/std':>9s} {'max_abs':>9s}")


def go(lab, cfg):
    a = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, compute_kernel_config=cfg)).float().flatten()
    pcc = torch.corrcoef(torch.stack([a, exp]))[0, 1].item()
    rms = ((a - exp).pow(2).mean().sqrt() / exp.std()).item()
    print(f"{lab:34s} {pcc:10.6f} {rms:9.5f} {(a-exp).abs().max().item():9.5f}", flush=True)


go("DEFAULT (HiFi4/approx/dest16)", None)
for approx in (True, False):
    for dest in (False, True):
        c = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=dest, math_approx_mode=approx
        )
        go(f"approx={int(approx)} fp32_dest={int(dest)}", c)
ttnn.close_device(device)
