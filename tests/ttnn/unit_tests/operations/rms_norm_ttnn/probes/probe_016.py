import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:

    def metrics(got, exp):
        g = got.to(torch.float32).flatten()
        t = exp.to(torch.float32).flatten()
        err = (g - t).abs()
        rms = (err.pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item()
        r = g / t
        q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95]))
        pcc = torch.corrcoef(torch.stack([g, t]))[0, 1].item()
        return rms, pcc, q[1].item(), q[0].item(), q[2].item(), r.std().item()

    for name, mk in [
        ("positive_only", lambda s: torch.rand(s) + 0.5),
        ("negative_only", lambda s: -(torch.rand(s) + 0.5)),
        ("randn        ", lambda s: torch.randn(s)),
    ]:
        for shape in [(1, 1, 32, 64), (1, 1, 128, 256)]:
            torch.manual_seed(42)
            x = mk(shape).to(torch.float32)
            exp = torch_rms_norm_ttnn(x, epsilon=1e-12)
            for label, cfg in [
                ("default(dest16)", None),
                (
                    "dest32",
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
                    ),
                ),
            ]:
                tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                got = ttnn.to_torch(rms_norm_ttnn(tx, compute_kernel_config=cfg))
                rms, pcc, med, p5, p95, std = metrics(got, exp)
                print(
                    f"{name} {str(shape):18s} {label:16s} rms={rms:.5f} pcc={pcc:.6f} "
                    f"r_med={med:.6f} r_p5={p5:.6f} r_p95={p95:.6f} r_std={std:.2e}"
                )
        print()
finally:
    ttnn.close_device(device)
