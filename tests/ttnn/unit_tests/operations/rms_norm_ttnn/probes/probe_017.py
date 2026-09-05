import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:

    def metrics(got, exp):
        g = got.to(torch.float32).flatten()
        t = exp.to(torch.float32).flatten()
        err = (g - t).abs()
        rms_std = (err.pow(2).mean().sqrt() / t.std()).item()  # eval.metrics normalization
        rms = (err.pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item()
        pcc = torch.corrcoef(torch.stack([g, t]))[0, 1].item()
        return rms, rms_std, pcc

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
                ("ACC(dest16)", None),
                (
                    "ACC(dest32)",
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
                    ),
                ),
            ]:
                tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                got = ttnn.to_torch(rms_norm_ttnn(tx, compute_kernel_config=cfg))
                rms, rms_std, pcc = metrics(got, exp)
                print(f"{name} {str(shape):16s} {label:12s} rms={rms:.5f} rms_vs_std={rms_std:.5f} pcc={pcc:.6f}")
    # bf16 must be untouched by the flag
    torch.manual_seed(0)
    x = torch.randn(1, 1, 128, 512).to(torch.bfloat16)
    exp = torch_rms_norm_ttnn(x, epsilon=1e-12)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm_ttnn(tx))
    rms, rms_std, pcc = metrics(got, exp)
    print(f"bf16 control                        rms={rms:.5f} rms_vs_std={rms_std:.5f} pcc={pcc:.6f}")
finally:
    ttnn.close_device(device)
