import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:

    def report(tag, got, exp):
        g = got.float().flatten()
        t = exp.float().flatten()
        keep = t.abs() > 1e-3 * t.abs().max()
        r = g[keep] / t[keep]
        rel_rms = ((g - t).pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item()
        pcc = torch.corrcoef(torch.stack([g, t]))[0, 1].item()
        print(f"CASE {tag}")
        print(f"   rel_rms={rel_rms:.5f} pcc={pcc:.6f} r_med={r.median().item():.6f} r_std={r.std().item():.3e}")

    # ONE dispatch per fresh program: randn, W=64, float32, TILE, two epsilons
    for eps in (1e-12, 1e-5):
        torch.manual_seed(0)
        x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
        exp = torch_rms_norm_ttnn(x, epsilon=eps)
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=eps))
        report(f"randn W=64 fp32 eps={eps}", got, exp)
    # the debug pattern, isolated
    x = torch.zeros(1, 1, 32, 64, dtype=torch.float32)
    x[..., 0] = 1.0
    for eps in (1e-12, 1e-5):
        exp = torch_rms_norm_ttnn(x, epsilon=eps)
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=eps))
        print(
            f"CASE sparse W=64 fp32 eps={eps}: hot={got[...,0].flatten()[0].item():.5f} expect={exp[...,0].flatten()[0].item():.5f}"
        )
    # bf16 control on the SAME sparse pattern
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12))
    print(f"CASE sparse W=64 bf16 eps=1e-12: hot={got[...,0].flatten()[0].item():.5f} expect=8.0")
finally:
    ttnn.close_device(device)
