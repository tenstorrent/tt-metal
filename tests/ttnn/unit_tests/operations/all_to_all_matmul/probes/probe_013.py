import torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(1234)
for K in (4096, 128):
    a = torch.randn(1, 16, 32, K).to(torch.bfloat16)
    w = (torch.randn(1, 16, K, 128) / K**0.5).to(torch.bfloat16)
    want = a.float() @ w.float()
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
    tw = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev)
    for fp32 in (True, False):
        cfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        got = ttnn.to_torch(ttnn.matmul(ta, tw, compute_kernel_config=cfg)).float()
        mask = want.abs() > 1e-2 * want.std()
        r = got[mask] / want[mask]
        rms = ((got - want).pow(2).mean().sqrt() / want.std()).item()
        print(
            f"RES K={K} fp32={fp32} median={r.median().item():.4f} p5={r.quantile(0.05).item():.3f} p95={r.quantile(0.95).item():.3f} relrms={rms:.4f}"
        )
ttnn.close_device(dev)
