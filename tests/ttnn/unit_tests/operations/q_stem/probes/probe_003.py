import torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(42)
for K in (1536, 4096):
    x = torch.randn(128, K).bfloat16()
    w = (torch.randn(K, 768) / K**0.5).bfloat16()
    for fp32 in (False, True):
        for kbs in (None,):
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
            wd = ttnn.to_torch(tw).float()
            cfg = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32, math_approx_mode=False
            )
            got = ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=cfg)).float()
            exp = x.float() @ wd
            m = exp.abs() > 0.01 * exp.std()
            r = got[m] / exp[m]
            q = torch.quantile(r.flatten()[: 1 << 20], torch.tensor([0.05, 0.5, 0.95]))
            rms = ((got - exp).pow(2).mean().sqrt() / exp.std()).item()
            print(f"RATIO K={K} fp32={fp32} p5/med/p95={q.tolist()} rel_rms={rms:.4f}")
ttnn.close_device(dev)
