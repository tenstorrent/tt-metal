import torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
K = 64
a = torch.randn(64, K).bfloat16()
w = torch.randn(K, 32).bfloat16()
ex = a.double() @ w.double()
for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi):
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
    )
    ad = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    wd = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(ttnn.matmul(ad, wd, compute_kernel_config=cfg, dtype=ttnn.float32)).float()
    bits = o.view(torch.int32) & 0x1FFF
    print(
        fid,
        "low13 zero frac",
        (bits == 0).float().mean().item(),
        "low8 zero",
        ((o.view(torch.int32) & 0xFF) == 0).float().mean().item(),
    )
    d = o.double() - ex
    print("rel", (d.norm() / ex.norm()).item(), "mean signed rel", ((d * ex.sign()).mean() / ex.abs().mean()).item())
    # one dot product, exact products: single row-col, inspect
    print("sample", o[0, :4].tolist(), ex[0, :4].tolist())
ttnn.close_device(dev)
